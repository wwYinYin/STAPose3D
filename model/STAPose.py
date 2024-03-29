import torch
from torchsummary import summary
import torch.nn as nn
from model.graph_attention import MultiGlobalGraph


class GraphAttentionBlock(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout):
        super(GraphAttentionBlock, self).__init__()
        
        hid_dim = output_dim
        self.relu = nn.ReLU(inplace=True)

        self.global_graph_layer = MultiGlobalGraph(adj, input_dim, input_dim//16, dropout=p_dropout)

        self.cat_conv = nn.Conv2d(2*output_dim, 2*output_dim, 1, bias=False)
        self.cat_bn = nn.BatchNorm2d(2*output_dim, momentum=0.1)

    def forward(self, x):
        # x: (B, C, T, N) --> (B, T, N, C)
        x = x.permute(0, 2, 3, 1)
        residual = x
        x_ = self.global_graph_layer(x)
        x = torch.cat((residual, x_), dim=-1)

        # x: (B, T, N, C) --> (B, C, T, N)
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.cat_bn(self.cat_conv(x)))
        return x


class SpatioTemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """

    def __init__(self, adj, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal, dropout, channels):
        super().__init__()

        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, 'Only odd filter widths are supported'

        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths

        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.pad = [filter_widths[0] // 2]
        self.init_bn = nn.BatchNorm2d(in_features, momentum=0.1)
        self.expand_bn = nn.BatchNorm2d(channels, momentum=0.1)
        self.shrink = nn.Conv2d(2**len(self.filter_widths)*channels, 3, 1, bias=False)
        self.upscale = nn.Linear(54, 1024)
        self.pose3d = nn.Linear(1024, 18*3)
        self.enc_rot = nn.Linear(1024, 3)

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2 * frames

    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames

    def forward(self, x):
        """
        X: (B, C, T, N)
            B: batchsize
            T: Temporal
            N: The number of keypoints
            C: The feature dimension of keypoints
        """

        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        #assert x.shape[-1] == self.in_features

        mean = x[:, :, 17:18, :].expand_as(x)
        input_pose_centered = x - mean
        # X: (B, T, N, C)
        x = self._forward_blocks(x, input_pose_centered)  #(B, C, T2, N)
        x = self.shrink(x)
        x=x.permute(0, 2, 1, 3) #(B, T2, 3, 18)

        B=x.shape[0]
        T=x.shape[1]
        x=x.reshape((B*T, 3, 18)) #(B*T, C, N)
        x=x.reshape((B*T,-1)) #(B*T, C*N)
        x=self.upscale(x)
        # pose path
        x_pose = self.pose3d(x) 

        # camera path
        xc = self.enc_rot(x)

        return x_pose, xc

class SpatioTemporalModelOptimized1f(SpatioTemporalModelBase):
    """
    3D pose estimation model optimized for single-frame batching, i.e.
    where batches have input length = receptive field, and output length = 1.
    This scenario is only used for training when stride == 1.

    This implementation replaces dilated convolutions with strided convolutions
    to avoid generating unused intermediate results. The weights are interchangeable
    with the reference implementation.
    """

    def __init__(self, adj, num_joints_in, in_features, filter_widths, 
                 causal=False, dropout=0.25, channels=64, num_joints_out=18):
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        """
        super().__init__(adj, num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)

        self.expand_conv = nn.Conv2d(in_features, channels, (filter_widths[0], 1), stride=(filter_widths[0], 1), bias=False)
        nn.init.kaiming_normal_(self.expand_conv.weight)

        self.cos_dis = nn.CosineSimilarity(dim=1, eps=1e-6)
        layers_tem_att = []
        layers_tem_bn = []
        self.frames = self.total_frame()

        layers_conv = []
        layers_graph_conv = []
        layers_bn = []

        layers_graph_conv.append(GraphAttentionBlock(adj, channels, channels, p_dropout=dropout))

        self.causal_shift = [(filter_widths[0] // 2) if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2) if causal else 0)

            layers_tem_att.append(nn.Linear(self.frames, self.frames // next_dilation))
            layers_tem_bn.append(nn.BatchNorm1d(self.frames // next_dilation))

            layers_conv.append(nn.Conv2d(2**i*channels, 2**i*channels, (filter_widths[i], 1), stride=(filter_widths[i], 1), bias=False))
            layers_bn.append(nn.BatchNorm2d(2**i*channels, momentum=0.1))
            layers_conv.append(nn.Conv2d(2**i*channels, 2**i*channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm2d(2**i*channels, momentum=0.1))

            layers_graph_conv.append(GraphAttentionBlock(adj, 2**i*channels, 2**i*channels, p_dropout=dropout))

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        self.layers_graph_conv = nn.ModuleList(layers_graph_conv)
        self.layers_tem_att = nn.ModuleList(layers_tem_att)
        self.layers_tem_bn = nn.ModuleList(layers_tem_bn)

    def total_frame(self):
        frames = 1
        for i in range(len(self.filter_widths)):
            frames *= self.filter_widths[i]
        return frames

    def _forward_blocks(self, x, input_2D_centered):
        input_2D_centered = input_2D_centered.view(input_2D_centered.shape[0], input_2D_centered.shape[1], -1)
        input_2D_centered = input_2D_centered.permute(0, 2, 1)
        x_target = input_2D_centered[:, :, [input_2D_centered.shape[2] // 2]]
        x_traget_matrix = x_target.expand_as(input_2D_centered)
        cos_score = self.cos_dis(x_traget_matrix, input_2D_centered)    #[192, 27]

        # x: (B, T, N, C) --> (B, C, T, N)
        x = x.permute(0, 3, 1, 2)
        x = self.init_bn(x) #[192, 3, 27, 18]
        x = self.relu(self.expand_bn(self.expand_conv(x)))  #[192, 128, 9, 18]
        x = self.layers_graph_conv[0](x)    #[192, 256, 9, 18]

        #print(self.pad)     #[1, 3, 9]
        for i in range(len(self.pad) - 1):
            res = x[:, :, self.causal_shift[i+1] + self.filter_widths[i+1]//2 :: self.filter_widths[i+1]]   #[192, 256, 3, 18]/[192, 512, 1, 18]
            t_attention = self.sigmoid(self.layers_tem_bn[i](self.layers_tem_att[i](cos_score)))            #[192, 9]/[192, 3]
            t_attention_expand0 = t_attention.unsqueeze(1)
            t_attention_expand = t_attention_expand0.unsqueeze(-1)  #[192, 1, 9, 1]/[192, 1, 3, 1]

            # x: (B, C, T, N)
            x = x * t_attention_expand  # broadcasting dot mul
            x = self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x)))
            x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))       #[192, 256, 3, 18]/[192, 512, 1, 18]

            x = self.layers_graph_conv[i+1](x)      #[192, 512, 3, 18]/[192, 1024, 1, 18]

        return x


