import torch
import numpy as np
from types import SimpleNamespace

def loss_weighted_rep_no_scale(p2d, p3d, confs, state):
    # the weighted reprojection loss as defined in Equation 5
    p2d=p2d.reshape(-1,36)
    p3d=p3d.reshape(-1,54)
    confs=confs.reshape(-1,18)
    # normalize by scale
    scale_p2d = torch.sqrt(p2d[:, 0:36].square().sum(axis=1, keepdim=True) / 36)
    p2d_scaled = p2d[:, 0:36]/scale_p2d

    # only the u,v coordinates are used and depth is ignored
    # this is a simple weak perspective projection
    scale_p3d = torch.sqrt(p3d[:, 0:36].square().sum(axis=1, keepdim=True) / 36)
    p3d_scaled = p3d[:, 0:36]/scale_p3d
    if state=="train":
        loss = ((p2d_scaled - p3d_scaled).abs().reshape(-1, 2, 18).sum(axis=1) * confs).sum() / (p2d_scaled.shape[0] * p2d_scaled.shape[1])
    elif state=="test":
        loss = ((p2d_scaled - p3d_scaled).abs().reshape(-1, 2, 18).sum(axis=1)).sum() / (p2d_scaled.shape[0] * p2d_scaled.shape[1])
    return loss


def calculate_loss(inp_poses,rot_poses,inp_confidences,pred_poses,pred_rot,state="train"):
    losses = SimpleNamespace()
    B=inp_poses.shape[0]
    T1=inp_poses.shape[1]
    # reprojection loss
    losses.rep = loss_weighted_rep_no_scale(inp_poses, rot_poses, inp_confidences, state)

    # view-consistency
    # to compute the different losses we need to do some reshaping
    pred_poses_rs = pred_poses.reshape((B,T1,54)).reshape((-1, 3, T1, 54))
    pred_rot_rs = pred_rot.reshape((B,T1,3,3)).reshape(-1, 3, T1, 3, 3)
    confidences_rs = inp_confidences.reshape((B,T1,18)).reshape(-1, 3, T1, 18)
    inp_poses_rs = inp_poses.reshape((B,T1,36)).reshape(-1, 3, T1, 36)

    # view and camera consistency are computed in the same loop
    losses.view = 0
    #losses.camera = 0
    all_cams = ['cam0', 'cam1', 'cam2']
    for c_cnt in range(len(all_cams)):
        ## view consistency
        # get all cameras and active cameras
        ac = np.array(range(len(all_cams)))
        coi = np.delete(ac, c_cnt)

        # view consistency
        projected_to_other_cameras = pred_rot_rs[:, coi].matmul(pred_poses_rs.reshape(-1, len(all_cams),T1, 3, 18)[:, c_cnt:c_cnt+1].repeat(1, len(all_cams)-1, 1, 1, 1)).reshape(-1, len(all_cams)-1, T1, 54)
        
        losses.view += loss_weighted_rep_no_scale(inp_poses_rs[:, coi].reshape(-1, T1, 36),
                                                projected_to_other_cameras.reshape(-1, T1, 54),
                                                confidences_rs[:, coi].reshape(-1, T1, 18),
                                                state)
    

    losses.loss = losses.rep + losses.view 
    #losses.loss = losses.rep
    return losses



