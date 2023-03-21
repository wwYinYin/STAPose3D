import numpy as np


class ChunkedGenerator:
    """
        Batched data generator, used for training.
        The sequences are split into equal-length chunks and padded as necessary.

        Arguments:
        batch_size -- the batch size to use for training
        cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
        poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
        poses_2d -- list of input 2D keypoints, one element for each video
        chunk_length -- number of output frames to predict for each training example (usually 1)
        pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
        causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
        shuffle -- randomly shuffle the dataset before each epoch
        random_seed -- initial seed to use for the random generator
        augment -- augment the dataset by flipping poses horizontally
        kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
        joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    def __init__(self, batch_size, cameras, poses_3d, poses_2d,
                 chunk_length, pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)

        # Build lineage info
        pairs = []  # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(poses_2d)):
            assert poses_3d is None or poses_3d[i][0].shape[0] == poses_2d[i][0].shape[0]
            n_chunks = (poses_2d[i][0].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d[i][0].shape[0]) // 2
            bounds = np.arange(n_chunks + 1) * chunk_length - offset
            augment_vector = np.full(len(bounds)-1, False, dtype=bool)
            pairs += zip(np.repeat(i, len(bounds)-1), bounds[:-1], bounds[1:], augment_vector)
            if augment:
                pairs += zip(np.repeat(i, len(bounds)-1), bounds[:-1], bounds[1:], ~augment_vector)

        # Initialize buffers
        if cameras is not None:
            self.batch_cam = np.empty((batch_size, cameras[0].shape[-1]))
        if poses_3d is not None:
            self.batch_3d = np.empty((batch_size, 3, chunk_length, poses_3d[0][0].shape[-2], poses_3d[0][0].shape[-1]))
        self.batch_2d = np.empty((batch_size, 3, chunk_length + 2*pad, poses_2d[0][0].shape[-2], poses_2d[0][0].shape[-1])) #3表示3个摄像头

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right

    def num_frames(self):
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):
        return self.augment

    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state

    def next_epoch(self):
        enabled = True
        while enabled:
            start_idx, pairs = self.next_pairs()
            for b_i in range(start_idx, self.num_batches):
                chunks = pairs[b_i*self.batch_size : (b_i+1)*self.batch_size]
                for i, (seq_i, start_3d, end_3d, flip) in enumerate(chunks):
                    start_2d = start_3d - self.pad - self.causal_shift
                    end_2d = end_3d + self.pad - self.causal_shift

                    # 2D poses
                    for j in range(len(self.poses_2d[seq_i])):
                        #seq_2d = self.poses_2d[seq_i]
                        one_cam=self.poses_2d[seq_i][j]
                        low_2d = max(start_2d, 0)
                        high_2d = min(end_2d, one_cam.shape[0])
                        pad_left_2d = low_2d - start_2d
                        pad_right_2d = end_2d - high_2d
                        if pad_left_2d != 0 or pad_right_2d != 0:
                            self.batch_2d[i,j] = np.pad(one_cam[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), "edge")
                        else:
                            self.batch_2d[i,j] = one_cam[low_2d:high_2d]
                    # 3D poses
                    if self.poses_3d is not None:
                        for j in range(len(self.poses_3d[seq_i])):
                            one_cam=self.poses_3d[seq_i][j]
                            low_3d = max(start_3d, 0)
                            high_3d = min(end_3d, one_cam.shape[0])
                            pad_left_3d = low_3d - start_3d
                            pad_right_3d = end_3d - high_3d
                            if pad_left_3d != 0 or pad_right_3d != 0:
                                self.batch_3d[i,j] = np.pad(one_cam[low_3d:high_3d], ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), "edge")
                            else:
                                self.batch_3d[i,j] = one_cam[low_3d:high_3d]


                if self.endless:
                    self.state = (b_i + 1, pairs)
                if self.poses_3d is None and self.cameras is None:
                    yield None, None, self.batch_2d[:len(chunks)].reshape((-1,self.batch_2d.shape[-3],self.batch_2d.shape[-2],self.batch_2d.shape[-1])) #B*3×T×N×2
                elif self.poses_3d is not None and self.cameras is None:
                    yield None, self.batch_3d[:len(chunks)].reshape((-1,self.batch_3d.shape[-3],self.batch_3d.shape[-2],self.batch_3d.shape[-1])), self.batch_2d[:(len(chunks))].reshape((-1,self.batch_2d.shape[-3],self.batch_2d.shape[-2],self.batch_2d.shape[-1])) #B*3×T×N×2
                elif self.poses_3d is None:
                    yield self.batch_cam, None, self.batch_2d[:len(chunks)]
                else:
                    yield self.batch_cam[:len(chunks)], self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]

            if self.endless:
                self.state = None
            else:
                enabled = False


