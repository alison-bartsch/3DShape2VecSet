import os
import math
import torch
import numpy as np
from scipy.spatial.transform import Rotation

class ActionPredDataset:
    def __init__(self, dataset_dir, trajectory_list, rot_aug):
        self.dataet_dir = dataset_dir
        self.trajectory_list = trajectory_list
        self.aug_step = rot_aug
        self.datapoint_dict = self._get_state_permutations(dataset_dir, trajectory_list)
        self.total_raw_data = len(self.datapoint_dict)
        self.n_datapoints_per_trajectory = (360 / self.aug_step) * self.total_raw_data

        # note: trajectory list is a list of indices corresponding to what trajectories to load (for )

    def _get_state_permutations(self, dataset_dir, trajectory_list):
        datapoint_dict = {}
        idx = 0
        for traj in trajectory_list:
            i = 0
            path = dataset_dir + '/Trajectory' + traj + '/unnormalized_pointcloud' + str(i) + '.npy'
            # while the state path exists:
            while os.path.exists(path):
                i += 1
                path = dataset_dir + '/Trajectory' + traj + '/unnormalized_pointcloud' + str(i) + '.npy'
                
            # iterate to get all posible 2-state pairs with i states
            for j in range(i):
                for k in range(j+1, i):
                    datapoint_dict[idx] = [traj, j, k] 
                    idx += 1
        return datapoint_dict

    def _rotate_pcl(self, state, center, rot):
        '''
        Faster implementation of rotation augmentation to fix slow down issue
        '''
        state = state - center
        R = Rotation.from_euler('xyz', np.array([0, 0, rot]), degrees=True).as_matrix()
        state = R @ state.T
        pcl_aug = state.T + center
        return pcl_aug
        
    def __getitem__(self, idx):
        raw_idx = int(idx // self.n_datapoints_per_trajectory) 
        aug_rot = (idx % self.n_datapoints_per_trajectory) * self.aug_step

        traj_idx, state1_idx, state2_idx = self.datapoint_dict[raw_idx]
        
        # load in the data
        state1 = np.load(self.dataset_dir + '/Trajectory' + traj_idx + '/unnormalized_pointcloud' + str(state1_idx) + '.npy')
        state2 = np.load(self.dataset_dir + '/Trajectory' + traj_idx + '/unnormalized_pointcloud' + str(state2_idx) + '.npy')
        n_actions = math.abs(state1_idx - state2_idx)

        # get center of state1
        center = np.mean(state1, axis=0)

        # rotate the states
        state1 = self._rotate_pcl(state1, center, aug_rot)
        state2 = self._rotate_pcl(state2, center, aug_rot)

        # convert to torch tensors of correct shape/format
        state1 = torch.from_numpy(state1).float()
        state2 = torch.from_numpy(state2).float()
        n_actions = torch.tensor(n_actions).float()
        
        return state1, state2, n_actions

class ReconDataset:
    pass

# get number of states in each trajectory
    # we know the # of state pairs per trajectory = permutation of the # of states

# dataloader needs to know the # of trajectories and the # of states per trajectory and the rotation augmentation degree