import os
import math
import torch
import numpy as np
from scipy.spatial.transform import Rotation

class ActionPredDataset:
    def __init__(self, dataset_dir, trajectory_list, rot_aug):
        self.dataset_dir = dataset_dir
        self.trajectory_list = trajectory_list
        self.aug_step = rot_aug
        self.datapoint_dict = self._get_state_permutations(dataset_dir, trajectory_list)
        self.total_raw_data = len(self.datapoint_dict)
        self.n_datapoints = int(360 / self.aug_step) * self.total_raw_data

    def _get_state_permutations(self, dataset_dir, trajectory_list):
        datapoint_dict = {}
        idx = 0
        for traj in trajectory_list:
            i = 0
            path = dataset_dir + '/Trajectory' + str(traj) + '/unnormalized_pointcloud' + str(i) + '.npy'
            # while the state path exists:
            while os.path.exists(path):
                i += 1
                path = dataset_dir + '/Trajectory' + str(traj) + '/unnormalized_pointcloud' + str(i) + '.npy'
                
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

    def __len__(self):
        return self.n_datapoints
        
    def __getitem__(self, idx):
        # raw_idx = int(idx // self.n_datapoints) 
        # print("\nRaw index: ", raw_idx)
        # aug_rot = (idx % self.n_datapoints) * self.aug_step
        # print("Augmentation rotation: ", aug_rot)

        raw_idx = int(idx // int(360 / self.aug_step)) # needs to convert the idx is from n_datapoints to total raw data
        aug_rot = ((idx % self.total_raw_data) * self.aug_step) % 360 # needs to be a scalar number times the self.aug_step to be the full rotation to apply given the index
            # for each datapoint, apply rot_aug (360 degrees / self.aug_step) times

        traj_idx, state1_idx, state2_idx = self.datapoint_dict[raw_idx]
        
        # load in the data
        state1 = np.load(self.dataset_dir + '/Trajectory' + str(traj_idx) + '/unnormalized_pointcloud' + str(state1_idx) + '.npy')
        state2 = np.load(self.dataset_dir + '/Trajectory' + str(traj_idx) + '/unnormalized_pointcloud' + str(state2_idx) + '.npy')
        n_actions = np.abs(state1_idx - state2_idx)

        # get center of state1
        center = np.mean(state1, axis=0)

        # rotate the states
        state1 = self._rotate_pcl(state1, center, aug_rot)
        state2 = self._rotate_pcl(state2, center, aug_rot)

        # center the point clouds about center
        state1 -= center
        state2 -= center

        # normalize the state point clouds to be between -1 and 1
        state1 = state1 / np.max(np.abs(state1))
        state2 = state2 / np.max(np.abs(state2))

        # convert to torch tensors of correct shape/format
        state1 = torch.from_numpy(state1).float()
        state2 = torch.from_numpy(state2).float()
        n_actions = torch.tensor(n_actions).unsqueeze(0).float()
        
        return state1, state2, n_actions
    

class SubGoalDataset:
    def __init__(self, dataset_dir, trajectory_list, rot_aug, n_steps=1):
        '''
        Real world dataset for clay pottery sub-goal generation.

        Args:
            dataset_dir (str): Directory containing the dataset.
            trajectory_list (list): List of trajectory indices to use (i.e. trajectories 1,2,3).
            rot_aug (int): Rotation augmentation step size in degrees.
            n_steps (int): Number of steps between each sub-goal (default is 1).
        '''
        self.n_steps = n_steps
        self.dataset_dir = dataset_dir
        self.trajectory_list = trajectory_list
        self.aug_step = rot_aug
        self.datapoint_dict = self._get_state_goal_permutations(dataset_dir, trajectory_list)
        self.total_raw_data = len(self.datapoint_dict)
        self.n_datapoints = int(360 / self.aug_step) * self.total_raw_data

    def _get_state_goal_permutations(self, dataset_dir, trajectory_list):
        '''
        For each trajectory, we take the goal to be one of the last n_steps states in the trajectory.
        We then pair each state with the goal, stepping n_steps back in the trajectory.
        '''
        datapoint_dict = {}
        idx = 0
        for traj in trajectory_list:
            i = 0
            path = dataset_dir + '/Trajectory' + str(traj) + '/unnormalized_pointcloud' + str(i) + '.npy'
            # while the state path exists:
            while os.path.exists(path):
                i += 1
                path = dataset_dir + '/Trajectory' + str(traj) + '/unnormalized_pointcloud' + str(i) + '.npy'
            
            # i now tells us how many states are in the trajectory

            # iterate through to get all goal indices
            for g in range(self.n_steps):
                # for each goal, we need to get the state pairs (i-g is the goal index)
                # for j in range(0, i - 2*g - 1, self.n_steps):

                # descending order range for loop
                for j in range(i - g - 1 - self.n_steps, -1, -self.n_steps):
                    # we want to pair each state with the goal
                    datapoint_dict[idx] = [traj, j, i - g - 1] 
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

    def __len__(self):
        return self.n_datapoints
        
    def __getitem__(self, idx):
        raw_idx = int(idx // int(360 / self.aug_step)) # needs to convert the idx is from n_datapoints to total raw data
        aug_rot = ((idx % self.total_raw_data) * self.aug_step) % 360 # needs to be a scalar number times the self.aug_step to be the full rotation to apply given the index
            # for each datapoint, apply rot_aug (360 degrees / self.aug_step) times

        traj_idx, state_idx, goal_idx = self.datapoint_dict[raw_idx]
        
        # load in the data
        state = np.load(self.dataset_dir + '/Trajectory' + str(traj_idx) + '/unnormalized_pointcloud' + str(state_idx) + '.npy')
        next_state = np.load(self.dataset_dir + '/Trajectory' + str(traj_idx) + '/unnormalized_pointcloud' + str(state_idx + self.n_steps) + '.npy')
        goal = np.load(self.dataset_dir + '/Trajectory' + str(traj_idx) + '/unnormalized_pointcloud' + str(goal_idx) + '.npy')

        # get center of state
        center = np.mean(state, axis=0)

        # rotate the states
        state = self._rotate_pcl(state, center, aug_rot)
        next_state = self._rotate_pcl(next_state, center, aug_rot)
        goal = self._rotate_pcl(goal, center, aug_rot)

        # center the point clouds about center
        state -= center
        next_state -= center
        goal -= center

        # normalize the state point clouds to be between -1 and 1
        state = state / np.max(np.abs(state))
        next_state = next_state / np.max(np.abs(next_state))
        goal = goal / np.max(np.abs(goal))

        # convert to torch tensors of correct shape/format
        state = torch.from_numpy(state).float()
        next_state = torch.from_numpy(next_state).float()
        goal = torch.from_numpy(goal).float()
        
        return state, next_state, goal

class ReconDataset:
    def __init__(self, dataset_dir, trajectory_list, rot_aug):
        self.dataset_dir = dataset_dir
        self.trajectory_list = trajectory_list
        self.aug_step = rot_aug
        self.datapoint_dict = self._get_state_permutations(dataset_dir, trajectory_list)
        self.total_raw_data = len(self.datapoint_dict)
        self.n_datapoints = int(360 / self.aug_step) * self.total_raw_data

    def _get_state_permutations(self, dataset_dir, trajectory_list):
        datapoint_dict = {}
        idx = 0
        for traj in trajectory_list:
            i = 0
            path = dataset_dir + '/Trajectory' + str(traj) + '/unnormalized_pointcloud' + str(i) + '.npy'
            # while the state path exists:
            while os.path.exists(path):
                i += 1
                path = dataset_dir + '/Trajectory' + str(traj) + '/unnormalized_pointcloud' + str(i) + '.npy'
                
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

    def __len__(self):
        return self.n_datapoints
        
    def __getitem__(self, idx):
        # raw_idx = int(idx // self.n_datapoints) 
        # print("\nRaw index: ", raw_idx)
        # aug_rot = (idx % self.n_datapoints) * self.aug_step
        # print("Augmentation rotation: ", aug_rot)

        raw_idx = int(idx // int(360 / self.aug_step)) # needs to convert the idx is from n_datapoints to total raw data
        aug_rot = ((idx % self.total_raw_data) * self.aug_step) % 360 # needs to be a scalar number times the self.aug_step to be the full rotation to apply given the index
            # for each datapoint, apply rot_aug (360 degrees / self.aug_step) times

        traj_idx, state1_idx, state2_idx = self.datapoint_dict[raw_idx]
        
        # load in the data
        state1 = np.load(self.dataset_dir + '/Trajectory' + str(traj_idx) + '/unnormalized_pointcloud' + str(state1_idx) + '.npy')
        state2 = np.load(self.dataset_dir + '/Trajectory' + str(traj_idx) + '/unnormalized_pointcloud' + str(state2_idx) + '.npy')
        n_actions = np.abs(state1_idx - state2_idx)

        # get center of state1
        center = np.mean(state1, axis=0)

        # rotate the states
        state1 = self._rotate_pcl(state1, center, aug_rot)
        state2 = self._rotate_pcl(state2, center, aug_rot)

        # center the point clouds about center
        state1 -= center
        state2 -= center

        # normalize the state point clouds to be between -1 and 1
        state1 = state1 / np.max(np.abs(state1))
        state2 = state2 / np.max(np.abs(state2))

        # convert to torch tensors of correct shape/format
        state1 = torch.from_numpy(state1).float()
        state2 = torch.from_numpy(state2).float()
        n_actions = torch.tensor(n_actions).unsqueeze(0).float()
        
        return state1, state2, n_actions

# get number of states in each trajectory
    # we know the # of state pairs per trajectory = permutation of the # of states

# dataloader needs to know the # of trajectories and the # of states per trajectory and the rotation augmentation degree