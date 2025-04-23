import os
import math
import torch
import json
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
    
    def _normalize_pcl(self, pcl):
        # min_pos = np.array([-0.75, -0.75, -0.75])
        # max_pos = np.array([0.75, 0.75, 0.75])
        min_pos = np.array([-0.058, -0.053, -0.043])
        max_pos = np.array([0.062, 0.062, 0.031])
        # Normalize the point cloud to be between -1 and 1
        pcl = (pcl - min_pos) / (max_pos - min_pos)
        pcl = 2 * pcl - 1
        return pcl

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

        # # normalize the state point clouds to be between -1 and 1
        # state = state / np.max(np.abs(state))
        # next_state = next_state / np.max(np.abs(next_state))
        # goal = goal / np.max(np.abs(goal))
        state = self._normalize_pcl(state)
        next_state = self._normalize_pcl(next_state)
        goal = self._normalize_pcl(goal)

        # convert to torch tensors of correct shape/format
        state = torch.from_numpy(state).float()
        next_state = torch.from_numpy(next_state).float()
        goal = torch.from_numpy(goal).float()
        # state_idx = torch.tensor(state_idx).unsqueeze(0).float()

        # NOTE: we need to standardize the min/max point cloud sizes for reconstruction consistency

        state_idx = torch.tensor(state_idx).unsqueeze(0).float()
        
        return state, next_state, goal, state_idx

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
        # center = np.mean(state1, axis=0)
        center = np.array([0.628, 0.000, 0.104])

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

class SubGoalQualityDataset:
    def __init__(self, model_name, sub_goal_step=3, n_runs = 5, train=True):
        self.n_runs = n_runs
        self.train = train
        self.sub_goal_step = sub_goal_step
        self.model_name = model_name
        self.subgoal_data_path = '/home/alison/Documents/GitHub/subgoal_diffusion/subgoal_evals/' + model_name 
        self.state_data_path = '/home/alison/Documents/Feb26_Human_Demos_Raw/pottery'
        if self.train:
            self.data_dict = {'/Trajectory0': 33, 
                              '/Trajectory1': 27, 
                              '/Trajectory2': 26, 
                              '/Trajectory3': 26}
        else:
            self.data_dict = {'/Trajectory4': 17, 
                              '/Trajectory5': 23}
            
        self.dataloding_dict = self._get_dataloading_dict()

    def _get_dataloading_dict(self):
        dataloading_dict = {}
        idx = 0
        # iterate through trajectories
        # for value in self.data_dict.values():
        for traj, value in self.data_dict.items():
            # iterate through states of the trajectory
            for i in range(0, value - (value % self.sub_goal_step) - 2 - self.sub_goal_step, self.sub_goal_step):
                for run in range(self.n_runs):
                    # do the process twice for single-step and auto-regressive
                    state_idx = i 
                    goal_idx = value - (value % self.sub_goal_step) - self.sub_goal_step
                    subgoal_idx = i 
                    single_step = True
                    dataloading_dict[idx] = [traj, state_idx, goal_idx, subgoal_idx, run, single_step]
                    idx += 1

                    if i != 0:
                        single_step = False
                        dataloading_dict[idx] = [traj, state_idx, goal_idx, subgoal_idx, run, single_step]
                        idx += 1
        return dataloading_dict
    
    def load_dist_data(self, txt_file_path):
        '''
        Read in a txt file which is a disctionary into a python dictionary, but the entire dictionary is on one line.
        The dictionary contains the following keys: 'CD', 'EMD', 'HAUSDORFF'.
        '''
        # load the data
        for line in open(txt_file_path, 'r'):
            # replace single quotes with double quotes
            line = line.replace("'", '"')
            data = json.loads(line)

        cd = data['CD']
        emd = data['EMD']
        hd = data['HAUSDORFF']
        return cd, emd, hd
    
    def generate_value(self, cd, emd):
        '''
        Generate a value between 0 and 1 from the cd and emd values.
        '''
        combined = 0.5 * cd + 0.5 * emd
        min_val = 0.002
        max_val = 0.05
        if combined <= min_val:
            value = 1
        elif combined >= max_val:
            value = 0
        else:
            value = 1 - (combined - min_val) / (max_val - min_val)
        return value

    def __len__(self):
        data_list1 = self.n_runs * [len(np.arange(0, value - (value % self.sub_goal_step) - 2 - self.sub_goal_step, self.sub_goal_step)) for value in self.data_dict.values()]
        data_list2 = self.n_runs * [len(np.arange(0, value - (value % self.sub_goal_step) - 2 - self.sub_goal_step, self.sub_goal_step)) - 1 for value in self.data_dict.values()]
        return sum(data_list1) + sum(data_list2)

    def __getitem__(self, idx):
        traj = self.dataloding_dict[idx][0]
        state_idx = self.dataloding_dict[idx][1]
        goal_idx = self.dataloding_dict[idx][2]
        subgoal_idx = self.dataloding_dict[idx][3]
        run = self.dataloding_dict[idx][4]
        single_step = self.dataloding_dict[idx][5]
        
        # load in the point clouds
        state = np.load(self.state_data_path + traj + '/unnormalized_pointcloud' + str(state_idx) + '.npy')
        goal = np.load(self.state_data_path + traj + '/unnormalized_pointcloud' + str(goal_idx) + '.npy')
        if single_step:
            subgoal = np.load(self.subgoal_data_path + traj + '/Run' + str(run) + '/one_step_subgoal' + str(subgoal_idx) + '.npy')
            dict_path = self.subgoal_data_path + traj + '/Run' + str(run) + '/single_step_dist_metrics_' + str(subgoal_idx) + '.txt'
        else:
            subgoal = np.load(self.subgoal_data_path + traj + '/Run' + str(run) + '/autoregressive_subgoal' + str(subgoal_idx) + '.npy')
            dict_path = self.subgoal_data_path + traj + '/Run' + str(run) + '/autoregressive_dist_metrics_' + str(subgoal_idx) + '.txt'
        
        # downsample the point clouds
        if state.shape[0] > 2048:
            state = state[np.random.choice(state.shape[0], 2048, replace=False), :]
        if goal.shape[0] > 2048: 
            goal = goal[np.random.choice(goal.shape[0], 2048, replace=False), :]
        if subgoal.shape[0] > 2048:
            subgoal = subgoal[np.random.choice(subgoal.shape[0], 2048, replace=False), :]
        
        # process the point clouds
        # center = np.mean(state, axis=0)
        center = np.array([0.628, 0.000, 0.104])
        state = state - center
        goal = goal - center
        subgoal = subgoal - center
        state = state / np.max(np.abs(state))
        goal = goal / np.max(np.abs(goal))
        subgoal = subgoal / np.max(np.abs(subgoal))

        # convert to torch tensors
        state = torch.from_numpy(state).float()
        goal = torch.from_numpy(goal).float()
        subgoal = torch.from_numpy(subgoal).float()
        
        # load in the distance dictionary
        cd, emd, _ = self.load_dist_data(dict_path)
        value = self.generate_value(cd, emd)
        value = torch.tensor(value).unsqueeze(0).float()

        return state, goal, subgoal, value