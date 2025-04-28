import numpy as np
import json
from tqdm import tqdm

def load_dist_data(txt_file_path):
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

# model_eval_list = [('latent_subgoal_1_subgoal_steps', 1), 
#                    ('latent_subgoal_3_subgoal_steps', 3), 
#                    ('latent_subgoal_more_epochs', 5),
#                    ('latent_subgoal_7_subgoal_steps', 7)]

model_eval_list = [('latent_subgoal_1_state_idx_global_pcl_normalization_fixed', 1),
                   ('latent_subgoal_3_state_idx_global_pcl_normalization_fixed', 3),
                   ('latent_subgoal_5_state_idx_global_pcl_normalization_fixed', 5),
                   ('latent_subgoal_7_state_idx_global_pcl_normalization_fixed', 7)]

traj_list = [('/Trajectory0', 33), 
            ('/Trajectory1', 27), 
            ('/Trajectory2', 26), 
            ('/Trajectory3', 26), 
            ('/Trajectory4', 17), 
            ('/Trajectory5', 23)]

pred_n = 5

# iterate through each model
for model, n_steps in tqdm(model_eval_list):
    single_step_test_cds = []
    single_step_best_test_cds = []
    single_step_train_cds = []
    single_step_best_train_cds = []

    single_step_test_emds = []
    single_step_best_test_emds = []
    single_step_train_emds = []
    single_step_best_train_emds = []

    single_step_test_hds = []
    single_step_best_test_hds = []
    single_step_train_hds = []
    single_step_best_train_hds = []

    autoregressive_test_cds = []
    autoregressive_best_test_cds = []
    autoregressive_train_cds = []
    autoregressive_best_train_cds = []

    autoregressive_test_emds = []
    autoregressive_best_test_emds = []
    autoregressive_train_emds = []
    autoregressive_best_train_emds = []

    autoregressive_test_hds = []
    autoregressive_best_test_hds = []
    autoregressive_train_hds = []
    autoregressive_best_train_hds = []
    # iterate through the traj list
    for traj, n_states in traj_list:
        # iterate through the subgoals
        for i in range(0, n_states - (n_states % n_steps) - 2 - n_steps, n_steps):
            best_test_autoregressive_cd = 1000000
            best_test_single_step_cd = 1000000
            best_test_autoregressive_emd = 1000000
            best_test_single_step_emd = 1000000
            best_test_autoregressive_hd = 1000000
            best_test_single_step_hd = 1000000

            best_train_autoregressive_cd = 1000000
            best_train_single_step_cd = 1000000
            best_train_autoregressive_emd = 1000000
            best_train_single_step_emd = 1000000
            best_train_autoregressive_hd = 1000000
            best_train_single_step_hd = 1000000
            # iterate through the pred_n runs
            # for run in range(pred_n):
                # load the data
                # base_path = '/home/alison/Documents/GitHub/subgoal_diffusion/subgoal_evals/autoregressive_vis/' + model + traj + '/Run' + str(run) + '/'
            base_path = '/home/alison/Documents/GitHub/subgoal_diffusion/subgoal_evals/autoregressive_vis/' + model + traj + '/'
            single_step_load_path = base_path + 'single_step_dist_metrics_' + str(i) + '.txt'
            single_cd, single_emd, single_hd = load_dist_data(single_step_load_path)
            # if i > 0:
            #     autoregressive_load_path = base_path + 'autoregressive_dist_metrics_' + str(i) + '.txt'
            #     autoregressive_cd, autoregressive_emd, autoregressive_hd = load_dist_data(autoregressive_load_path)

            # add all data to train/test lists
            if traj == '/Trajectory4' or traj == '/Trajectory5':
                single_step_test_cds.append(single_cd)
                single_step_test_emds.append(single_emd)
                single_step_test_hds.append(single_hd)

                # update best numbers
                if single_cd < best_test_single_step_cd:
                    best_test_single_step_cd = single_cd
                if single_emd < best_test_single_step_emd:
                    best_test_single_step_emd = single_emd
                if single_hd < best_test_single_step_hd:
                    best_test_single_step_hd = single_hd

                # if i > 0:
                #     autoregressive_test_cds.append(autoregressive_cd)
                #     autoregressive_test_emds.append(autoregressive_emd)
                #     autoregressive_test_hds.append(autoregressive_hd)
                #     if autoregressive_cd < best_test_autoregressive_cd:
                #         best_test_autoregressive_cd = autoregressive_cd
                #     if autoregressive_emd < best_test_autoregressive_emd:
                #         best_test_autoregressive_emd = autoregressive_emd
                #     if autoregressive_hd < best_test_autoregressive_hd:
                #         best_test_autoregressive_hd = autoregressive_hd
            else:
                single_step_train_cds.append(single_cd)
                single_step_train_emds.append(single_emd)
                single_step_train_hds.append(single_hd)
                
                
                # update best numbers
                if single_cd < best_train_single_step_cd:
                    best_train_single_step_cd = single_cd
                if single_emd < best_train_single_step_emd:
                    best_train_single_step_emd = single_emd
                if single_hd < best_train_single_step_hd:
                    best_train_single_step_hd = single_hd
                
                # if i > 0:
                #     autoregressive_train_cds.append(autoregressive_cd)
                #     autoregressive_train_emds.append(autoregressive_emd)
                #     autoregressive_train_hds.append(autoregressive_hd)
                #     if autoregressive_cd < best_train_autoregressive_cd:
                #         best_train_autoregressive_cd = autoregressive_cd
                #     if autoregressive_emd < best_train_autoregressive_emd:
                #         best_train_autoregressive_emd = autoregressive_emd
                #     if autoregressive_hd < best_train_autoregressive_hd:
                #         best_train_autoregressive_hd = autoregressive_hd

        if traj == '/Trajectory4' or traj == '/Trajectory5':
            single_step_best_test_cds.append(best_test_single_step_cd)
            single_step_best_test_emds.append(best_test_single_step_emd)
            single_step_best_test_hds.append(best_test_single_step_hd)
            

            # if i > 0:
            #     autoregressive_best_test_cds.append(best_test_autoregressive_cd)
            #     autoregressive_best_test_emds.append(best_test_autoregressive_emd)
            #     autoregressive_best_test_hds.append(best_test_autoregressive_hd)
        
        else:
            single_step_best_train_cds.append(best_train_single_step_cd)
            single_step_best_train_emds.append(best_train_single_step_emd)
            single_step_best_train_hds.append(best_train_single_step_hd)

            # if i > 0:
            #     autoregressive_best_train_cds.append(best_train_autoregressive_cd)
            #     autoregressive_best_train_emds.append(best_train_autoregressive_emd)
            #     autoregressive_best_train_hds.append(best_train_autoregressive_hd)
                    

    # process all the data --> report mean and std for each list for each model
    print("\n\n--------MODEL: " + model + "--------")
    print("\nSingle Step Test CDS: " + str(np.mean(single_step_test_cds)) + " +/- " + str(np.std(single_step_test_cds)))
    print("Single Step Best Test CDS: " + str(np.mean(single_step_best_test_cds)) + " +/- " + str(np.std(single_step_best_test_cds)))
    print("Single Step Train CDS: " + str(np.mean(single_step_train_cds)) + " +/- " + str(np.std(single_step_train_cds)))
    print("Single Step Best Train CDS: " + str(np.mean(single_step_best_train_cds)) + " +/- " + str(np.std(single_step_best_train_cds)))
    print("Single Step Test EMDS: " + str(np.mean(single_step_test_emds)) + " +/- " + str(np.std(single_step_test_emds)))
    print("Single Step Best Test EMDS: " + str(np.mean(single_step_best_test_emds)) + " +/- " + str(np.std(single_step_best_test_emds)))
    print("Single Step Train EMDS: " + str(np.mean(single_step_train_emds)) + " +/- " + str(np.std(single_step_train_emds)))
    print("Single Step Best Train EMDS: " + str(np.mean(single_step_best_train_emds)) + " +/- " + str(np.std(single_step_best_train_emds)))
    print("Single Step Test HDS: " + str(np.mean(single_step_test_hds)) + " +/- " + str(np.std(single_step_test_hds)))
    print("Single Step Best Test HDS: " + str(np.mean(single_step_best_test_hds)) + " +/- " + str(np.std(single_step_best_test_hds)))
    print("Single Step Train HDS: " + str(np.mean(single_step_train_hds)) + " +/- " + str(np.std(single_step_train_hds)))
    print("Single Step Best Train HDS: " + str(np.mean(single_step_best_train_hds)) + " +/- " + str(np.std(single_step_best_train_hds)))

    # print("\nAutoregressive Test CDS: " + str(np.mean(autoregressive_test_cds)) + " +/- " + str(np.std(autoregressive_test_cds)))
    # print("Autoregressive Best Test CDS: " + str(np.mean(autoregressive_best_test_cds)) + " +/- " + str(np.std(autoregressive_best_test_cds)))
    # print("Autoregressive Train CDS: " + str(np.mean(autoregressive_train_cds)) + " +/- " + str(np.std(autoregressive_train_cds)))
    # print("Autoregressive Best Train CDS: " + str(np.mean(autoregressive_best_train_cds)) + " +/- " + str(np.std(autoregressive_best_train_cds)))
    # print("Autoregressive Test EMDS: " + str(np.mean(autoregressive_test_emds)) + " +/- " + str(np.std(autoregressive_test_emds)))
    # print("Autoregressive Best Test EMDS: " + str(np.mean(autoregressive_best_test_emds)) + " +/- " + str(np.std(autoregressive_best_test_emds)))
    # print("Autoregressive Train EMDS: " + str(np.mean(autoregressive_train_emds)) + " +/- " + str(np.std(autoregressive_train_emds)))
    # print("Autoregressive Best Train EMDS: " + str(np.mean(autoregressive_best_train_emds)) + " +/- " + str(np.std(autoregressive_best_train_emds)))
    # print("Autoregressive Test HDS: " + str(np.mean(autoregressive_test_hds)) + " +/- " + str(np.std(autoregressive_test_hds)))
    # print("Autoregressive Best Test HDS: " + str(np.mean(autoregressive_best_test_hds)) + " +/- " + str(np.std(autoregressive_best_test_hds)))
    # print("Autoregressive Train HDS: " + str(np.mean(autoregressive_train_hds)) + " +/- " + str(np.std(autoregressive_train_hds)))
    # print("Autoregressive Best Train HDS: " + str(np.mean(autoregressive_best_train_hds)) + " +/- " + str(np.std(autoregressive_best_train_hds)))
    print("\n\n----------------------------------") 
    print("\n\n")
