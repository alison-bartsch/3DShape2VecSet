import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.utils.data as data
import matplotlib.pyplot as plt
import models_ae
import models_class_cond
from real_world_dataset import SubGoalDataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler

dataset_dir = '/home/alison/Documents/Feb26_Human_Demos_Raw/pottery'
save_dir = '/home/alison/Documents/GitHub/subgoal_diffusion/model_weights/'
# exp_folder = 'latent_subgoal_3_state_idx_global_pcl_normalization_fixed'

ablation_list = [1,5,7]
for elem in ablation_list:

    exp_folder = 'latent_subgoal_' + str(elem) + '_state_idx_global_pcl_normalization_fixed'
    os.makedirs(save_dir + exp_folder)

    # parameters
    n_epochs = 500 # 200 # 800
    n_subgoal_steps = elem # 3
    lr_param = 1e-5
    batch = 4
    state_idx_conditioning = True
    global_normalization = True

    # create and save dictionary with parameters
    params = {
        'n_epochs': n_epochs,
        'n_subgoal_steps': n_subgoal_steps,
        'lr_param': lr_param,
        'batch': batch,
        'state_idx_conditioning': state_idx_conditioning,
        'global_normalization': global_normalization,
    }
    # write params to text file
    with open(save_dir + exp_folder + '/params.txt', 'w') as f:
        for key, value in params.items():
            f.write(f'{key}: {value}\n')

    # create datasets and dataloaders for train/test
    train_dataset = SubGoalDataset(dataset_dir, [0,1,2,3], 30, n_subgoal_steps)
    test_dataset = SubGoalDataset(dataset_dir, [4,5], 30, n_subgoal_steps)
    train_loader = data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch, shuffle=True)

    # load in the pretrained embedding model
    ae_pth = '/home/alison/Downloads/checkpoint-199.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ae = models_ae.__dict__['kl_d512_m512_l8']()
    ae.eval()
    ae.load_state_dict(torch.load(ae_pth, map_location='cpu')['model'])
    ae.to(device)

    model = models_class_cond.__dict__['kl_d512_m512_l8_edm']()
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=lr_param)
    # loss_scaler = NativeScaler()
    loss_scaler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.1)

    criterion = models_class_cond.__dict__['EDMLoss']()

    print("criterion = %s" % str(criterion))

    prev_best_test_loss = float('inf')
    for epoch in tqdm(range(n_epochs)):
        # train for one epoch
        model.train(True)
        ae.eval()
        train_loss = 0
        for state, next_state, goal, state_idx in tqdm(train_loader):
            state = state.to(device)
            next_state = next_state.to(device)
            goal = goal.to(device)
            state_idx = state_idx.to(device)

            with torch.cuda.amp.autocast(enabled=False):
                with torch.no_grad():

                    _, state_latent = ae.encode(state)
                    _, next_state_latent = ae.encode(next_state)
                    _, goal_latent = ae.encode(goal)

                loss = criterion(model, next_state_latent, state_latent, goal_latent, state_idx)

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_scaler.step()
        train_loss /= len(train_dataset)
        print(f'Epoch: {epoch}, Scaled Train Loss: {train_loss}')

        if epoch % 5 == 0 or epoch + 1 == n_epochs:
            model.eval()
            ae.eval()
            test_loss = 0
            for state, next_state, goal, state_idx in tqdm(test_loader):
                state = state.to(device)
                next_state = next_state.to(device)
                goal = goal.to(device)
                state_idx = state_idx.to(device)

                with torch.cuda.amp.autocast(enabled=False):
                    with torch.no_grad():
                        _, state_latent = ae.encode(state)
                        _, next_state_latent = ae.encode(next_state)
                        _, goal_latent = ae.encode(goal)

                    loss = criterion(model, next_state_latent, state_latent, goal_latent, state_idx)

                test_loss += loss.item()
            test_loss /= len(test_dataset)
            print(f'Scaled Test Loss: {test_loss}')

            # if epoch % 2 == 0:
            #     # save the model weights
            #     torch.save(model.state_dict(), save_dir + exp_folder + '/diffusion_model' + str(epoch) + '.pt')

            if test_loss < prev_best_test_loss:
                prev_best_test_loss = test_loss
                torch.save(model.state_dict(), save_dir + exp_folder + '/best_test_loss_diffusion_model.pt')

            # save the test loss to .txt file
            with open(save_dir + exp_folder + '/test_loss.txt', 'a') as f:
                f.write(f'Epoch: {epoch}, Test Loss: {test_loss}\n')
                f.write(f'Epoch: {epoch}, Train Loss: {train_loss}\n')