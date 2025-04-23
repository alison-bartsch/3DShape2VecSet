import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.utils.data as data
import matplotlib.pyplot as plt
import models_ae
import models_class_cond
from real_world_dataset import SubGoalQualityDataset
from action_pred_model import SubGoalValueModel, SubGoalValueModelCrossAttn

save_dir = '/home/alison/Documents/GitHub/subgoal_diffusion/model_weights/'
exp_folder = 'subgoal_new_value_model'
os.makedirs(save_dir + exp_folder)

# parameters
n_epochs = 500
n_subgoal_steps = 3
lr_param = 1e-4
batch = 4
model_name = 'latent_subgoal_3_subgoal_steps'

# create and save dictionary with parameters
params = {
    'n_epochs': n_epochs,
    'n_subgoal_steps': n_subgoal_steps,
    'lr_param': lr_param,
    'batch': batch,
    'model_name': model_name,
}
# write params to text file
with open(save_dir + exp_folder + '/params.txt', 'w') as f:
    for key, value in params.items():
        f.write(f'{key}: {value}\n')

# create datasets and dataloaders for train/test
train_dataset = SubGoalQualityDataset(model_name, sub_goal_step=3, n_runs = 5, train=True)
test_dataset = SubGoalQualityDataset(model_name, sub_goal_step=3, n_runs = 5, train=False)
train_loader = data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=batch, shuffle=True)

# load in the pretrained embedding model
ae_pth = '/home/alison/Downloads/checkpoint-199.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ae = models_ae.__dict__['kl_d512_m512_l8']()
ae.eval()
ae.load_state_dict(torch.load(ae_pth, map_location='cpu')['model'])
ae.to(device)

model = SubGoalValueModel() # SubGoalValueModelCrossAttn()
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr_param)
criterion = nn.MSELoss()

prev_best_test_loss = float('inf')
for epoch in tqdm(range(n_epochs)):
    # train for one epoch
    model.train(True)
    ae.eval()
    train_loss = 0
    for state, goal, subgoal, value in tqdm(train_loader):
        state = state.to(device)
        goal = goal.to(device)
        subgoal = subgoal.to(device)

        _, state_latent = ae.encode(state)
        _, subgoal_latent = ae.encode(subgoal)
        _, goal_latent = ae.encode(goal)

        pred = model(state_latent, goal_latent, subgoal_latent)
        value = value.to(device)

        loss = criterion(pred, value)

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # loss_scaler.step()
    train_loss /= len(train_dataset)
    print(f'Epoch: {epoch}, Scaled Train Loss: {train_loss}')

    if epoch % 5 == 0 or epoch + 1 == n_epochs:
        model.eval()
        ae.eval()
        test_loss = 0
        for state, goal, subgoal, value in tqdm(test_loader):
            state = state.to(device)
            goal = goal.to(device)
            subgoal = subgoal.to(device)

            _, state_latent = ae.encode(state)
            _, subgoal_latent = ae.encode(subgoal)
            _, goal_latent = ae.encode(goal)

            pred = model(state_latent, goal_latent, subgoal_latent)
            value = value.to(device)

            loss = criterion(pred, value)
            test_loss += loss.item()
        test_loss /= len(test_dataset)
        print(f'Scaled Test Loss: {test_loss}')

        # save the model weights
        torch.save(model.state_dict(), save_dir + exp_folder + '/best_value_model.pt')

        if test_loss < prev_best_test_loss:
            prev_best_test_loss = test_loss
            torch.save(model.state_dict(), save_dir + exp_folder + '/best_test_loss_value_model.pt')

        # save the test loss to .txt file
        with open(save_dir + exp_folder + '/test_loss.txt', 'a') as f:
            f.write(f'Epoch: {epoch}, Test Loss: {test_loss}\n')
            f.write(f'Epoch: {epoch}, Train Loss: {train_loss}\n')