import os
import torch
from tqdm import tqdm
import torch.utils.data as data
import matplotlib.pyplot as plt
import models_ae
import torch.nn as nn
from action_pred_model import ActionPredModel
from real_world_dataset import ActionPredDataset

dataset_dir = '/home/alison/Documents/Feb26_Human_Demos_Raw/pottery'
save_dir = '/home/alison/Documents/GitHub/subgoal_diffusion/model_weights/'
exp_folder = 'latent_action_pcl_normalized_smaller_lr_scheduler' 
os.makedirs(save_dir + exp_folder)

# create datasets and dataloaders for train/test
train_dataset = ActionPredDataset(dataset_dir, [0,1,2,3], 5)
test_dataset = ActionPredDataset(dataset_dir, [4,5], 5)
train_loader = data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=8, shuffle=True)

# load in the pretrained embedding model
ae_pth = '/home/alison/Downloads/checkpoint-199.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ae = models_ae.__dict__['kl_d512_m512_l8']()
ae.eval()
ae.load_state_dict(torch.load(ae_pth, map_location='cpu')['model'])
ae.to(device)

# initialize the action prediction model
action_model = ActionPredModel()
action_model.to(device)

# setup the training loop
epochs = 50
best_test_loss = float('inf')
optimizer = torch.optim.Adam(list(action_model.parameters()), lr=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

train_losses = []
test_losses = []

for epoch in tqdm(range(epochs)):
    # training loop
    ae.eval()
    action_model.train()
    train_loss = 0
    for state1, state2, nactions in tqdm(train_loader):
        state1 = state1.to(device)
        state2 = state2.to(device)
        nactions = nactions.to(device)
        latent1 = ae.encode(state1)[1]
        latent2 = ae.encode(state2)[1]
        pred_nactions = action_model(latent1, latent2)
        loss = torch.nn.MSELoss()(pred_nactions, nactions)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    train_loss /= len(train_dataset)
    train_losses.append(train_loss)
    print(f'Scaled Train Loss: {train_loss}')

    # once trained, test the model on the test set and report the MSE for the action prediction
    ae.eval()
    action_model.eval()
    test_loss = 0
    for state1, state2, nactions in tqdm(test_loader):
        state1 = state1.to(device)
        state2 = state2.to(device)
        nactions = nactions.to(device)
        latent1 = ae.encode(state1)[1]
        latent2 = ae.encode(state2)[1]
        pred_nactions = action_model(latent1, latent2)
        loss = torch.nn.MSELoss()(pred_nactions, nactions)
        test_loss += loss.item()

    # scale the test loss by the number of samples
    test_loss /= len(test_dataset)
    test_losses.append(test_loss)
    print(f'Scaled Test Loss: {test_loss}')

    # save the model weights if the val loss decreases
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(action_model.state_dict(), save_dir + exp_folder + '/best_action_model.pt')
        print('Model saved!')

        # save the train, val and test losses to .txt file for later analysis
        with open(save_dir + exp_folder + '/train_test_loss.txt', 'w') as f:
            f.write(f'Train Loss: {train_loss}\nTest Loss: {test_loss}')
    elif epoch % 50 == 0:
        torch.save(action_model.state_dict(), save_dir + exp_folder + '/action_model_epoch' + str(epoch) + '.pt')
        

# create a plot of the train, val and test losses in the same plot with a legend and save the plot
x = [i for i in range(epochs)]
plt.plot(x, train_losses, label='Train Loss')
plt.plot(x, test_losses, label='Test Loss')
plt.legend()
plt.title('Train, Test Losses')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.savefig(save_dir + exp_folder + '/train_test_loss_plot.png')