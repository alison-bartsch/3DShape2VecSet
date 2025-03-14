import os
import torch
from tqdm import tqdm
import torch.utils.data as data
import matplotlib.pyplot as plt
import models_ae
from action_pred_model import ActionPredModel
from real_world_dataset import ActionPredDataset

dataset_dir = '/home/alison/Documents/Feb26_Human_Demos_Raw/pottery'

# create datasets and dataloaders for train/test
test_dataset = ActionPredDataset(dataset_dir, [4,5], 60)
test_loader = data.DataLoader(test_dataset, batch_size=8, shuffle=True)

# load in the pretrained embedding model
ae_pth = '/home/alison/Downloads/checkpoint-199.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ae = models_ae.__dict__['kl_d512_m512_l8']()
ae.eval()
ae.load_state_dict(torch.load(ae_pth, map_location='cpu')['model'])
ae.to(device)
ae.eval()

# initialize the action prediction model
action_model = ActionPredModel()
checkpoint = torch.load('/home/alison/Documents/GitHub/subgoal_diffusion/model_weights/latent_action_pred/best_action_model.pt')
action_model.load_state_dict(checkpoint)
action_model.to(device)
action_model.eval()

for state1, state2, nactions in tqdm(test_loader):
    state1 = state1.to(device)
    state2 = state2.to(device)
    nactions = nactions.to(device)
    latent1 = ae.encode(state1)[1]
    latent2 = ae.encode(state2)[1]
    pred_nactions = action_model(latent1, latent2)
    loss = torch.nn.MSELoss()(pred_nactions, nactions)
    print(f'\nTest Loss: {loss.item()}, Predicted Actions: {pred_nactions}, True Actions: {nactions}')