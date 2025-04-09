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
exp_folder = 'latent_subgoal_more_epochs_smaller_lr_more_augs' 
os.makedirs(save_dir + exp_folder)

# parameters
n_epochs = 800
n_subgoal_steps = 5
lr_param = 1e-6
batch = 4

# create and save dictionary with parameters
params = {
    'n_epochs': n_epochs,
    'n_subgoal_steps': n_subgoal_steps,
    'lr_param': lr_param,
    'batch': batch,
}
# write params to text file
with open(save_dir + exp_folder + '/params.txt', 'w') as f:
    for key, value in params.items():
        f.write(f'{key}: {value}\n')

# create datasets and dataloaders for train/test
train_dataset = SubGoalDataset(dataset_dir, [0,1,2,3], 3, n_subgoal_steps)
test_dataset = SubGoalDataset(dataset_dir, [4,5], 3, n_subgoal_steps)
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

# print("Model = %s" % str(model_without_ddp))
# print('number of params (M): %.2f' % (n_parameters / 1.e6))

# eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

# if args.lr is None:  # only base_lr is specified
#     args.lr = args.blr * eff_batch_size / 256

# print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
# print("actual lr: %.2e" % args.lr)

# print("accumulate grad iterations: %d" % args.accum_iter)
# print("effective batch size: %d" % eff_batch_size)

# if args.distributed:
#     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
#     model_without_ddp = model.module

optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=lr_param)
# loss_scaler = NativeScaler()

criterion = models_class_cond.__dict__['EDMLoss']()

print("criterion = %s" % str(criterion))


for epoch in tqdm(range(n_epochs)):
    # train for one epoch
    model.train(True)
    ae.eval()
    train_loss = 0
    for state, next_state, goal in tqdm(train_loader):
        state = state.to(device)
        next_state = next_state.to(device)
        goal = goal.to(device)

        state_latent = ae.encode(state)[1]
        latent2 = ae.encode(next_state)[1]
        goal_latent = ae.encode(goal)[1]

        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():

                _, state_latent = ae.encode(state)
                _, next_state_latent = ae.encode(next_state)
                _, goal_latent = ae.encode(goal)

            loss = criterion(model, next_state_latent, state_latent, goal_latent)

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_dataset)
    print(f'Scaled Train Loss: {train_loss}')

    if epoch % 5 == 0 or epoch + 1 == n_epochs:
        model.eval()
        ae.eval()
        test_loss = 0
        for state, next_state, goal in tqdm(test_loader):
            state = state.to(device)
            next_state = next_state.to(device)
            goal = goal.to(device)

            state_latent = ae.encode(state)[1]
            latent2 = ae.encode(next_state)[1]
            goal_latent = ae.encode(goal)[1]

            with torch.cuda.amp.autocast(enabled=False):
                with torch.no_grad():
                    _, state_latent = ae.encode(state)
                    _, next_state_latent = ae.encode(next_state)
                    _, goal_latent = ae.encode(goal)

                loss = criterion(model, next_state_latent, state_latent, goal_latent)

            test_loss += loss.item()
        test_loss /= len(test_dataset)
        print(f'Scaled Test Loss: {test_loss}')

        # save the model weights
        torch.save(model.state_dict(), save_dir + exp_folder + '/best_diffusion_model.pt')