import torch


# TODO: create simple multi-layered model to project (batch, 512, 8) to (batch, 1)
class ActionPredModel(torch.nn.Module):
    def __init__(self):
        super(ActionPredModel, self).__init__()
        self.fc11 = torch.nn.Linear(512*8, 1024)
        self.fc21 = torch.nn.Linear(1024, 512)
        self.fc31 = torch.nn.Linear(512, 256)

        self.fc12 = torch.nn.Linear(512*8, 1024)
        self.fc22 = torch.nn.Linear(1024, 512)
        self.fc32 = torch.nn.Linear(512, 256)
        
        self.fc4 = torch.nn.Linear(2*256, 256)
        self.fc5 = torch.nn.Linear(256, 1)

    def forward(self, x1, x2):
        # flatten x from shape (batch, 512, 8) to (batch, 512*8)
        x1 = x1.view(-1, 512*8)
        x1 = torch.nn.functional.relu(self.fc11(x1))
        x1 = torch.nn.functional.relu(self.fc21(x1))
        x1 = torch.nn.functional.relu(self.fc31(x1))

        x2 = x2.view(-1, 512*8)
        x2 = torch.nn.functional.relu(self.fc12(x2))
        x2 = torch.nn.functional.relu(self.fc22(x2))
        x2 = torch.nn.functional.relu(self.fc32(x2))
         
        # # sum the two latent vectors
        # x = x1 + x2

        # concatenate the two latent vectors
        x = torch.cat((x1, x2), dim=1)
        x = torch.nn.functional.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
class CrossAttn(torch.nn.Module):
    def __init__(self):
        super(CrossAttn, self).__init__()
        '''
        Single cross attention layer
        '''
        self.attn = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.fc = torch.nn.Linear(512, 512)
        self.norm1 = torch.nn.LayerNorm(512)
        self.norm2 = torch.nn.LayerNorm(512)

    def forward(self, x1, x2):
        # x1 shape (batch, 512, 8)
        # x2 shape (batch, 512, 8)
        # x1 = x1.permute(0, 2, 1) # (batch, 8, 512)
        # x2 = x2.permute(0, 2, 1) # (batch, 8, 512)

        attn_out, _ = self.attn(x1, x2, x2)
        attn_out = self.fc(attn_out)
        attn_out = self.norm1(attn_out + x1)

        out = self.norm2(attn_out + x2)
        return out
    
class SubGoalValueModelCrossAttn(torch.nn.Module):
    def __init__(self):
        super(SubGoalValueModelCrossAttn, self).__init__()
        self.cross_attn = CrossAttn()
        self.fc1 = torch.nn.Linear(512*8, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 256)
        self.fc4 = torch.nn.Linear(256, 1)

    def forward(self, state, goal, subgoal):
        # state shape (batch, 512, 8)
        # goal shape (batch, 512, 8) 
        # subgoal shape (batch, 512, 8)
        x = self.cross_attn(state, goal)
        x = self.cross_attn(x, subgoal)
        x = x.view(-1,512*8)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))

    

class SubGoalValueModel(torch.nn.Module):
    def __init__(self):
        super(SubGoalValueModel, self).__init__()
        self.fc1 = torch.nn.Linear(512*8, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 256)
        self.fc4 = torch.nn.Linear(256, 1)

    def forward(self, state, goal, subgoal):
        # state shape (batch, 512, 8)
        # goal shape (batch, 512, 8) 
        # subgoal shape (batch, 512, 8)
        x = torch.matmul(state, goal.transpose(1, 2))
        x = torch.matmul(x, subgoal)
        x = x.view(-1, 512*8)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x)) # constrain output to be between 0 and 1
        return x