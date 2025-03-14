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
        
        self.fc4 = torch.nn.Linear(256, 1)

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
         
        # sum the two latent vectors
        x = x1 + x2
        x = self.fc4(x)
        return x