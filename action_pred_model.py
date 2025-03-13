import torch


# TODO: create simple multi-layered model to project (batch, 512, 8) to (batch, 1)
class ActionPredModel(torch.nn.Module):
    def __init__(self):
        super(ActionPredModel, self).__init__()
        self.fc1 = torch.nn.Linear(512*8, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 256)
        self.fc4 = torch.nn.Linear(256, 1)

    def forward(self, x):
        # flatten x from shape (batch, 512, 8) to (batch, 512*8)
        x = x.view(-1, 512*8)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x