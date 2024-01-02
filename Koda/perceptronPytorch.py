
import torch
from torch import nn, optim

class PerceptronPytorch(nn.Module):
    def __init__(self, inNo, hidNo, outNo):
        super(PerceptronPytorch, self).__init__()
        # Podatki o omrežju
        self.vhodiNo = inNo # stevilo vhodov
        self.skritiNo = hidNo # stevilo skritih plasti
        self.izhodiNo = outNo # stevilo izhodov

        self.layer1 = nn.Linear(inNo, hidNo)
        self.layer2 = nn.Linear(hidNo, hidNo)
        self.output_layer = nn.Linear(hidNo, outNo)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x.float()))
        x = torch.sigmoid(self.layer2(x))
        x = self.output_layer(x)
        return x


