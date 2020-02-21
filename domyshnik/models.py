import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp

from dltranz.metric_learn.ml_models import L2Normalization
from domyshnik.constants import *

class MnistClassificationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        if len(x.size()) == 4:
            x = x[:, -1, :, :].unsqueeze(1) # get augmented image
        else:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
# --------------------------------------------------------------------------------------------------

class MnistMetricLearningNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 256)
        self.fc2 = nn.Linear(256, 32)
        self.norm = L2Normalization()

    def forward(self, x):
        x = x.view(-1, 1, x.size(-2), x.size(-1)) # b, augs, x, y -> b*augs, 1, x, y
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = self.norm(x)
        return output

# --------------------------------------------------------------------------------------------------

class MnistDomyshnikNetNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 1, x.size(-2), x.size(-1)) # b, augs, x, y -> b*augs, 1, x, y
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# --------------------------------------------------------------------------------------------------

def save_model_params(model, model_name):
    pth = osp.join(WEIGHTS_PATH, model_name, MODEL_POSTFIX)
    torch.save(model.state_dict(), pth)
    print(f'model saved to {pth}')

def load_model_params(model, model_name):
    pth = osp.join(WEIGHTS_PATH, model_name, MODEL_POSTFIX)
    model.load_state_dict(torch.load(pth))
    model.eval()
    print(f'load model {pth}')
    return model