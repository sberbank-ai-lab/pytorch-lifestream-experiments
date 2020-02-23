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

class MnistMetricLearningNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 256)

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
        return x

# --------------------------------------------------------------------------------------------------

class MnistClassificationMetricLearningModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.metric_learn_model = get_mnist_metriclearning_model()
        self.metric_learn_model.train()
        for param in self.metric_learn_model.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(32, 10)

    def forward(self, x):
        if len(x.size()) == 4:
            x = x[:, -1, :, :].unsqueeze(1) # get augmented image
        else:
            x = x.unsqueeze(1)
        x = self.metric_learn_model(x)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output


# --------------------------------------------------------------------------------------------------

class MnistDomyshnikNetNet(nn.Module):
    def __init__(self):
        super().__init__()
        #self.metric_learn_model = get_mnist_metriclearning_model()
        self.metric_learn_model = get_mnist_metriclearning_model_cated()
        #self.metric_learn_model = MnistMetricLearningNet2()
        self.metric_learn_model.train()
        for param in self.metric_learn_model.parameters():
            param.requires_grad = False

        self.fc0 = nn.Linear(256, 512)
        self.fc1_ = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 1, x.size(-2), x.size(-1)) # b, augs, x, y -> b*augs, 1, x, y
        x = self.metric_learn_model(x)
        x = F.relu( self.fc0(x))
        x = self.fc1_(x)
        output = F.log_softmax(x, dim=1)
        return output

class MnistDomyshnikNetNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.metric_learn_model = MnistMetricLearningNet2()

        self.fc0 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 1, x.size(-2), x.size(-1)) # b, augs, x, y -> b*augs, 1, x, y
        x = self.metric_learn_model(x)
        x = F.relu(x)
        x = self.fc0(x)
        output = F.log_softmax(x, dim=1)
        return output

# --------------------------------------------------------------------------------------------------

def save_model_params(model, model_name):
    pth = osp.join(WEIGHTS_PATH, MODEL_POSTFIX + '_' + model_name)
    torch.save(model.state_dict(), pth)
    print(f'model saved to {pth}')

def load_model_params(model, model_name):
    pth = osp.join(WEIGHTS_PATH, MODEL_POSTFIX + '_' + model_name)
    model.load_state_dict(torch.load(pth))
    model.eval()
    print(f'load model {pth}')
    return model

def get_mnist_metriclearning_model():
    model = MnistMetricLearningNet()
    return load_model_params(model, 'mnist_metric_learning.w')

def get_mnist_metriclearning_model_cated():
    tmp = get_mnist_metriclearning_model()
    model = MnistMetricLearningNet2()
    model.load_state_dict(tmp.state_dict(), strict=False)
    return model

def get_mnist_domyshnik_model():
    model = MnistDomyshnikNetNet()
    return load_model_params(model, 'mnist_domushnik.w')