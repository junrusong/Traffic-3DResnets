"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.deepst import *


class STResnet(nn.Module):

    def __init__(self, params):
        super(STResnet, self).__init__()

        self.layer_c = nn.Sequential(SingleResnet(params.n_flow * params.len_close, params))
        self.layer_p = nn.Sequential(SingleResnet(params.n_flow * params.len_period, params))
        self.layer_t = nn.Sequential(SingleResnet(params.n_flow * params.len_trend, params))
        self.params = params
        self.tanh = nn.Tanh()

    def forward(self, x):
        x_c = x[:, :self.params.len_close, :, :, :]
        x_p = x[:, self.params.len_close:self.params.len_close + self.params.len_period, :, :, :]
        x_t = x[:, self.params.len_close + self.params.len_period:, :, :, :]

        out_c = self.layer_c(x_c)
        out_p = self.layer_p(x_p)
        out_t = self.layer_t(x_t)

        # merge outputs
        out = out_c + out_p + out_t

        # external components
        # meta, todo

        # out = self.tanh(out)
        return torch.relu(out)


#########################################
########## Section for 3DResnet #########
class BasicBlock3D_1conv(nn.Module):
    def __init__(self):
        super(BasicBlock3D_1conv, self).__init__()
        self.conv = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        # print(x.size())
        return x + residual


class TTP(nn.Module):
    def __init__(self, params, n_res_unit=8):
        super(TTP, self).__init__()
        self.params = params
        # self.n_timesteps = params.len_close + params.len_period + params.len_trend
        self.n_timesteps = 672
        self.topK = 10
        self.next = True
        self.dropout = nn.Dropout(0.15)
        self.conv1 = nn.Conv3d(self.n_timesteps, 64, kernel_size=(3, 3, 3), stride=1, padding=1, dilation=1)
        self.res = self.get_residual_unit(n_res_unit)
        self.conv2 = nn.Conv3d(64, 1, kernel_size=(3, 3, 3), stride=1, padding=1, dilation=1)
        self.bn64 = nn.BatchNorm3d(64)
        self.bnin = nn.BatchNorm3d(self.n_timesteps)
        self.trans1 = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        self.cosine = torch.nn.CosineSimilarity(dim=2)

    def get_residual_unit(self, n):
        block = BasicBlock3D_1conv
        layers = []
        for _ in range(n):
            layers.append(block())
        residual_unit = nn.Sequential(*layers)
        return residual_unit

    def select_topK(self, input_data, k):
        """
        This function is used to select the topK frames
        Input: original frames. size: [batch_size, all_timesteps, n_flow, map_height, map_width]
        Output: selected frames. size: [batch_size, selected_timesteps, n_flow, map_height, map_width]
        """
        batch_size = input_data.size(0)
        input_data = input_data.view(batch_size, self.n_timesteps,
                                     self.params.n_flow * self.params.map_height * self.params.map_width)

        x_last_step = input_data[:, -1, :]
        x_last_step = torch.unsqueeze(x_last_step, 1)
        # print(x_last_step)
        # print(x_last_step.size())
        x_rest_step = input_data[:, :-1, :]
        # print(x_rest_step.size())
        repeat_time = x_rest_step.size(1)
        # print(repeat_time)
        x_lastrepeat = x_last_step.repeat(1, repeat_time, 1)
        # print(x_lastrepeat.size())
        logits = self.cosine(x_lastrepeat, x_rest_step)
        # print(logits)
        logits2 = torch.softmax(logits, dim=1)
        # print(logits2)
        sorted_logits, indices = torch.sort(logits2, descending=True)
        print(sorted_logits)
        # print(indices)
        selected_indices = indices[:, :k]
        if self.next:
            selected_indices = selected_indices + 1
            # print(selected_indices.device)

        mask = torch.zeros(batch_size, repeat_time)
        # print(mask.device)
        mask.cuda()
        mask.scatter_(1, selected_indices, 1.)
        mask = mask.unsqueeze(2).repeat(1, 1, self.params.n_flow * self.params.map_height * self.params.map_width)
        mask = torch.ge(mask, 0.5)

        selected_indices = torch.masked_select(input_data, mask).view(batch_size, k,
                                                                      self.params.n_flow * self.params.map_height * self.params.map_width)
        final_selected_data = torch.cat([selected_indices, x_last_step], dim=1)
        return final_selected_data

    def forward(self, x):
        # print(x.size())
        # exit(1)
        batch_size = x.size(0)

        selected_x = self.select_topK(x, self.topK)


        # x = x.view(batch_size, self.n_timesteps, self.params.n_flow * self.params.map_height * self.params.map_width)
        # x_trans1 = self.trans1(x)
        # x = x_trans1
        # # x = self.trans2(x)

        # x = torch.cat((x, x_trans1),2)
        # x = self.dropout(x)
        # print(x.size())
        # exit(1)

        x = selected_x.view(batch_size, self.topK, self.params.n_flow, self.params.map_height, self.params.map_width)
        x = self.bnin(x)
        x = torch.relu(x)
        # print("relu1", x.size())

        x = self.conv1(x)
        # print("conv1",x.size())
        # exit(1)

        x = self.res(x)
        # print("res", x.size())

        x = self.bn64(x)
        x = torch.relu(x)
        # print("relu2", x.size())
        x = self.dropout(x)
        x = self.conv2(x)
        # print("conv2", x.size())
        # exit(1)

        # x = x.view(batch_size, 2, self.params.n_flow, self.params.map_height, self.params.map_width)
        # x = torch.mean(x,1)
        # x = torch.squeeze(x)

        out = x.view(batch_size, self.params.n_flow, self.params.map_height, self.params.map_width)
        return torch.relu(out)


class ResNet3D(nn.Module):
    def __init__(self, params, n_res_unit=8):
        super(ResNet3D, self).__init__()
        self.params = params
        self.n_timesteps = params.len_close + params.len_period + params.len_trend
        self.conv1 = nn.Conv3d(self.n_timesteps, 64, kernel_size=(3, 3, 3), stride=1, padding=1, dilation=1)
        self.res = self.get_residual_unit(n_res_unit)
        self.conv2 = nn.Conv3d(64, 1, kernel_size=(3, 3, 3), stride=1, padding=1, dilation=1)
        self.bn64 = nn.BatchNorm3d(64)
        self.bnin = nn.BatchNorm3d(self.n_timesteps)

    def get_residual_unit(self, n):
        block = BasicBlock3D_1conv
        layers = []
        for _ in range(n):
            layers.append(block())
        residual_unit = nn.Sequential(*layers)
        return residual_unit

    def forward(self, x):
        # print(x.size())
        # exit(1)
        batch_size = x.size(0)

        x = x.view(batch_size, self.n_timesteps, self.params.n_flow, self.params.map_height, self.params.map_width)
        x = self.bnin(x)
        x = torch.relu(x)
        # print("relu1", x.size())

        x = self.conv1(x)
        # print("conv1",x.size())

        x = self.res(x)
        # print("res", x.size())

        x = self.bn64(x)
        x = torch.relu(x)
        # print("relu2", x.size())
        x = self.conv2(x)
        # print("conv2", x.size())
        # exit(1)

        out = x.view(batch_size, self.params.n_flow, self.params.map_height, self.params.map_width)
        return torch.relu(out)


class SRCNs(nn.Module):
    def __init__(self, params):
        super(SRCNs, self).__init__()

        self.params = params

        # CNN Layers
        kernel_size = (3, 3)
        padding = (3, 3)
        poolingsize = (2, 2)
        self.conv1 = nn.Conv2d(1, params.cnn1out, kernel_size=kernel_size, padding=padding)
        self.Maxp1 = nn.MaxPool2d(poolingsize)
        self.conv2 = nn.Conv2d(params.cnn1out, params.cnn2out, kernel_size=kernel_size, padding=padding)
        self.Maxp2 = nn.MaxPool2d(poolingsize)
        self.conv3 = nn.Conv2d(params.cnn2out, params.cnn3out, kernel_size=kernel_size, padding=padding)
        self.Maxp3 = nn.MaxPool2d(poolingsize)
        self.conv4 = nn.Conv2d(params.cnn3out, params.cnn4out, kernel_size=kernel_size, padding=padding)
        self.Maxp4 = nn.MaxPool2d(poolingsize)
        self.conv5 = nn.Conv2d(params.cnn4out, params.cnn5out, kernel_size=kernel_size, padding=padding)
        self.Maxp5 = nn.MaxPool2d(poolingsize)

        # Batch Normalization
        self.bn16 = nn.BatchNorm2d(16)
        self.bn32 = nn.BatchNorm2d(32)
        self.bn64 = nn.BatchNorm2d(64)
        self.bn128 = nn.BatchNorm2d(128)

        # Dropout
        self.drop = nn.Dropout2d(params.droprate)

        # LSTM Layers
        self.lstm1 = nn.LSTM(1, params.lstm1out, batch_first=True)
        self.lstm2 = nn.LSTM(params.lstm1out, params.lstm2out, batch_first=True)

        # Fully Connected Layer
        self.fc1 = nn.Linear(params.f1, params.n_flow * params.map_height * params.map_width)
        self.fc2 = nn.Linear(params.f2, params.n_flow * params.map_height * params.map_width)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(x.shape[0], 1, x.shape[1] * x.shape[2], x.shape[3] * x.shape[4])

        x = self.bn16(F.relu(self.Maxp1(self.conv1(x))))
        x = self.bn32(F.relu(self.Maxp2(self.conv2(x))))
        x = self.bn64(F.relu(self.Maxp3(self.conv3(x))))
        x = self.bn64(F.relu(self.Maxp4(self.conv4(x))))
        x = self.bn128(F.relu(self.Maxp5(self.conv5(x))))
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = x.view(x.size(0), x.size(1), 1)

        x, _ = self.lstm1(x)
        x = torch.tanh(x)
        x, _ = self.lstm2(x)
        x = torch.tanh(x)

        x = x.contiguous().view(batch_size, -1)
        x = self.drop(x)
        x = self.fc2(x)
        out = x.view(batch_size, self.params.n_flow, self.params.map_height, self.params.map_width)

        return torch.relu(out)


class ResNet34(nn.Module):
    def __init__(self, params):
        super(ResNet34, self).__init__()

        self.n_flow = params.n_flow
        self.map_height = params.map_height
        self.map_width = params.map_width

        # CNN Layers
        restnet = models.resnet34()
        restnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        restnet.fc = nn.Linear(2048, params.resnet_fc_out)
        restnet.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=(1, 1))
        self.resnet34 = restnet

        # Fully Connected Layer
        self.fc = nn.Linear(params.resnet_fc_out, params.n_flow * params.map_height * params.map_width)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(x.shape[0], 1, x.shape[1] * x.shape[2], x.shape[3] * x.shape[4])

        x = self.resnet34(x)

        x = x.view(batch_size, -1)
        # print(x.shape)
        x = self.fc(x)
        out = x.view(batch_size, self.n_flow, self.map_height, self.map_width)

        return torch.relu(out)


def rmse(predictions, targets):
    """Compute root mean squared error"""
    rmse = np.sqrt(((predictions - targets) ** 2).mean())
    return rmse


def mse(predictions, targets):
    """Compute mean squared error"""
    return ((predictions - targets) ** 2).mean()


def mae(predictions, targets):
    """Compute mean absolute error"""
    s = np.sum(np.absolute((predictions - targets)))
    for dim in targets.shape:
        s /= dim
    return s


def mape(predictions, targets):
    """Compute mean absolute precentage error"""
    mask = targets != 0
    return (np.fabs(targets[mask] - predictions[mask]) / targets[mask]).mean()


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'mse': mse,
    'rmse': rmse,
    'mae': mae,
    'mape': mape
}
