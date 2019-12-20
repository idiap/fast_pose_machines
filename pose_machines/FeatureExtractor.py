"""
FastPose: Efficient human pose estimation in depth images with CNN.


Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
Written by Angel Martinez Gonzalez <angel.martinez@idiap.ch>

This file is part of the project FastPose for TSCVT paper:

Efficient Convolutional Neural Networks for Depth-Based Multi-Person Pose Estimation

FastPose is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3 as
published by the Free Software Foundation.

FastPose is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FastPose. If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
sys.path.append('./../')



class ResModule(nn.Module):
    def __init__(self, inputChannels, outputChannels):
        super(ResModule, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=inputChannels,\
                               out_channels=outputChannels,\
                               kernel_size=3, \
                               stride=1, \
                               padding=1)

        self.conv2 = nn.Conv2d(in_channels=outputChannels,\
                               out_channels=outputChannels,\
                               kernel_size=3, \
                               stride=1, \
                               padding=1)



        self.shortcut = None
        if inputChannels != outputChannels:
            self.shortcut = nn.Conv2d(in_channels=inputChannels,\
                                      out_channels=outputChannels,\
                                      kernel_size=1, \
                                      stride=1, \
                                      padding=0)



        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(num_features=outputChannels)
        self.bn2 = nn.BatchNorm2d(num_features=outputChannels)
        self.bn3 = nn.BatchNorm2d(num_features=outputChannels)


    def forward(self, x):
        res = x
        y = self.conv1(x)
        y = self.bn1(y)
        self.relu1(y)
        y = self.conv2(y)
        y = self.bn2(y)


        # TODO Implement other shortcut
        # Some do upsample or downsample
        if self.shortcut is not None:
            res = self.shortcut(res)
        y = res + y

        y = self.bn3(y)
        self.relu2(y)

        return y




class ResidualFeatureExtractor(nn.Module):
    def __init__(self, inputChannels, outputChannels):
        # Uper class initalization. 
        # Call needed to be an actual pytorch module
        super(ResidualFeatureExtractor, self).__init__()
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels


        print('[INFO] (%s) Intializing Residual feature extractor module ' % self.__class__.__name__)
        print('[INFO] (%s) Number of input channels %d ' % (self.__class__.__name__, self.inputChannels))
        print('[INFO] (%s) Number of output channels %d ' % (self.__class__.__name__, self.outputChannels))

        self.init_layers()


    def forward(self, x):
        x = self.front(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.pool2(x)
        x = self.res3(x)
        x = self.back(x)

        return x



    def init_layers(self):
        frontLayers = [nn.Conv2d(self.inputChannels, 128, kernel_size=7, stride=1, padding=3),\
                       nn.BatchNorm2d(128),\
                       nn.ReLU(inplace=True),\
                       nn.AvgPool2d(kernel_size=2, stride=2, padding=0)]
        backLayers = [nn.AvgPool2d(kernel_size=2, stride=2, padding=0),\
                      nn.Conv2d(in_channels=self.outputChannels,\
                                out_channels=self.outputChannels,\
                                kernel_size=1, stride=1, padding=0)]


        self.front = nn.Sequential(*frontLayers)
        self.res1  = ResModule(128, self.outputChannels)
        self.res2  = ResModule(self.outputChannels, self.outputChannels)
        self.pool2 = nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
        self.res3  = ResModule(self.outputChannels, self.outputChannels) 
        self.back  = nn.Sequential(*backLayers)


