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


import torch
import torch.nn as nn


class ConvBn(nn.Module):
    def __init__(self, inputChannels, outputChannels, stride):
        super(ConvBn, self).__init__()
        self.inputChannels=inputChannels
        self.outputChannels=outputChannels
        self.stride= stride

        self.mod = nn.Sequential(
                   nn.Conv2d(self.inputChannels, self.outputChannels, 
                              kernel_size=7, stride=self.stride, 
                                padding=3, bias=False),
                   nn.BatchNorm2d(self.outputChannels),
                   nn.ReLU(inplace=True)
                   )

    def forward(self, x):
        return self.mod(x)

class ConvDW(nn.Module):
    def __init__(self, inputChannels, outputChannels, stride):
        super(ConvDW, self).__init__()
        self.inputChannels  = inputChannels
        self.outputChannels = outputChannels
        self.stride         = stride


        self.mod = nn.Sequential(
                    nn.Conv2d(self.inputChannels, self.inputChannels, 
                              kernel_size=3,  stride=self.stride, 
                              padding=1, groups=self.inputChannels, bias=False),
                    nn.BatchNorm2d(self.inputChannels),
                    nn.ReLU(inplace=True),
        
                    nn.Conv2d(self.inputChannels, self.outputChannels, 
                              kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(self.outputChannels),
                    nn.ReLU(inplace=True),
                   )


    def forward(self, x):
        return self.mod(x)




class ResDwModule(nn.Module):
    def __init__(self, inputChannels, outputChannels, stride):
        super(ResDwModule, self).__init__()


        self.main = ConvDW(inputChannels, outputChannels, stride) 

        self.shortcut = None
        if inputChannels != outputChannels:
            self.shortcut = nn.Conv2d(in_channels=inputChannels,\
                                      out_channels=outputChannels,\
                                      kernel_size=1, \
                                      stride=stride, \
                                      padding=0)



        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_features=outputChannels)


    def forward(self, x):
        res = x
        y = self.main(x)

        # TODO Implement other shortcut
        # Some do upsample or downsample
        if self.shortcut is not None:
            res = self.shortcut(res)
        y = res + y

        y = self.bn1(y)
        self.relu1(y)

        return y






class MobileNetFeatureExtractor(nn.Module):
    def __init__(self, inChannels, outChannels, featType='flat'):
        super(MobileNetFeatureExtractor, self).__init__()

        self.inputChannels=inChannels
        self.outputChannels=outChannels
        self.featType=featType


        print('[INFO] (%s) The feature extractor to build is %s ' % (self.__class__.__name__, self.featType))

        if featType=='flat':
        ### fixed number of feature maps in the feature extractor
            self.init_layers_first()
        else:# expand
        ### different number of feature maps in feat extractor
            self.init_layers()


    def init_layers(self):
        self.model = nn.Sequential(
            ConvBn(inputChannels=1,   outputChannels=64, stride=2), ## 
            ConvDW(inputChannels=64,  outputChannels=64, stride=1),
            ### residual
            ResDwModule(inputChannels=64, outputChannels=64, stride=1),
            ConvDW(inputChannels=64,  outputChannels=128, stride=2), ##
            ### residual
            ResDwModule(inputChannels=128,outputChannels=128,stride=1),
            ConvDW(inputChannels=128,  outputChannels=128, stride=1),
            ### residual
            ResDwModule(inputChannels=128, outputChannels=128, stride=1), 
            ConvDW(inputChannels=128,  outputChannels=256, stride=2),##
            ConvDW(inputChannels=256,  outputChannels=self.outputChannels, stride=1),
        )

    def init_layers_first(self):
        self.model = nn.Sequential(
            ConvBn(  1,  32, 2), ## 
            ConvDW( 32,  64, 1),
            ResDwModule(64, 64, 1),
            ConvDW( 64,  64, 2), ##
            ResDwModule(64,64,1),
            ConvDW( 64,  64, 1),
            ResDwModule(64,64,1), 
            ConvDW( 64,  64, 2),##
            ConvDW( 64,  self.outputChannels, 1),
        )



    def init_layers_dumm(self):
        self.model = nn.Sequential(
            ConvBn(  1,  32, 2), ## 
            ConvDW( 32,  64, 1),
            ConvDW( 64,  64, 1),
            ConvDW( 64,  64, 2), ##
            ConvDW( 64,  64, 1),
            ConvDW( 64,  64, 1),
            ConvDW( 64,  64, 1),
            ConvDW( 64,  64, 2),##
            ConvDW( 64,  64, 1),
        )



    def forward(self, x):
        return self.model(x)


