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
from collections import OrderedDict

import torch
import torch.nn as nn

import numpy as np
import time

from . import FeatureExtractor
from . import SqueezeNet
from . import MobileNet


def get_mobile_stage_(inputFeatures, widthFeatures, nParts, nLimbs, stagen):
    # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    print('[INFO] (get_mobile_stage_) inputFeatures= %d widthFeatures= %d' % (inputFeatures, widthFeatures))
    stageL1 = nn.Sequential(OrderedDict([\
                   ('MobMod1_S%s_L1'%stagen, MobileNet.ConvDW(inputFeatures, widthFeatures, 1)), 
                   ('MobMod2_S%s_L1'%stagen, MobileNet.ConvDW(widthFeatures, widthFeatures, 1)), 
                   ('MobMod3_S%s_L1'%stagen, MobileNet.ConvDW(widthFeatures, widthFeatures, 1)), 
                   ('MobMod4_S%s_L1'%stagen, MobileNet.ConvDW(widthFeatures, widthFeatures, 1)), 
                   ('MobMod5_S%s_L1'%stagen, MobileNet.ConvDW(widthFeatures, widthFeatures, 1)), 
                   ('Conv1_S%s_L1'%stagen, nn.Conv2d(widthFeatures, widthFeatures, kernel_size=1, stride=1,padding=0)),
                   ('Relu1_S%s_L1'%stagen, nn.ReLU(inplace=True)),
                   ('Conv2_S%s_L1'%stagen, nn.Conv2d(widthFeatures, nLimbs, kernel_size=1, stride=1, padding=0))
                   ]))

    stageL2 = nn.Sequential(OrderedDict([\
                   ('MobMod1_S%s_L2'%stagen, MobileNet.ConvDW(inputFeatures, widthFeatures, 1)), 
                   ('MobMod2_S%s_L2'%stagen, MobileNet.ConvDW(widthFeatures, widthFeatures, 1)), 
                   ('MobMod3_S%s_L2'%stagen, MobileNet.ConvDW(widthFeatures, widthFeatures, 1)), 
                   ('MobMod4_S%s_L2'%stagen, MobileNet.ConvDW(widthFeatures, widthFeatures, 1)), 
                   ('MobMod5_S%s_L2'%stagen, MobileNet.ConvDW(widthFeatures, widthFeatures, 1)), 
                   ('Conv1_S%s_L2'%stagen, nn.Conv2d(widthFeatures, widthFeatures, kernel_size=1, stride=1,padding=0)),
                   ('Relu1_S%s_L2'%stagen, nn.ReLU(inplace=True)),
                   ('Conv2_S%s_L2'%stagen, nn.Conv2d(widthFeatures, nParts, kernel_size=1, stride=1, padding=0))
                   ]))


    return stageL1, stageL2




def get_squeeze_stage_(inputFeatures, widthFeatures, nParts, nLimbs, stagen):
    ### Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    ### Fire(64, 16, 32, 32)
    sratio=0.75
    pct   =0.5

    s1x1= int(sratio*widthFeatures)
    e1x1= int(pct*widthFeatures)
    e3x3= int(pct*widthFeatures)


    print('[INFO] (get_squeeze_stage_) inputFeatures= %d widthFeatures= %d s1x1= %d e1x1= %d e3x3= %d' % (inputFeatures, widthFeatures, s1x1, e1x1, e3x3))

    stageL1 = nn.Sequential(OrderedDict([\
                   ('SqMod1_S%s_L1'%stagen, SqueezeNet.Fire(inputFeatures, s1x1, e1x1, e3x3)), 
                   ('SqMod2_S%s_L1'%stagen, SqueezeNet.Fire(widthFeatures, s1x1, e1x1, e3x3)), 
                   ('SqMod3_S%s_L1'%stagen, SqueezeNet.Fire(widthFeatures, s1x1, e1x1, e3x3)), 
                   ('SqMod4_S%s_L1'%stagen, SqueezeNet.Fire(widthFeatures, s1x1, e1x1, e3x3)), 
                   ('SqMod5_S%s_L1'%stagen, SqueezeNet.Fire(widthFeatures, s1x1, e1x1, e3x3)), 
                   ('Conv1_S%s_L1'%stagen, nn.Conv2d(widthFeatures, widthFeatures, kernel_size=1, stride=1,padding=0)),
                   ('Relu1_S%s_L1'%stagen, nn.ReLU(inplace=True)),
                   ('Conv2_S%s_L1'%stagen, nn.Conv2d(widthFeatures, nLimbs, kernel_size=1, stride=1, padding=0))
                   ]))

    stageL2 = nn.Sequential(OrderedDict([\
                   ('SqMod1_S%s_L2'%stagen, SqueezeNet.Fire(inputFeatures, s1x1, e1x1, e3x3)), 
                   ('SqMod2_S%s_L2'%stagen, SqueezeNet.Fire(widthFeatures, s1x1, e1x1, e3x3)), 
                   ('SqMod3_S%s_L2'%stagen, SqueezeNet.Fire(widthFeatures, s1x1, e1x1, e3x3)), 
                   ('SqMod4_S%s_L2'%stagen, SqueezeNet.Fire(widthFeatures, s1x1, e1x1, e3x3)), 
                   ('SqMod5_S%s_L2'%stagen, SqueezeNet.Fire(widthFeatures, s1x1, e1x1, e3x3)), 
                   ('Conv1_S%s_L2'%stagen, nn.Conv2d(widthFeatures, widthFeatures, kernel_size=1, stride=1,padding=0)),
                   ('Relu1_S%s_L2'%stagen, nn.ReLU(inplace=True)),
                   ('Conv2_S%s_L2'%stagen, nn.Conv2d(widthFeatures, nParts, kernel_size=1, stride=1, padding=0))
                   ]))


    return stageL1, stageL2




#######
### Pytorch implementation of Convolutional Pose Machines
### The class aims to implement the pose estimator of the paper
### Real time human pose estimation using part affinity fields.
### The module had been implemented as a proper pytorch network
### module to ease the gradient computation and weight update.
###
class PoseMachine(nn.Module):


    ##
    # @brief Constructor
    #
    def __init__(self, params):
        # Uper class initalization. 
        # Call needed to be an actual pytorch module
        super(PoseMachine, self).__init__()

        self.params= params
        # Containers of per stage maps and stage modules
        self.stageMapsL1 = [None]*self.params['n_stages']
        self.stageMapsL2 = [None]*self.params['n_stages']
        self.stagesL1 = nn.ModuleList()
        self.stagesL2 = nn.ModuleList()

        # Initialize feat extractor and stages
        self.init_stages()




    ##
    # @brief Forward pass of the network.
    #
    def forward(self, x):
        out = self.featExtractor(x)

        # Operating in deploy mode
        return self.forward_stages_deploy(out), out



    ##
    # @brief Forward pass of the network for deploy mode
    def forward_stages_deploy(self, x):
        out = x
        catStage = 1
        self.catFeats= x

        self.outL1= [None]*self.params['n_stages']
        self.outL2= [None]*self.params['n_stages']

        for i in range(self.params['n_stages']):
            out1 = self.stagesL1[i](out)
            out2 = self.stagesL2[i](out)

            self.outL1[i] = out1.clone()
            self.outL2[i] = out2.clone()

            if catStage < self.params['n_stages']:
                out  = torch.cat([out1, out2,x], 1)
                self.catFeats= out
                catStage += 1

        #### out1= PAF maps, out2= conf maps
        return out1, out2



    ##
    # @brief Initialization of the VGG block. Either from scratch or by finetuning.
    #
    def init_feature_extractor(self):
        thisName = self.__class__.__name__
        if self.params['feat_extractor_type']== 'rpm_feats':
            print('[INFO] (%s) Intializing feature extractor with residual modules!' % (thisName))
            self.featExtractor= FeatureExtractor.ResidualFeatureExtractor(self.params['input_channels'], self.params['feat_chann_width'])

        elif self.params['feat_extractor_type']== 'squeeze_net':
            print('[INFO] (%s) Intializing feature extractor with squeeze nets!' % (thisName))
            self.featExtractor= SqueezeNet.SqueezeNetFeatureExtractor(self.params['input_channels'], self.params['feat_chann_width'])

        elif self.params['feat_extractor_type'] == 'mobile_net':
            print('[INFO] (%s) Intializing feature extractor with mobile nets!' % (thisName))
            self.featExtractor = MobileNet.MobileNetFeatureExtractor(self.params['input_channels'], self.params['feat_chann_width'])
        else:
            #TODO
            print('[WARNIG] (%s) Initializing VGG from scratch!!!' % thisName)
            self.featExtractor= FeatureExtractor.ResidualFeatureExtractor(self.params['input_channels'], self.params['feat_chann_width'])

        #### Get the number of parameters in the feature extractor
        model_parameters = filter(lambda p: p.requires_grad, self.featExtractor.parameters())
        nparams = sum([np.prod(p.size()) for p in model_parameters])
        print('[INFO] (%s) The feature extractor contains %d paramters' % (thisName, nparams))

        return self.params['feat_chann_width']



    ##
    # @brief Initalization of the different stages of the network.
    #
    def init_stages(self):
        # Initialize the feature extractor and get the 
        # number of features it produces
        oSize = self.init_feature_extractor()
        print('[INFO] The number of features from feat extractor is', oSize)

        ### initialization of stages depending on type of architecture
        if self.params['cascade_type'] == 'rpm_cascade':
            # Initialize the first stage
            self.init_stage1(oSize)

            # Intialize the rest of the stages
            for i in range(1, self.params['n_stages']):
                self.init_stage_n(i+1, oSize + self.params['n_limbs'] + self.params['n_parts'])

        elif self.params['cascade_type'] == 'mobilenet_cascade':
            self.init_cascade_mobilenet(oSize)
        else: ### squezzenet_cascade
            self.init_cascade_squeeze(oSize)
            ##



    def init_cascade_mobilenet(self, inputFeatures):
        print('[INFO] (%s) Prediction cascade mobilenet layers!' % self.__class__.__name__)


        fSize= inputFeatures
        for s in range(self.params['n_stages']):
            stageL1, stageL2= get_mobile_stage_(fSize, 
                                                self.params['feat_chann_width'], 
                                                self.params['n_parts'], 
                                                self.params['n_limbs'], s+1)

            fSize= self.params['n_limbs']+ self.params['n_parts'] + inputFeatures

            self.stagesL1.append(stageL1)
            self.stagesL2.append(stageL2)



    def init_cascade_squeeze(self, inputFeatures):
        print('[INFO] (%s) Prediction cascade squeezenet layers!' % self.__class__.__name__)

        fSize= inputFeatures
        for s in range(self.params['n_stages']):
            stageL1, stageL2= get_squeeze_stage_(fSize, 
                                                 self.params['feat_chann_width'], 
                                                 self.params['n_parts'], 
                                                 self.params['n_limbs'], s+1)

            fSize= self.params['n_parts'] + self.params['n_limbs'] + inputFeatures

            self.stagesL1.append(stageL1)
            self.stagesL2.append(stageL2)



    ##
    # @brief Initalization of the first stage in the model. This stage is 
    # particular of the other stages, this is why has a separate 
    # method for initalization.
    # @param inputFeatures Number of feature maps that the stage will have as input.
    #
    def init_stage1(self, inputFeatures):
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        the_channels= self.params['feat_chann_width']
        stageL1 = nn.Sequential(OrderedDict([\
                       ('Conv1_S1_L1', nn.Conv2d(inputFeatures, the_channels, kernel_size=3, stride=1, padding=1)), \
                       ('Relu1_S1_L1', nn.ReLU(inplace=True)),\
                       ('Conv2_S1_L1', nn.Conv2d(the_channels, the_channels, kernel_size=3, stride=1, padding=1)),\
                       ('Relu2_S1_L1', nn.ReLU(inplace=True)),\
                       ('Conv3_S1_L1', nn.Conv2d(the_channels, the_channels, kernel_size=3, stride=1, padding=1)),\
                       ('Relu3_S1_L1', nn.ReLU(inplace=True)),\
                       ('Conv4_S1_L1', nn.Conv2d(the_channels, the_channels, kernel_size=1, stride=1,padding=0)),\
                       ('Relu4_S1_L1', nn.ReLU(inplace=True)),\
                       ('Conv5_S1_L1', nn.Conv2d(the_channels, self.params['n_limbs'], kernel_size=1, stride=1, padding=0))\
                       ]))


        stageL2 = nn.Sequential(OrderedDict([\
                        ('Conv1_S1_L2',nn.Conv2d(inputFeatures, the_channels, kernel_size=3, stride=1, padding=1)), \
                        ('Relu1_S1_L2',nn.ReLU(inplace=True)),\
                        ('Conv2_S1_L2',nn.Conv2d(the_channels, the_channels, kernel_size=3, stride=1, padding=1)),\
                        ('Relu2_S1_L2',nn.ReLU(inplace=True)),\
                        ('Conv3_S1_L2',nn.Conv2d(the_channels, the_channels, kernel_size=3, stride=1,padding=1)),\
                        ('Relu3_S1_L2',nn.ReLU(inplace=True)),\
                        ('Conv4_S1_L2',nn.Conv2d(the_channels, the_channels, kernel_size=1, stride=1, padding=0)),\
                        ('Relu4_S1_L2',nn.ReLU(inplace=True)),\
                        ('Conv5_S1_L2',nn.Conv2d(the_channels, self.params['n_parts'], kernel_size=1, stride=1, padding=0))\
                        ]))

        self.stagesL1.append(stageL1)
        self.stagesL2.append(stageL2)



    ##
    # @brief Initialize the convolutional layers for the stage and given branch.
    # @param stageNum Number of stage in the architecture.
    # @param inputSize Number of input channels to the stage.
    # @param branchNum Number of branch that the stage corresponds to.
    # @return List containing all the convolutional layers definition for the stage.
    #
    def init_branch_n(self, stageNum, inputSize, outputSize, branchNum):
        # Stage layer definition each sublist referes to:
        # [# of input channels, # of output channels, kernelsize, stride, padding]
        kSize = 7
        pad   = 3
        defs = [[inputSize, self.params['feat_chann_width'], kSize, 1, pad],\
                [self.params['feat_chann_width'],self.params['feat_chann_width'],kSize,1,pad],\
                [self.params['feat_chann_width'],self.params['feat_chann_width'],kSize,1,pad],\
                [self.params['feat_chann_width'],self.params['feat_chann_width'],kSize,1,pad],\
                [self.params['feat_chann_width'],self.params['feat_chann_width'],kSize,1,pad],\
                [self.params['feat_chann_width'],self.params['feat_chann_width'],1,1,0]]

      
        branch = []
        # TODO This was made to fit caffe layers. Need to make smaller code
        for i in range(len(defs)):
            conv2d = nn.Conv2d(defs[i][0], defs[i][1],\
                                kernel_size=defs[i][2],\
                                stride=defs[i][3],\
                                padding=defs[i][4])


            # Add relu and name to the layer
            relu     = nn.ReLU(inplace=True)
            convName = 'Conv%d_S%d_L%d' % (i+1, stageNum, branchNum)
            reluName = 'Relu%d_S%d_L%d' % (i+1, stageNum, branchNum)
            # Append current
            branch  += [(convName, conv2d), (reluName, relu)]

        # Last layer of the branch block
        branch += [('Conv7_S%d_L%d'%(stageNum, branchNum), \
                    nn.Conv2d(self.params['feat_chann_width'], outputSize, 
                              kernel_size=1,\
                              stride=1,\
                              padding=0))]

        return branch




    ##
    # @brief Initalization of stage n the network with n>1. The function effectively
    # increases the size of stagesList of the module.
    # @param stageNum Number of stage in the network to be initialized.
    # @param inputSize Numer of feature maps that the stage will have as input.
    #
    def init_stage_n(self, stageNum, inputSize):
        print('[INFO] (%s) Intializing branches' % self.__class__.__name__)
        # Limbs branch
        branchL1 = self.init_branch_n(stageNum, inputSize, self.params['n_limbs'], 1)
        stageL1 = nn.Sequential(OrderedDict(branchL1))
        self.stagesL1.append(stageL1)

        # Parts branch
        branchL2 = self.init_branch_n(stageNum, inputSize, self.params['n_parts'], 2)
        stageL2 = nn.Sequential(OrderedDict(branchL2))
        self.stagesL2.append(stageL2)








        








