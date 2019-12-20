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
import torchvision.models



class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes

        ### squeeze definition
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        ### adding batch normalization
        self.squeeze_batchnorm  = nn.BatchNorm2d(squeeze_planes)
        self.squeeze_activation = nn.ReLU(inplace=True)


        ### expand 1x1
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_batchnorm = nn.BatchNorm2d(expand1x1_planes)
        self.expand1x1_activation = nn.ReLU(inplace=True)

        ### expand 3x3
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_batchnorm = nn.BatchNorm2d(expand3x3_planes)
        self.expand3x3_activation = nn.ReLU(inplace=True)


        self.squeezeSeq = nn.Sequential(*[self.squeeze, 
                                        self.squeeze_batchnorm, 
                                        self.squeeze_activation])

        self.expand1x1Seq = nn.Sequential(*[self.expand1x1,
                                          self.expand1x1_batchnorm,
                                          self.expand1x1_activation])

        self.expand3x3Seq = nn.Sequential(*[self.expand3x3,
                                          self.expand3x3_batchnorm,
                                          self.expand3x3_activation])



    def forward(self, x):
#      print('[INFO] (Fire@forward)', x.size())
      x=self.squeezeSeq(x)
      return torch.cat( [self.expand1x1Seq(x), self.expand3x3Seq(x)], 1)



class ResFireModule(nn.Module):
    def __init__(self, inputChannels, outputChannels, 
                  squeezePlanes, expand1x1Planes, expand3x3Planes):
        super(ResFireModule, self).__init__()


        self.fire = Fire(inputChannels, squeezePlanes, expand1x1Planes, expand3x3Planes)

        self.shortcut = None
        if inputChannels != outputChannels:
            self.shortcut = nn.Conv2d(in_channels=inputChannels,\
                                      out_channels=outputChannels,\
                                      kernel_size=1, \
                                      stride=1, \
                                      padding=0)



        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_features=outputChannels)


    def forward(self, x):
        res = x
        y = self.fire(x)

        # TODO Implement other shortcut
        # Some do upsample or downsample
        if self.shortcut is not None:
            res = self.shortcut(res)
        y = res + y

        y = self.bn1(y)
        self.relu1(y)

        return y






class SqueezeNetFeatureExtractor(nn.Module):
    def __init__(self, inputChannels, outputChannels, netType='flat'):
        super(SqueezeNetFeatureExtractor, self).__init__()

        self.inputChannels = inputChannels
        self.outputChannels= outputChannels
        thisname= self.__class__.__name__
        print("[INFO] ({}) Input channels: {} output channels: {}".format(thisname, self.inputChannels, self.outputChannels))

        ### params for squeeze net arch
        self.sqRatio= 0.125  ## squeeze net is  0.125 but squeeze paper suggest .75
        self.base   = 64   ## this is the one we should modify
        self.incr   = 64   ## this one also modify, put to zero to keep same
        self.freq   = 3
        self.pct    = 0.5   ## squeeze paper suggest this as tradeoff

        self.div    = 2

        self.netType = netType

        if netType== 'flat':
        ### feature extractor with feature maps fixed
            self.init_base_arch_prev()
        else:
        ### feature extractor with different number of feature maps
            self.init_base_arch()

    def init_base_arch(self):
        
        frontLayers = [nn.Conv2d(self.inputChannels, 64, 
                                kernel_size=7, stride=1, padding=3),\
                       nn.BatchNorm2d(64),\
                       nn.ReLU(inplace=True),
                       nn.AvgPool2d(kernel_size=2, stride=2, padding=0)]

        layers=[]
        inChann= 64
        doincr=[0,0,0,1,1,1,1]
        for l in range(7):
            #el= self.base + self.incr*((l)/self.freq)
            el = self.base + self.incr*doincr[l]
            s1x1=     int(self.sqRatio*el)
            e1x1=     int(self.pct*el)
            e3x3=     int(self.pct*el)
            outChan = e1x1+e3x3

            layers.append([inChann, outChan, s1x1, e1x1, e3x3])

            print('[INFO] (%s) Fire layer %d with following configuration' % (self.__class__.__name__, l) )
            print('[INFO] (%s) inChan= %d  outChan= %d s1x1= %d  e1x1= %d e3x3= %d'% 
                        (self.__class__.__name__, inChann, outChan, s1x1,e1x1, e3x3))
 
            inChann= e1x1+e3x3

        ### output the firenet channels divided
        ### self.outputChannels= layers[-1][1]/self.div

        downSam=layers[6][1]+self.incr

        midLayers = [Fire(layers[0][0], layers[0][2], 
                          layers[0][3], layers[0][4]),
                     ### residual fire module
                     ResFireModule(layers[1][0], layers[1][1], layers[1][2], 
                                   layers[1][3], layers[1][4]), 
                     Fire(layers[2][0], layers[2][2], 
                          layers[2][3], layers[2][4] ),
                     nn.AvgPool2d(kernel_size=2, stride=2, padding=0)]

                      ### start with residual
        backLayers = [ResFireModule(layers[3][0], layers[3][1], 
                                    layers[3][2], layers[3][3], layers[3][4]),
                      ### fire
                      Fire(layers[4][0], layers[4][2], 
                           layers[4][3], layers[4][4]),
                      ### residual
                      ResFireModule(layers[5][0], layers[5][1], 
                                    layers[5][2], layers[5][3], 
                                    layers[5][4]),
                      ### fire
                      Fire(layers[6][0], layers[6][2], layers[6][3], layers[6][4]),
                      nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

                      ### double the previous given the pooling layer
                      nn.Conv2d(layers[6][1], downSam, kernel_size=1, stride=1, padding=0),
                      nn.BatchNorm2d(downSam),
                      nn.ReLU(inplace=True),

                      ### reduce channels to outputchannels
                      nn.Conv2d(downSam, self.outputChannels, kernel_size=1, stride=1, padding=0),
                      nn.BatchNorm2d(self.outputChannels),
                      nn.ReLU(inplace=True)
                     ]

        
 
        print('[INFO] (%s) conv1x1 inchan= %d outchann= %d' % (self.__class__.__name__,layers[6][1], downSam))
        print('[INFO] (%s) conv1x1 inchan= %d outchann= %d' % (self.__class__.__name__,downSam, self.outputChannels))

        print('[INFO] (%s) This feature extractor outputs %d channels' %(self.__class__.__name__, self.outputChannels))

        self.frontLayers= nn.Sequential(*frontLayers)
        self.midLayers  = nn.Sequential(*midLayers)
        self.backLayers = nn.Sequential(*backLayers)



    def init_base_arch_prev(self):
        frontLayers = [nn.Conv2d(self.inputChannels, 64, 
                                kernel_size=7, stride=1, padding=3),\
                       nn.BatchNorm2d(64),\
                       nn.ReLU(inplace=True),
                       nn.AvgPool2d(kernel_size=2, stride=2, padding=0)]

        midLayers = [Fire(64, 16, 32, 32),
                     #Fire(64, 16, 32, 32),
                     ResFireModule(64,64, 16,32,32),  ### residual module
                     Fire(64, 16, 32, 32),
                     nn.AvgPool2d(kernel_size=2, stride=2, padding=0)]

        backLayers = [ResFireModule(64,64,16,32,32),
                      Fire(64, 16,32,32),
                      ResFireModule(64, 64, 16,32,32),
                      Fire(64, 16,32,32),
                      nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                      nn.Conv2d(64, self.outputChannels, kernel_size=1, stride=1, padding=0),
                      nn.BatchNorm2d(self.outputChannels),
                      nn.ReLU(inplace=True)
                     ]
        

        self.frontLayers= nn.Sequential(*frontLayers)
        self.midLayers  = nn.Sequential(*midLayers)
        self.backLayers = nn.Sequential(*backLayers)


    
    def forward(self, x):
        x=self.frontLayers(x)
        x=self.midLayers(x)
        return self.backLayers(x)



