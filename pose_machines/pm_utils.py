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
import numpy as np
import cv2


sys.path.append('./../')

import utils.utils as CvUtils
import torch.nn.init as nninit


# Colors to paint the keypoints and limbs
colors = CvUtils.colors


def normal_init_(layer, mean_, sd_, bias):
    classname = layer.__class__.__name__
    # Only use the convolutional layers of the module
    if (classname.find('Conv') != -1 ) or (classname.find('Linear')!=-1):
        print('[INFO] (normal_init) Initializing layer {}'.format(classname))
        layer.weight.data.normal_(mean_, sd_)
        layer.bias.data.fill_(bias)


def normal_init(module, mean_=0, sd_=0.004, bias=0.0):
    moduleclass= module.__class__.__name__
    try:
        for layer in module:
            if layer.__class__.__name__ == 'Sequential':
                for l in layer:
                    normal_init_(l, mean_, sd_, bias)
            else:
                normal_init_(layer, mean_, sd_, bias)
    except TypeError:
        normal_init_(module, mean_, sd_, bias)


def xavier_init(layer):
    classname = layer.__class__.__name__
    if (classname.find('Conv') != -1 ) or (classname.find('Linear')!=-1):
        print('[INFO] (xavier_init) Initializing layer {}'.format(classname))
        nninit.xavier_normal(layer.weight.data)
        # nninit.xavier_normal(layer.bias.data)
        layer.bias.data.zero_()

def layer_init(module):
    moduleclass= module.__class__.__name__
    try:
        for layer in module:
            if layer.__class__.__name__ == 'Sequential':
                for l in layer:
                    xavier_init(l)
            else:
                xavier_init(layer)
    except TypeError:
        xavier_init(module)


##
# @brief Definitions of body model structure by means of keypoints and limbs
# for the depth CPM with data acquired from mocap sequence.
#
class PMDepthUtils:
    def __init__(self):
        ## Definition of all the 32 joints in the mocap sequence
        self.skeletonBoneNames = ["Hips", "LHipJoint", "LeftUpLeg", "LeftLeg", "LeftFoot",\
                                 "LeftToeBase", "LowerBack", "Spine", "Spine1", "LeftShoulder",\
                                 "LeftArm", "LeftForeArm", "LeftHand", "LThumb", "LeftFingerBase",\
                                 "LeftHandFinger1", "Neck", "Neck1", "Head", "RightShoulder",\
                                 "RightArm", "RightForeArm", "RightHand", "RThumb", "RightFingerBase",\
                                 "RightHandFinger1", "RHipJoint", "RightUpLeg", "RightLeg",\
                                 "RightFoot", "RightToeBase"]


        # Definition of the joints that compose the model structure to follow for detection
        self.partList = ["Neck", "Head", "HeadUp", "LeftUpLeg", "LeftLeg", "LeftFoot",\
                          "LeftArm", "LeftForeArm", "LeftHand", "RightUpLeg", "RightLeg", "RightFoot",\
                            "RightArm", "RightForeArm", "RightHand", "Eye1", "Eye2"]


        self.hPartList = ["Neck", "Head", "HeadTop", "LeftHip", "LeftKnee", "LeftAnkle",\
                          "LeftShoulder", "LeftElbow", "LeftWrist", "RightHip", "RightKnee", "RightAnkle",\
                            "RightShoulder", "RightElbow", "RightWrist", "LeftEye", "RightEye"]

        self.joint_id_to_name = {0: "Neck", 
                                 1: "Head", 
                                 2: "HeadTop", 
                                 3: "LeftHip", 
                                 4: "LeftKnee", 
                                 5: "LeftAnkle",
                                 6: "LeftShoulder", 
                                 7: "LeftElbow", 
                                 8: "LeftWrist", 
                                 9: "RightHip", 
                                 10: "RightKnee", 
                                 11: "RightAnkle",
                                 12: "RightShoulder", 
                                 13: "RightElbow", 
                                 14: "RightWrist", 
                                 15: "LeftEye", 
                                 16: "RightEye"}


        # Joint map contains the index of the joint in the partList
        self.jointMap = {}
        i = 0
        for joint in self.partList:
            self.jointMap[joint] = i
            i+=1

        self.hJointMap = {}
        i = 0
        for joint in self.hPartList:
            self.hJointMap[joint] = i
            i+=1

        # Definition of the  limbs by defining what joints will compose each of them
        self.limbList  = [[self.jointMap["Neck"],        self.jointMap["Head"]],
                          [self.jointMap["Head"],        self.jointMap["HeadUp"]],
                          [self.jointMap["Neck"],        self.jointMap["LeftUpLeg"]],
                          [self.jointMap["LeftUpLeg"],   self.jointMap["LeftLeg"]],
                          [self.jointMap["LeftLeg"],     self.jointMap["LeftFoot"]],
                          [self.jointMap["Neck"],        self.jointMap["LeftArm"]],
                          [self.jointMap["LeftArm"],     self.jointMap["LeftForeArm"]],
                          [self.jointMap["LeftForeArm"], self.jointMap["LeftHand"]],
                          [self.jointMap["Neck"],        self.jointMap["RightUpLeg"]],
                          [self.jointMap["RightUpLeg"],  self.jointMap["RightLeg"]],
                          [self.jointMap["RightLeg"],    self.jointMap["RightFoot"]],
                          [self.jointMap["Neck"],        self.jointMap["RightArm"]],
                          [self.jointMap["RightArm"],    self.jointMap["RightForeArm"]],
                          [self.jointMap["RightForeArm"],self.jointMap["RightHand"]],
                          [self.jointMap["Head"],       self.jointMap["Eye1"]],
                          [self.jointMap["Head"],        self.jointMap["Eye2"]]]

        self.cpm2rpm = {0 : -1,\
                        1 : self.jointMap["Neck"],\
                        2 : self.jointMap["RightArm"],\
                        3 : self.jointMap["RightForeArm"],\
                        4 : self.jointMap["RightHand"],\
                        5 : self.jointMap["LeftArm"],\
                        6 : self.jointMap["LeftForeArm"],\
                        7 : self.jointMap["LeftHand"],\
                        8 : self.jointMap["RightUpLeg"],\
                        9 : self.jointMap["RightLeg"],\
                        10: self.jointMap["RightFoot"],\
                        11: self.jointMap["LeftUpLeg"],\
                        12: self.jointMap["LeftLeg"], \
                        13: self.jointMap["LeftFoot"],\
                        14: self.jointMap["Eye2"],\
                        15: self.jointMap["Eye1"],\
                        16: -1, \
                        17: -1}




    def cpm_to_rpm_order(self, cpmPred, doMidHeadAvg=False):
        points = [(0,0,0)]*len(self.partList)
        extPoints = []

        for i in range(18):
            rpmIdx = self.cpm2rpm[i]

            if rpmIdx is -1:
                extPoints.append(cpmPred[i])
                continue

            points[rpmIdx] = cpmPred[i]

        if doMidHeadAvg:
            REar = cpmPred[16]
            LEar = cpmPred[17]

            if REar[2] > 0 or LEar[2] > 0:
                div = 2.0 if REar[2] > 0 and LEar[2] > 0 else 1.0
                midHead = [(x+y)/div for x,y in zip(REar, LEar)]
                midHead[2] = 1.0
                points[self.jointMap["Head"]] = midHead


        return points





def get_magnitude_paf_2(paf):
    mag = np.zeros((paf.shape[0], paf.shape[1], int(paf.shape[2]/2)), np.float32)
    c = 0

    for i in range(0,paf.shape[2],2):
        U = paf[:,:,i]
        V = paf[:,:,i+1]

        mag[:,:,c] = np.sqrt(U**2 + V**2)
        c=c+1


    return mag




def visualize_confmaps(img, partmaps, limbmaps):
    heats = partmaps.max(2)
    print('The heats shape is', heats.shape, img.shape)
    partMaps = CvUtils.get_overlayed_heatmap(heats, img.copy(),1.15, 0.5)

    mag   = get_magnitude_paf_2(limbmaps)
    mag = mag.max(2)
    mag[mag > 1.0] = 1.0
    limbMaps = CvUtils.get_overlayed_heatmap(mag, img.copy(),1.0, 0.5)


    return partMaps, limbMaps





def visualize_connection(img, candidates, subset, ellipse_=False):
    if len(subset.shape)==0:
        return img

    utils= CpmDepthUtils()
    limbSeq = utils.limbList
    limbSeq = np.asarray(limbSeq)

    canvas = img.copy()

    for num in range(subset.shape[0]): #= 1:size(subset,1)
        for i in range(17):
            index = int(subset[num, i])
            if index == -1:
              continue

            Y = int(candidates[index,0])
            X = int(candidates[index,1])
            cv2.circle(canvas, (X, Y), 5, colors[i], -1)
            # image = insertShape(image, 'FilledCircle', [X Y 5], 'Color', joint_color(i,:));


    for i in range(16):
        for num in range(subset.shape[0]):
            indexa = int(subset[num, limbSeq[i,0]])
            indexb = int(subset[num, limbSeq[i,1]])

            ## Check if any of the joint's limb was not detected
            if (indexa==-1) or (indexb==-1):
                continue

            cur_canvas = canvas.copy()

            Y = candidates[[indexa, indexb], 0]
            X = candidates[[indexa, indexb], 1]


            if ellipse_:
                stickwidth=4
                # This visualization is slow 
                if np.isnan(X).sum() == 0:
                    mx = np.mean(X)
                    my = np.mean(Y)
    
                    length = ((X[0]-X[1])** 2 + (Y[0]-Y[1])** 2)** 0.5
                    angle = math.degrees(math.atan2(X[0]-X[1], Y[0]-Y[1]))
                    polygon = cv2.ellipse2Poly((int(my),int(mx)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
                    cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
                    canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
            else:
                pa = (X[0], Y[0])
                pb = (X[1], Y[1])
                cv2.line(canvas, pa, pb, colors[i], 3)




    return canvas







