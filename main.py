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


import cv2
import sys
import os
import argparse
import json
import numpy as np
import torch


import utils.utils as utils
import pose_machines.pose_machines as pose_machines
import pose_machines.pm_utils as pm_utils
from   part_association.pose_construction import DepthPoseConstructor


#### get available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply efficient pose machines on given depth image.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--img_path', type=str, help='Path (full) to image.')
    parser.add_argument('--model_config', type=str, default='config_files/RPM_config.json', help='Model config parameters.')
    parser.add_argument('--dataset_type', type=str, default='panoptic', help='Dataset model. Options: panoptic or dih')
    parser.add_argument('--conf_thresh', type=float, default=0.1, help='Confidence map threshold for NMS.')
    parser.add_argument('--paf_thresh', type=float, default=0.05, help='Part affinity field threshold.')
    args= parser.parse_args()

    part_map= pm_utils.PMDepthUtils().jointMap

    with torch.no_grad():
        ### load params and create model
        params= json.load(open(args.model_config))
        pm_instance= pose_machines.PoseMachine(params['model_params'])
        pm_instance.load_state_dict(torch.load(params[args.dataset_type+'_trained_model'], map_location=device))
        pm_instance.to(device)
        pm_instance.eval()

        #### pose constructor
        constructor = DepthPoseConstructor(args.conf_thresh, args.paf_thresh)
        n_parts= params['model_params']['n_parts']

        ### normalizer
        normalization= utils.apply_depth_normalization_single_channel_cube

        #### depth image is loaded in metters
        img= utils.load_image(args.img_path)

        if args.dataset_type=='dih':
            r_y, r_x, r_width, r_height  = 2, 64, 799, 537
            img= img[r_y:(r_y+r_height), r_x:(r_x+r_width)].copy()

        img= cv2.medianBlur(img, 3)
        color_img= utils.convert_to_uchar(img, 8.0)
        color_img= cv2.cvtColor(color_img, cv2.COLOR_GRAY2BGR)

        #### Set the training aspect ratio
        nHeight= 368
        nWidth=  444 

        #### resize and normalize
        test_image= cv2.resize(img, (nWidth, nHeight), interpolation=cv2.INTER_CUBIC)
        test_image= normalization(test_image)
        test_image= torch.from_numpy(test_image).float()

        #### forward passs
        (pafs_maps, conf_maps), feats= pm_instance(test_image.to(device))

        #### BxCxWxH otput; remove batch dimension
        conf_maps= np.squeeze(conf_maps.data.cpu().numpy())
        pafs_maps= np.squeeze(pafs_maps.data.cpu().numpy())

        ### set to dims WxHxC
        conf_maps= np.transpose(conf_maps, (1,2,0))
        pafs_maps= np.transpose(pafs_maps, (1,2,0))


        #### reshape to proceed on finding pose
        conf_maps = cv2.resize(conf_maps, (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        conf_maps = cv2.resize(conf_maps, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        pafs_maps = cv2.resize(pafs_maps, (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        pafs_maps = cv2.resize(pafs_maps, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)


        #### single map set without background prediction
        heatmaps_mat = np.concatenate((conf_maps, pafs_maps), axis=2)
        heatmaps_mat = np.delete(heatmaps_mat, (n_parts-1), axis=2)

        #### do pose association
        candidates, subset = constructor.part_association(heatmaps_mat)
        canvas = constructor.visualize_connection(color_img, candidates, subset)
        prediction = constructor.extract_keypoints(candidates, subset)


        ##### extract the points to save detections in image
        result_list=[]
        total_conf=0.
        for pred in prediction:
            detection = {'keypoints':[[0.0,0.0,0.0,0.0]]*(n_parts-1), 
                          'score':pred['score']}

            detScore = 0.0
            for point in pred['points']:
                x, y, partId, score = point[0], point[1], point[3], point[2]
                partIdx = part_map[partId] 
                detection['keypoints'][partIdx] = ((x, y, 1.0, score))
                detScore+= score

            detection['score'] = detScore
            total_conf+= detScore
            result_list.append(detection)


        ##### save results
        the_path= os.path.join("results",args.dataset_type)
        if not os.path.exists(the_path):
            os.makedirs(the_path)

        with open(os.path.join(the_path, "detections_%s.json"%params['model_alias']), "w") as file_:
            json.dump(result_list, file_, indent=4)

        ### visualize gaussian blobs and magnitude of confidence maps 
        conf_img, paf_img =  pm_utils.visualize_confmaps(color_img, conf_maps[:,:,0:17], pafs_maps)

        cv2.imwrite(os.path.join(the_path, 'partmaps_%s.jpg'%params['model_alias']), conf_img)
        cv2.imwrite(os.path.join(the_path, 'limbmaps_%s.jpg'%params['model_alias']), paf_img)
        cv2.imwrite(os.path.join(the_path, 'canvas_%s_score_%f.jpg'%(params['model_alias'],total_conf)), canvas)








