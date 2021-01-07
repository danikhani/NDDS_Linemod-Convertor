import os
import numpy as np
import cv2
import yaml
import random
import copy
from plyfile import PlyData
from PIL import Image
import matplotlib.pyplot as plt


import test_lib as tlib



def load_linemod_data(models_info_yml_path,gt_yml_path,info_yml_path,frame_number,model_number):
    # load points from model_info.yml
    with open(models_info_yml_path) as file:
        model_info = yaml.load(file, Loader=yaml.FullLoader)
    current_model = model_info[model_number]
    # calculate 3dpoints
    bb3d_points = tlib.get_bbox_3d(current_model)

    #get groundtruth info from each scene
    with open(gt_yml_path) as file:
        gt = yaml.load(file, Loader=yaml.FullLoader)
    # load rotation and t matrices
    cam_R_m2c = gt[frame_number][0]['cam_R_m2c']
    cam_R_m2c = np.reshape(np.array(cam_R_m2c), newshape=(3, 3))
    cam_t_m2c = np.array(gt[frame_number][0]['cam_t_m2c'])
    obj_bb = gt[frame_number][0]['obj_bb']

    with open(info_yml_path) as file:
        info = yaml.load(file, Loader=yaml.FullLoader)
    #load camera settings
    cam_K = info[frame_number]['cam_K']
    cam_K = np.reshape(np.array(cam_K), newshape=(3, 3))
    depth_scale = info[frame_number]['depth_scale']

    return bb3d_points,cam_R_m2c,cam_t_m2c,obj_bb,cam_K,depth_scale


def vis_bb(rgb_image_path,models_info_yml_path,gt_yml_path,info_yml_path,img_export_folder,frame_number,model_number):
    # set the saving paths
    img_path_2d =  img_export_folder + '/{}_{}.bbox_2d.png'.format(model_number,frame_number)
    img_path_3d = img_export_folder + '/{}_{}.bbox_3d.png'.format(model_number,frame_number)

    # get the data
    b3d_points, cam_R_m2c, cam_t_m2c, obj_bb, cam_K, depth_scale = load_linemod_data(models_info_yml_path,gt_yml_path,info_yml_path,frame_number,model_number)

    # load rgb image as base.
    loaded_rgb = Image.open(rgb_image_path)

    #draw 2d bounding box
    rgb_2d_bbox = np.array(loaded_rgb.copy())
    cv2.rectangle(rgb_2d_bbox, (obj_bb[0], obj_bb[1]), (obj_bb[0]+obj_bb[2], obj_bb[1]+obj_bb[3]), (255, 0, 0), 2)
    cv2.imwrite(img_path_2d, rgb_2d_bbox)
    #draw 3d bounding box
    rgb_3d_bbox = np.array(loaded_rgb.copy())
    projected_points = tlib.project_bbox_3D_to_2D(b3d_points, cam_R_m2c, cam_t_m2c, cam_K,
                               append_centerpoint=False)
    tlib.draw_bbox_8_2D(rgb_3d_bbox, projected_points)
    cv2.imwrite(img_path_3d, rgb_3d_bbox)

    img = cv2.imread(img_path_2d, 0)

    plt.imshow(img)
    plt.show()