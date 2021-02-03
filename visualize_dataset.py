import os
import numpy as np
import cv2
import yaml
from PIL import Image

import lib.utils.visualisation_utils as v_util


####### read_yml_file ##########
def generate_paths(dataset_main_path,model_number):
    models_info_yml_path = os.path.join(dataset_main_path,'models/models_info.yml')
    gt_yml_path = os.path.join(dataset_main_path, 'data/{}/gt.yml'.format(model_number))
    info_yml_path = os.path.join(dataset_main_path, 'data/{}/info.yml'.format(model_number))
    rgb_path = os.path.join(dataset_main_path, 'data/{}/rgb'.format(model_number))
    img_export_path = os.path.join(dataset_main_path, 'data/{}/rgb'.format(model_number))

    return models_info_yml_path,gt_yml_path,info_yml_path,rgb_path

####### read_yml_file ##########
def read_yml_file(yml_path):
    with open(yml_path) as file:
        yml_file = yaml.load(file, Loader=yaml.FullLoader)
    return yml_file

###### load different files from dataset folder ##########
def load_linemod_data(models_info_yml_path, gt_yml_path, info_yml_path, frame_number, model_number):

    ########### load model_info.yml ################
    model_info = read_yml_file(models_info_yml_path)
    # get the model_number from the file
    current_model = model_info[model_number]
    # calculate real 3d_points of cuboid
    bb3d_points = v_util.get_bbox_3d(current_model)

    ########### load gt.yml ################
    gt = read_yml_file(gt_yml_path)
    # load rotation matrix
    cam_R_m2c = gt[frame_number][0]['cam_R_m2c']
    # convert rotation to 3x3 matrix
    cam_R_m2c = np.reshape(np.array(cam_R_m2c), newshape=(3, 3))
    # load translation matrix
    cam_t_m2c = np.array(gt[frame_number][0]['cam_t_m2c'])
    # load 2d bounding box
    obj_bb = gt[frame_number][0]['obj_bb']

    ########### load info.yml ################
    info = read_yml_file(info_yml_path)
    #load camera settings
    cam_K = info[frame_number]['cam_K']
    # convert intrinsic to 3x3 matrix
    cam_K = np.reshape(np.array(cam_K), newshape=(3, 3))
    print(cam_K)

    return bb3d_points,cam_R_m2c, cam_t_m2c, obj_bb, cam_K



def vis_bb(dataset_main_path,frame_number,model_number,export_path):
    # get paths
    models_info_yml_path, gt_yml_path, info_yml_path, rgb_path = generate_paths(dataset_main_path,model_number)

    # get the data
    b3d_points, cam_R_m2c, cam_t_m2c, obj_bb, cam_K = load_linemod_data(models_info_yml_path, gt_yml_path,
                                                                        info_yml_path, frame_number, model_number)

    # load rgb image as base.
    rgb_image_path = os.path.join(rgb_path, '{}.png'.format(frame_number))
    loaded_rgb = Image.open(rgb_image_path)

    ###### draw 2d bounding box ######
    rgb_2d_bbox = np.array(loaded_rgb.copy())
    cv2.rectangle(rgb_2d_bbox, (obj_bb[0], obj_bb[1]), (obj_bb[0]+obj_bb[2], obj_bb[1]+obj_bb[3]), (255, 0, 0), 2)

    ###### draw 3d bounding box ######
    rgb_3d_bbox = np.array(loaded_rgb.copy())
    projected_points = v_util.project_bbox_3D_to_2D(b3d_points, cam_R_m2c, cam_t_m2c, cam_K,
                                                    append_centerpoint=False)
    v_util.draw_bbox_8_2D(rgb_3d_bbox, projected_points)

    # show images
    print('want to show image but dont know')
    cv2.imshow('2D bounding box',rgb_2d_bbox)
    cv2.imshow('3D bounding box', rgb_3d_bbox)
    # press any key to close
    print("Press any key to close the images. DO NOT clsoe the images by clicking on X")
    cv2.waitKey(0)


    # set the saving paths
    if export_path is not None:
        img_path_2d =  export_path + '/exported_images/{}_{}.bbox_2d.png'.format(model_number,frame_number)
        img_path_3d = export_path + '/exported_images/{}_{}.bbox_3d.png'.format(model_number,frame_number)
        cv2.imwrite(img_path_2d, rgb_2d_bbox)
        cv2.imwrite(img_path_3d, rgb_3d_bbox)
