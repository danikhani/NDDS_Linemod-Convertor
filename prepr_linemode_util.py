import os
import shutil
from PIL import Image
import yaml
import io
import json
import numpy as np


# makes the folder structure
def make_empty_folder(main_path,folder):
    folder_path = os.path.join(main_path, folder)
    try:
        shutil.rmtree(folder_path)
        os.mkdir(folder_path)
    except FileNotFoundError:
        os.mkdir(folder_path)
    return folder_path

# copy the files and rename them in the new folder.
def copy_and_rename(raw_data_directory,picturename,index,destination_path):
    source_file = os.path.join(raw_data_directory, picturename)
    mask_name = '{}.png'.format(index)
    edited_filename = picturename.replace(picturename, mask_name)
    new_destination = os.path.join(destination_path, edited_filename)
    shutil.copy2(source_file, new_destination)


def parse_camera_intrinsic_yml(file_path, number_of_frame, cam_K_array, depth_scale):
    camera_intrinsic = {
        number_of_frame: {
            'cam_K': cam_K_array,
            'depth_scale': depth_scale,
        }
    }
    with io.open(file_path, 'a', encoding='utf8') as outfile:
        yaml.dump(camera_intrinsic, outfile, default_flow_style=None)


# {0: [{'cam_R_m2c': [0.0963063, 0.99404401, 0.0510079, 0.57332098, -0.0135081, -0.81922001, -0.81365103, 0.10814, -0.57120699],
# 'cam_t_m2c': [-105.3577515, -117.52119142, 1014.8770132], 'obj_bb': [244, 150, 44, 58], 'obj_id': 1}], 1: ... }
def parse_groundtruth_yml(file_path, number_of_frame, cam_R_m2c_array, cam_t_m2c_array, obj_bb, obj_id):
    ground_truth = {
        number_of_frame: [{
            'cam_R_m2c': cam_R_m2c_array,
            'cam_t_m2c': cam_t_m2c_array,
            'obj_bb': obj_bb,
            'obj_id': obj_id,
        }]
    }
    with io.open(file_path, 'a', encoding='utf8') as outfile:
        yaml.dump(ground_truth, outfile, default_flow_style=None)

def get_camera_intrinsic(raw_data_directory,json_file):
    # https://github.com/thodan/sixd_toolkit/blob/master/doc/sixd_2017_datasets_format.md
    # cam_K - 3x3 intrinsic camera matrix K (saved row-wise)
    # https://github.com/zju3dv/clean-pvnet/issues/118:
    # [572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0]
    # [fx, 0, ux, 0, fy, uy, 0, 0, 1]

    #
    # {"fx": 614.4744262695312, "fy": 614.4745483398438, "height": 480, "width": 640, "ppy": 233.29214477539062, "ppx": 308.8282470703125, "ID": "620201000292"}
    #
    # If you don't know your camera's intrinsic, you can put a rough estimation in.
    # All parameters required are fx, fy, cx, cy, where commonly fx = fy and equals to the width of the image and cx and cy is the center of the image.
    # For example, for a 640 x 480 resolution image, fx, fy = 640, cx = 320, cy = 240.
    # {'cam_K': [572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0], 'depth_scale': 1.0}
    source_file = os.path.join(raw_data_directory, json_file)
    with open(source_file, 'r') as file:
        annotation = json.loads(file.read())

    camera_settings = annotation['camera_settings']
    intrinsic_settings = camera_settings[0]['intrinsic_settings']
    captured_image_size = camera_settings[0]['captured_image_size']
    # intrinsic matrix
    fx = float(intrinsic_settings['resX'])
    ux = float(intrinsic_settings['cx'])
    fy = float(intrinsic_settings['resY'])
    uy = float(intrinsic_settings['cy'])
    s = float(intrinsic_settings['s'])
    cam_K = [fx, 0, ux, 0, fy, uy, 0, 0, 1.0]
    depth_scale = 1.0
    # image size:
    image_size = [captured_image_size['width'], captured_image_size['height']]

    return cam_K, depth_scale, image_size


def get_groundtruth_data(raw_data_directory,json_file):
    source_file = os.path.join(raw_data_directory, json_file)
    with open(source_file, 'r') as file:
        annotation = json.loads(file.read())

    object_from_annotation = annotation['objects']
    # object_class = object_from_annotation[0]["class"]

    # translation
    translation = np.array(object_from_annotation[0]['location']) * 10
    translation = np.round(translation, 8)

    # rotation
    rotation = np.asarray(object_from_annotation[0]['pose_transform'])[0:3, 0:3]
    rotation = np.dot(rotation, np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]))
    rotation = np.dot(rotation.T, np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))
    # make list from rows and add them to a single array
    rotation = np.round(rotation, 8)
    rotation_list = list(rotation[0, :]) + list(rotation[1, :]) + list(rotation[2, :])

    # get bounding box:
    bounding_box = object_from_annotation[0]['bounding_box']
    bb_top_left = np.array(bounding_box['top_left'])
    bottom_right = np.array(bounding_box['bottom_right'])
    width_height = bottom_right - bb_top_left
    # round to int since they are pixel coord. Also make a list out of them for yaml export
    x_y = list(np.rint(bb_top_left))
    dx_dy = list(np.rint(width_height))

    obj_bb = list(map(int, x_y)) + list(map(int, dx_dy))

    # yaml.dump doenst support numpy arrays. Here they are converted to python array
    cam_R_m2c_array =[]
    cam_t_m2c_array = []
    for r in rotation_list:
        cam_R_m2c_array.append(float(r))

    for t in translation:
        cam_t_m2c_array.append(float(t))

    return cam_R_m2c_array, cam_t_m2c_array, obj_bb