import os
import io
import shutil
from scipy.spatial.transform import Rotation as rotate
import numpy as np
import json
import yaml
import random

def make_empty_folder(main_path,folder):
    folder_path = os.path.join(main_path, folder)
    try:
        shutil.rmtree(folder_path)
        os.mkdir(folder_path)
    except FileNotFoundError:
        os.mkdir(folder_path)
    return folder_path

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
        yaml.dump(camera_intrinsic, outfile, default_flow_style=None,width=1000)

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
        yaml.dump(ground_truth, outfile, default_flow_style=None,width=1000)

def get_camera_intrinsic(raw_data_directory,json_file):
    # https://github.com/thodan/sixd_toolkit/blob/master/doc/sixd_2017_datasets_format.md
    # cam_K - 3x3 intrinsic camera matrix K (saved row-wise)
    # https://github.com/zju3dv/clean-pvnet/issues/118:
    # [fx, 0, ux, 0, fy, uy, 0, 0, 1]
    source_file = os.path.join(raw_data_directory, json_file)
    with open(source_file, 'r') as file:
        annotation = json.loads(file.read())

    camera_settings = annotation['camera_settings']
    intrinsic_settings = camera_settings[0]['intrinsic_settings']
    captured_image_size = camera_settings[0]['captured_image_size']
    # intrinsic matrix
    fx = float(intrinsic_settings['fx'])
    ux = float(intrinsic_settings['cx'])
    fy = float(intrinsic_settings['fy'])
    uy = float(intrinsic_settings['cy'])
    s = float(intrinsic_settings['s'])
    cam_K = [fx, 0, ux, 0, fy, uy, 0, 0, 1.0]
    depth_scale = 1.0
    # image size:
    image_size = [captured_image_size['width'], captured_image_size['height']]

    return cam_K, depth_scale, image_size


def get_groundtruth_data(raw_data_directory,json_file,length_multipler):
    source_file = os.path.join(raw_data_directory, json_file)
    with open(source_file, 'r') as file:
        annotation = json.loads(file.read())

    object_from_annotation = annotation['objects']
    # object_class = object_from_annotation[0]["class"]

    translation = np.array(annotation['objects'][0]['location'])*length_multipler # change box length units

    quaternion_obj2cam = rotate.from_quat(np.array(annotation['objects'][0]['quaternion_xyzw']))
    quaternion_cam2world = rotate.from_quat(np.array(annotation['camera_data']['quaternion_xyzw_worldframe']))
    quaternion_obj2world = quaternion_obj2cam * quaternion_cam2world
    r1 = rotate.from_euler('x', 90, degrees=True)
    rotation_matrix = np.dot(quaternion_obj2world.as_dcm(), r1.as_dcm())
    rotation_list = list(rotation_matrix[0, :]) + list(rotation_matrix[1, :]) + list(rotation_matrix[2, :])

    # get bounding box:
    bounding_box = object_from_annotation[0]['bounding_box']
    xmin,ymin = np.array(bounding_box['top_left'])
    xmax,ymax = np.array(bounding_box['bottom_right'])
    # round the pixels
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    deltax= xmax-xmin
    deltay = ymax - ymin
    obj_bb =[ymin, xmin,deltay,deltax]

    # yaml.dump doenst support numpy arrays. Here they are converted to python array
    cam_R_m2c_array =[]
    cam_t_m2c_array = []
    for r in rotation_list:
        cam_R_m2c_array.append(float(r))

    for t in translation:
        cam_t_m2c_array.append(float(t))

    return cam_R_m2c_array, cam_t_m2c_array, obj_bb

# generate test and train data numbers:
def make_training_set(start,end,training_percent):
    # Generate 'n' unique random numbers within a range
    number_of_test = int((1-training_percent)*(end-start))
    test_frames = random.sample(range(start, end), number_of_test)
    test_frames.sort()
    training_frames = list(range(start, end))
    # removing test_frames from the trainings_frame
    for i in test_frames:
        if i in training_frames:
            training_frames.remove(i)

    return training_frames,test_frames
