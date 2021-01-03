import os
import shutil
from PIL import Image
import yaml
import io


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

# {0: {'cam_K': [572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0], 'depth_scale': 1.0}, ...}
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