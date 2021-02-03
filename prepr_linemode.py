import os
from natsort import natsorted
import lib.utils.base_utils as util


# make the subfolder structure and return their paths
def create_saving_folders(saving_path,object_id):

    #structure of data:
    main_data_path = os.path.join(saving_path, 'data/{}'.format(object_id))
    main_model_path = os.path.join(saving_path, 'models')

    depth_folder = 'depth'
    mask_folder = 'mask'
    rgb_folder = 'rgb'

    # yaml files path
    ground_truth_path = main_data_path+'/gt.yml'
    camera_info_path = main_data_path+'/info.yml'

    # test.txt & train.txt path
    test_txt_path = main_data_path + '/test.txt'
    train_txt_path = main_data_path + '/train.txt'


    # create folder structure
    try:
        os.makedirs(main_data_path)
    except FileExistsError:
        print(main_data_path + ' exist')
    try:
        os.makedirs(main_model_path)
    except FileExistsError:
        print(main_model_path + ' exist')
    # create folders
    depth_path = util.make_empty_folder(main_data_path,depth_folder)
    mask_path = util.make_empty_folder(main_data_path,mask_folder)
    rgb_path = util.make_empty_folder(main_data_path,rgb_folder)

    #removing and recreating the yml files:
    try:
        os.remove(ground_truth_path)
    except FileNotFoundError:
        print(ground_truth_path + ' doesnt exist')

    try:
        os.remove(camera_info_path)
    except FileNotFoundError:
        print(camera_info_path + ' doesnt exist')

    return depth_path,mask_path,rgb_path,main_model_path,ground_truth_path,camera_info_path,test_txt_path,train_txt_path


def make_linemode_dataset(raw_NDDS_directory,saving_path,object_id,length_scale,train_percentage):

    # kind of data which are going to extracted
    depth_ending = '.depth.cm.16.png'
    mask_ending = '.cs.png'
    rgb_ending = '.png'
    all_endings = ['.cs.png', '.16.png', '.8.png', '.micon.png', '.depth.png', '.is.png']

    # get saving paths
    depth_path,mask_path,rgb_path,main_model_path,ground_truth_path,camera_info_path,test_txt_path,train_txt_path = create_saving_folders(saving_path,object_id)

    # get camera instrinsics
    cam_K, depth_scale, image_size = util.read_camera_intrinsic_json(raw_NDDS_directory + '/_camera_settings.json')

    mask_index = 0
    depth_index = 0
    rgb_index = 0
    sorted_list_of_files = natsorted(os.listdir(raw_NDDS_directory))
    for data_name in sorted_list_of_files:
        # find mask data
        if data_name.endswith(mask_ending):
            util.copy_and_rename(raw_NDDS_directory, data_name, mask_index, mask_path)
            mask_index += 1
        # find depth data
        if data_name.endswith(depth_ending):
            util.copy_and_rename(raw_NDDS_directory, data_name, depth_index, depth_path)
            depth_index += 1
        # find rgb data
        if data_name.endswith(rgb_ending) and not any(data_name.endswith(x) for x in all_endings):
            util.copy_and_rename(raw_NDDS_directory, data_name, rgb_index, rgb_path)
            rgb_index += 1


    # writing the yaml filses:
    yml_index = 0
    for data_name in sorted_list_of_files:
        # writing gt.yml
        if data_name.endswith('.json') and not data_name.endswith('settings.json'):
            # get the arrays from the json files.
            cam_R_m2c_array, cam_t_m2c_array, obj_bb = util.get_groundtruth_data(raw_NDDS_directory, data_name,length_scale)
            # save arrays in gt.yml
            util.parse_groundtruth_yml(ground_truth_path, yml_index, cam_R_m2c_array, cam_t_m2c_array, obj_bb,
                                       object_id)
            # save info.yml
            util.parse_camera_intrinsic_yml(camera_info_path, yml_index, cam_K, depth_scale)
            yml_index += 1

    #generate training and test_files
    training_frames,test_frames = util.make_training_set(0,rgb_index,train_percentage)
    with open(train_txt_path, 'w') as f:
        for item in training_frames:
            f.write("%s\n" % item)

    with open(test_txt_path, 'w') as f:
        for item in test_frames:
            f.write("%s\n" % item)

    print('data generated!')
    print('yml_index,mask_index,depth_index,rgb_index are:')
    print(yml_index,mask_index,depth_index,rgb_index)





