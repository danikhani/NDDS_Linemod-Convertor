from prepr_linemode import make_linemode_dataset
from visualize_dataset import vis_bb

import argparse

def parse_input():
    parser = argparse.ArgumentParser(description='Linemode dataset generator')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_input()
    make_linemode_dataset('datasets/blech', 'datasets/blech', 17)

    # to visulize the generation:
    model_number = 1
    frame_number = 23
    dataset_main_path = 'datasets/Linemod_preprocessed/'
    rgb_image_path = dataset_main_path + 'data/01/rgb/0072.png'
    models_info_yml_path = dataset_main_path + 'models/models_info.yml'
    gt_yml_path = dataset_main_path + 'data/01/gt.yml'
    info_yml_path = dataset_main_path + 'data/01/info.yml'
    img_export_folder = 'datasets/testfolder'

    model_number = 17
    frame_number = 26
    dataset_main_path = 'datasets/blech/generated_dataset/'
    rgb_image_path = dataset_main_path + 'data/17/rgb/{}.png'.format(frame_number)
    models_info_yml_path = 'datasets/blech_linemode/models_info.yml'
    gt_yml_path = dataset_main_path + 'data/17/gt.yml'
    info_yml_path = dataset_main_path + 'data/17/info.yml'
    img_export_folder = 'datasets/testfolder/owndata'

    vis_bb(rgb_image_path, models_info_yml_path, gt_yml_path, info_yml_path, img_export_folder, frame_number,model_number)
