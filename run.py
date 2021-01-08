from prepr_linemode import make_linemode_dataset
from visualize_dataset import vis_bb

import argparse

def parse_input():
    parser = argparse.ArgumentParser(description='Linemode dataset generator')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_input()
    # generate the linemode
    make_linemode_dataset('datasets/test_dataset/NDDS_generated', 'datasets/test_dataset', 17)


    model_number = 17
    frame_number = 0
    dataset_main_path = 'datasets/test_dataset/generated_dataset/'
    export_path = 'datasets/test_dataset'
    vis_bb(dataset_main_path, frame_number,model_number,export_path)
