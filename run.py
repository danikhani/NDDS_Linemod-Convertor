from prepr_linemode import make_linemode_dataset
from visualize_dataset import vis_bb
import write_model_info as model_info
import argparse



def parse_input():
    parser = argparse.ArgumentParser(description='Linemode dataset generator')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_input()

    ############ generate the linemode dataset #############
    # every model should have a uniq number
    # this number is also used to read the line in models_info.yml:
    model_number = 18
    # the frame to visulize:
    frame_number_to_visulize = 0
    # path of the folder:
    dataset_main_path = 'datasets/final_captures'
    # since the files are from cad and each one of them has a different length unit.
    # use this length factor if the visulized box is too small or too big!
    # use one of these values: 0.01,0.1,1,10,100 <<<<< the smaller value makes a bigger box
    length_multipler = 0.01

    ############ generate the linemode dataset #############
    #make_linemode_dataset(dataset_main_path+'/NDDS_10k', dataset_main_path, model_number, length_multipler)
    ############ generate the linemode dataset #############
    generated_dataset_path = dataset_main_path+'/generated_dataset'
    #vis_bb(generated_dataset_path, frame_number_to_visulize,model_number,dataset_main_path)
    model_info.export_model_info('obj_01.yml','datasets/Linemod_preprocessed/models/obj_08.ply',1)
