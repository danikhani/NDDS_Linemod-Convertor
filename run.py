import argparse
import os
import sys

from prepr_linemode import make_linemode_dataset
from visualize_dataset import vis_bb
import write_model_info as model_info



def parse_args(args):
    """
    Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Linemode dataset generator')
    subparsers = parser.add_subparsers(help='Arguments for specific conversion process.', dest='conversion')
    subparsers.required = True

    # every model should have a uniq number
    parser.add_argument('--model-number', help='Number of Model', type=int, required=True)

    # args for the image viewer
    view_annotation = subparsers.add_parser('view')
    view_annotation.add_argument('--dataset-path', help='Path to images to dataset that needs to be visulized (ie. datasets/test_dataset/generated_dataset)')
    view_annotation.add_argument('--frame-number', help='number of picture to show', default=1, type=int)
    view_annotation.add_argument('--save-path', help='Path to store visualized images', default=None)

    # args for the ply reader
    ply_reader = subparsers.add_parser('ply')
    ply_reader.add_argument('--ply-path', help='path to .ply file (ie. datasets/test_dataset/generated_dataset/models/obj_11.ply).')
    ply_reader.add_argument('--save-path', help='Path to save the dataset (ie. datasets/test_dataset/generated_dataset/models)',required=True)

    #args for ndds convertor
    NDDS_convertor = subparsers.add_parser('ndds')
    NDDS_convertor.add_argument('--ndds-path',
                                  help='Path to data from NDDS (ie. datasets/test_dataset/NDDS_generated).')
    NDDS_convertor.add_argument('--save-path',help='Path to save the dataset',default='datasets/test_dataset/generated_dataset')
    # since the files are from cad and each one of them has a different length unit.
    # use this length factor if the visulized box is too small or too big!
    NDDS_convertor.add_argument('--scaler',
                        help='Use one of these values to scale the box: (0.01,0.1,1,10,100). The smaller value makes a bigger box',
                        type=float, default=1)

    NDDS_convertor.add_argument('--train-percentage',
                                help='Percentage amount of data used for training',
                                type=float, default=0.9)

    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)
    # parser


def main(args = None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    # use as a dictionary
    args = vars(args)

    model_number = args['model_number']
    if args['conversion'] == 'view':
        frame_number_to_visulize =  args['frame_number']
        dataset_path = args['dataset_path']
        pic_with_boundingbox_path = args['save_path']
        vis_bb(dataset_path, frame_number_to_visulize, model_number, pic_with_boundingbox_path)

    if args['conversion'] == 'ply':
        yml_save_path = args['save_path']
        ply_path = args['ply_path']
        yml_file = os.path.join(yml_save_path, 'cubiod_obj_{}.yml'.format(model_number))
        model_info.export_model_info(yml_file, ply_path, model_number)

    if args['conversion'] == 'ndds':
        raw_ndds_data = args['ndds_path']
        export_dataset_path = args['save_path']
        scaler = args['scaler']
        train_percentage = args['train_percentage']
        make_linemode_dataset(raw_ndds_data, export_dataset_path, model_number, scaler,train_percentage)


if __name__ == '__main__':
    main()