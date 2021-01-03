import os
import shutil
from PIL import Image
from natsort import natsorted

import prepr_linemode_util as util

raw_data_directory = 'for_mesh_generation/raw_data'
corrected_ending = '.png'


#structure of data:
main_data_path = 'for_mesh_generation'

mask_folder = 'mask'
depth_folder = 'depth'
jpeg_rgb_folder = 'JPEGImages'
png_rgb_folder = 'PNGImages'

#what kind of files want to get extraceted
depth_ending = 'depth.mm.16.png'
mask_ending = '.cs.png'
rgb_ending = '.png'
all_endings = ['.cs.png','.16.png','.8.png','.micon.png','.depth.png','.is.png']

# create folder structure
try:
    os.makedirs(main_data_path)
except FileExistsError:
    print(main_data_path + ' exist')

# create folders
depth_path = util.make_empty_folder(main_data_path,depth_folder)
mask_path = util.make_empty_folder(main_data_path,mask_folder)
png_rgb_path = util.make_empty_folder(main_data_path,png_rgb_folder)
jpeg_rgb_path = util.make_empty_folder(main_data_path,jpeg_rgb_folder)

mask_index = 0
depth_index = 0
rgb_index = 0
sorted_list_of_files = natsorted(os.listdir(raw_data_directory))
for picture_name in sorted_list_of_files:
    # find mask data
    if picture_name.endswith(mask_ending):
        util.copy_and_rename(raw_data_directory, picture_name, mask_index, mask_path)
        mask_index += 1
        print(picture_name)
    # find depth data
    if picture_name.endswith(depth_ending):
        util.copy_and_rename(raw_data_directory,picture_name,depth_index,depth_path)
        depth_index += 1
        print(picture_name)
    # find depth data
    if picture_name.endswith(rgb_ending) and not any(picture_name.endswith(x) for x in all_endings):
        util.copy_and_rename(raw_data_directory, picture_name, rgb_index, png_rgb_path)
        rgb_index += 1
        print(picture_name)

sorted_jpeg_images = natsorted(os.listdir(png_rgb_path))
for images in sorted_jpeg_images:
    im = Image.open(png_rgb_path +'/'+ images)
    jpg_name = images.replace('.png', '')
    im.convert('RGB').save("{}.jpg".format(jpeg_rgb_path+'/'+jpg_name), "JPEG")  # this converts png image as jpeg
    print(jpg_name)
