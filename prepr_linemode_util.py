import os
import shutil
from PIL import Image


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