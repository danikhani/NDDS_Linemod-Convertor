import prepr_linemode_util as util
import test_lib as tlib
import os
import shutil
from PIL import Image
import yaml
import io
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2


def get_bbox(filename):
    ply_modell_np = tlib.load_model_ply('datasets/obj_16.ply')
    ply_modell_np = ply_modell_np[:5,:]
    cam_K_np = np.reshape(np.array([640.0, 0, 320.0, 0, 480.0, 240.0, 0, 0, 1.0]), newshape=(3, 3))
    with open(os.path.join('datasets/try1', filename)) as jsonfile:
        object_data = json.load(jsonfile)

        bbox = object_data['objects'][0]['bounding_box']
        projected_3d = np.array(object_data['objects'][0]['projected_cuboid'])

        bbox_x = round(bbox['top_left'][0])
        bbox_y = round(bbox['top_left'][1])
        bbox_w = round(bbox['bottom_right'][0]) - bbox_x
        bbox_h = round(bbox['bottom_right'][1]) - bbox_y

        yaml_out = {}

        translation = np.array(object_data['objects'][0]['location']) * 10  # NDDS gives units in centimeters

        quaternion_obj2cam = R.from_quat(np.array(object_data['objects'][0]['quaternion_xyzw']))
        quaternion_cam2world = R.from_quat(np.array(object_data['camera_data']['quaternion_xyzw_worldframe']))
        quaternion_obj2world = quaternion_obj2cam * quaternion_cam2world

        mirrored_y_axis = np.dot(quaternion_obj2world.as_dcm(), np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))


        # get the corners of bounding boxes from annotations
        bbox = object_data['objects'][0]['bounding_box']
        rmax, cmax = np.asarray(bbox['top_left'])
        rmin, cmin = np.asarray(bbox['bottom_right'])
        rmax, rmin, cmax, cmin = int(rmin), int(rmax), int(cmin), int(cmax)

        #get pointsclouds
        #imgpts, jac = cv2.projectPoints(self.cld[idx] * 1e3, cam_rotation4, cam_translation * 1e3, cam_mat, dist)
        #cv2_img = cv2.polylines(np.array(cv2_img), np.int32([np.squeeze(imgpts)]), True, (0, 255, 255))
        '''points_2D, jacobian =tlib.project_points_3D_to_2D( points_3D=ply_modell_np,  # transform the object origin point which is the centerpoint
                                    rotation_vector=tlib.transform_rotation(np.array(mirrored_y_axis), "axis_angle"),
                                    translation_vector=np.array(translation),
                                    camera_matrix=cam_K_np)
        points_2D = np.squeeze(points_2D)'''
        points_2D = 1

        #print(bbox)
        #print(translation)
        #print(mirrored_y_axis)
        print(points_2D)
        return rmin, rmax, cmin, cmax, points_2D,projected_3d


def vis_bb(image_name,name,savefolder):
    #cam_R_m2c_array, cam_t_m2c_array, obj_bb = util.get_groundtruth_data('datasets/try1', name)
    #print(cam_R_m2c_array, cam_t_m2c_array, obj_bb)

    rmin, rmax, cmin, cmax, points_2D,projected_3d = get_bbox(name)
    img = Image.open(os.path.join('datasets/try1', image_name))
    img_bbox = np.array(img.copy())
    img_name = savefolder + '/1.bbox.png'
    cv2.rectangle(img_bbox, (cmin, rmin), (cmax, rmax), (255, 0, 0), 2)
    cv2.imwrite(img_name, img_bbox)



    img_3d = np.array(img.copy())
    img_3d_name = savefolder + '/1.3d.png'
    tlib.draw_bbox_8_2D(img_3d,projected_3d)
    #cv2_img = cv2.polylines(img_3d, points_2D, True, (0, 255, 255))
    cv2.imwrite(img_3d_name, img_3d)


vis_bb('000012.png','000012.json','datasets/testfolder')