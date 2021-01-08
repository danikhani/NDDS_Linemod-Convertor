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
import matplotlib.pyplot as plt


def get_bbox(filename,source_folder):
    #ply_modell_np = tlib.load_model_ply('datasets/obj_16.ply')
    #ply_modell_np = ply_modell_np[:5,:]
    #cam_K_np = np.reshape(np.array([640.0, 0, 320.0, 0, 480.0, 240.0, 0, 0, 1.0]), newshape=(3, 3))
    with open(os.path.join(source_folder, '_camera_settings.json')) as jsonfile:
        camera_data = json.load(jsonfile)

    with open(os.path.join(source_folder, filename)) as jsonfile:
        object_data = json.load(jsonfile)

        bbox = object_data['objects'][0]['bounding_box']
        projected_3d = np.array(object_data['objects'][0]['projected_cuboid'])

        bbox_x = round(bbox['top_left'][0])
        bbox_y = round(bbox['top_left'][1])
        bbox_w = round(bbox['bottom_right'][0]) - bbox_x
        bbox_h = round(bbox['bottom_right'][1]) - bbox_y
        intrinsic_settings = camera_data['camera_settings'][0]['intrinsic_settings']
        fx = float(intrinsic_settings['fx'])
        ux = float(intrinsic_settings['cx'])
        fy = float(intrinsic_settings['fy'])
        uy = float(intrinsic_settings['cy'])
        s = float(intrinsic_settings['s'])
        cam_K = [fx, 0, ux, 0, fy, uy, 0, 0, 1.0]
        cam_K_np = np.reshape(np.array(cam_K), newshape=(3, 3))
        yaml_out = {}

        translation = np.array(object_data['objects'][0]['location'])  # NDDS gives units in centimeters

        quaternion_obj2cam = R.from_quat(np.array(object_data['objects'][0]['quaternion_xyzw']))
        quaternion_cam2world = R.from_quat(np.array(object_data['camera_data']['quaternion_xyzw_worldframe']))
        quaternion_obj2world = quaternion_obj2cam * quaternion_cam2world
        mirrored_y_axis = quaternion_obj2world.as_dcm()
        r1 = R.from_euler('x', 90, degrees=True)
        r1 = r1.as_dcm()
        mirrored_y_axis = np.dot(quaternion_obj2world.as_dcm(), r1)
        #mirrored_y_axis = np.dot(quaternion_obj2world.as_dcm(), np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))
        #mirrored_y_axis = np.dot(quaternion_obj2world.as_dcm(), np.array([[0, 0, 1], [0, 1, 0], [1,0, 0]]))
        #mirrored_y_axis = np.dot(mirrored_y_axis, np.array([[0, 0, 1], [1, 0, 0], [1, 0, 0]]))
        #mirrored_y_axis = quaternion_obj2world.as_dcm()

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

        '''experimental'''
        cuboid_3d = np.array(object_data['objects'][0]['cuboid'])
        cuboid_3d = np.reshape(cuboid_3d, newshape=(8, 3))
        cuboid_2d = np.array(object_data['objects'][0]['projected_cuboid'])
        cuboid_2d = np.reshape(cuboid_2d, newshape=(8, 2))
        rt, tt = util.pnp(cuboid_3d, cuboid_2d, cam_K_np)

        '''experimental2'''
        t2 = np.array(object_data['objects'][0]['location'])
        t2 = np.round(t2, 8)

        r2 = np.asarray(object_data['objects'][0]['pose_transform'])[0:3, 0:3]
        r2 = np.dot(r2, np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]))
        r2 = np.dot(r2.T, np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))
        r2 = np.dot(r2, np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
        r2 = np.dot(r2.T, np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
        #r2 = np.dot(r2.T, np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        #r2 = np.dot(r2, np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]))
        # make list from rows and add them to a single array
        #r2 = np.round(r2, 8)
        #rotation_list = list(rotation[0, :]) + list(rotation[1, :]) + list(rotation[2, :])

        print(bbox)
        print(translation)
        print(mirrored_y_axis)
        print(cam_K_np)
        print(points_2D)
        return rmin, rmax, cmin, cmax, points_2D,projected_3d,mirrored_y_axis,translation,cam_K_np,rt, tt,r2,t2


def vis_bb(image_name,name,savefolder,source_folder,model_number):
    #cam_R_m2c_array, cam_t_m2c_array, obj_bb = util.get_groundtruth_data('datasets/try1', name)
    #print(cam_R_m2c_array, cam_t_m2c_array, obj_bb)

    rmin, rmax, cmin, cmax, points_2D,projected_3d,cam_R_m2c,cam_t_m2c,cam_K,cam_R_m2c2, cam_t_m2c2,cam_R_m2c3, cam_t_m2c3= get_bbox(name,source_folder)
    img = Image.open(os.path.join(source_folder, image_name))
    img_bbox = np.array(img.copy())
    img_name = savefolder + '/1.bbox.png'
    cv2.rectangle(img_bbox, (cmin, rmin), (cmax, rmax), (255, 0, 0), 2)

    cv2.imwrite(img_name, img_bbox)
    img2d = cv2.imread(img_name, 0)
    plt.imshow(img2d)
    plt.show()

    '''3d projection'''
    with open('datasets/models_info.yml') as file:
        model_info = yaml.load(file, Loader=yaml.FullLoader)

    current_model = model_info[model_number]
    # calculate 3dpoints
    img_3d = np.array(img.copy())
    bb3d_points = tlib.get_bbox_3d(current_model)
    projected_points = tlib.project_bbox_3D_to_2D(bb3d_points, cam_R_m2c, cam_t_m2c, cam_K,
                                                  append_centerpoint=False)
    tlib.draw_bbox_8_2D(img_3d, projected_points)
    tlib.draw_bbox_8_2D(img_3d, projected_3d)
    img_3d_name = savefolder + '/1.3d.png'


    #cv2_img = cv2.polylines(img_3d, points_2D, True, (0, 255, 255))
    cv2.imwrite(img_3d_name, img_3d)
    img3dd = cv2.imread(img_3d_name, 0)
    plt.imshow(img3dd)
    plt.show()


    '''other loading
    # load rgb image as base.
    loaded_rgb = Image.open(rgb_image_path)

    # draw 2d bounding box
    rgb_2d_bbox = np.array(loaded_rgb.copy())
    cv2.rectangle(rgb_2d_bbox, (obj_bb[0], obj_bb[1]), (obj_bb[0] + obj_bb[2], obj_bb[1] + obj_bb[3]), (255, 0, 0), 2)
    #cv2.imwrite(img_path_2d, rgb_2d_bbox)
    # draw 3d bounding box
    rgb_3d_bbox = np.array(loaded_rgb.copy())
    projected_points = tlib.project_bbox_3D_to_2D(b3d_points, cam_R_m2c, cam_t_m2c, cam_K,
                                                  append_centerpoint=False)
    tlib.draw_bbox_8_2D(rgb_3d_bbox, projected_points)
    #cv2.imwrite(img_path_3d, rgb_3d_bbox)

    img = cv2.imread(img_path_3d, 0)

    plt.imshow(img)
    plt.show()
'''

vis_bb('000012.png','000012.json','datasets/testfolder','datasets/diff_focal_length',17)

