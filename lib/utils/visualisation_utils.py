import numpy as np
import cv2


def transform_rotation(rotation_matrix, rotation_representation):
    """
    Transforms the input rotation matrix into the given rotation representation. Currently only axis_angle is supported.
    Arguments:
        rotation_matrix: numpy array with shape (3, 3) containing the rotation
        rotation_representation: String with the rotation representation. Currently only 'axis_angle' is supported
    Returns:
        numpy array containing the rotation in the given representation
    """
    # possible rotation representations: "axis_angle", "rotation_matrix", "quaternion"
    if rotation_representation == "rotation_matrix":
        return rotation_matrix
    elif rotation_representation == "axis_angle":
        reshaped_rot_mat = np.reshape(rotation_matrix, newshape = (3, 3))
        axis_angle, jacobian = cv2.Rodrigues(reshaped_rot_mat)
        return np.squeeze(axis_angle)


def project_points_3D_to_2D(points_3D, rotation_vector, translation_vector, camera_matrix):
    """
    Transforms and projects the input 3D points onto the 2D image plane using the given rotation, translation and camera matrix
    Arguments:
        points_3D: numpy array with shape (num_points, 3) containing 3D points (x, y, z)
        rotation_vector: numpy array containing the rotation vector with shape (3,)
        translation_vector: numpy array containing the translation vector with shape (3,)
        camera_matrix: numpy array with shape (3, 3) containing the intrinsic camera parameters
    Returns:
        points_2D: numpy array with shape (num_points, 2) with the 2D projections of the given 3D points
    """
    points_2D, jacobian = cv2.projectPoints(points_3D, rotation_vector, translation_vector, camera_matrix, None)
    points_2D = np.squeeze(points_2D)

    return points_2D



def get_bbox_3d(model_dict):
    """
    Converts the 3D model cuboid from the Linemod format (min_x, min_y, min_z, size_x, size_y, size_z) to the (num_corners = 8, num_coordinates = 3) format
    Args:
        model_dict: Dictionary containing the cuboid information of a single Linemod 3D model in the Linemod format
    Returns:
        bbox: numpy (8, 3) array containing the 3D model's cuboid, where the first dimension represents the corner points and the second dimension contains the x-, y- and z-coordinates.

    """
    # get infos from model dict
    min_point_x = model_dict["min_x"]
    min_point_y = model_dict["min_y"]
    min_point_z = model_dict["min_z"]

    size_x = model_dict["size_x"]
    size_y = model_dict["size_y"]
    size_z = model_dict["size_z"]

    bbox = np.zeros(shape=(8, 3))
    # untere ebende
    bbox[0, :] = np.array([min_point_x, min_point_y, min_point_z])
    bbox[1, :] = np.array([min_point_x + size_x, min_point_y, min_point_z])
    bbox[2, :] = np.array([min_point_x + size_x, min_point_y + size_y, min_point_z])
    bbox[3, :] = np.array([min_point_x, min_point_y + size_y, min_point_z])
    # obere ebene
    bbox[4, :] = np.array([min_point_x, min_point_y, min_point_z + size_z])
    bbox[5, :] = np.array([min_point_x + size_x, min_point_y, min_point_z + size_z])
    bbox[6, :] = np.array([min_point_x + size_x, min_point_y + size_y, min_point_z + size_z])
    bbox[7, :] = np.array([min_point_x, min_point_y + size_y, min_point_z + size_z])

    return bbox


def project_bbox_3D_to_2D(points_bbox_3D, rotation_vector, translation_vector, camera_matrix, append_centerpoint=True):
    """ Projects the 3D model's cuboid onto a 2D image plane with the given rotation, translation and camera matrix.

    Arguments:
        points_bbox_3D: numpy array with shape (8, 3) containing the 8 (x, y, z) corner points of the object's 3D model cuboid
        rotation_vector: numpy array containing the rotation vector with shape (3,)
        translation_vector: numpy array containing the translation vector with shape (3,)
        camera_matrix: numpy array with shape (3, 3) containing the intrinsic camera parameters
        append_centerpoint: Boolean indicating wheter to append the centerpoint or not
    Returns:
        points_bbox_2D: numpy array with shape (8 or 9, 2) with the 2D projections of the object's 3D cuboid
    """
    if append_centerpoint:
        points_bbox_3D = np.concatenate([points_bbox_3D, np.zeros(shape=(1, 3))], axis=0)
    points_bbox_2D, jacobian = cv2.projectPoints(points_bbox_3D, rotation_vector, translation_vector, camera_matrix,
                                                 None)
    points_bbox_2D = np.squeeze(points_bbox_2D)

    return points_bbox_2D


def draw_bbox_8_2D(draw_img, bbox_8_2D, color=(0, 255, 0), thickness=2):
    """ Draws the 2D projection of a 3D model's cuboid on an image with a given color.

    # Arguments
        draw_img     : The image to draw on.
        bbox_8_2D    : A [8 or 9, 2] matrix containing the 8 corner points (x, y) and maybe also the centerpoint.
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    # convert bbox to int and tuple
    bbox = np.copy(bbox_8_2D).astype(np.int32)
    bbox = tuple(map(tuple, bbox))

    # lower level
    cv2.line(draw_img, bbox[0], bbox[1], color, thickness)
    cv2.line(draw_img, bbox[1], bbox[2], color, thickness)
    cv2.line(draw_img, bbox[2], bbox[3], color, thickness)
    cv2.line(draw_img, bbox[0], bbox[3], color, thickness)
    # upper level
    cv2.line(draw_img, bbox[4], bbox[5], color, thickness)
    cv2.line(draw_img, bbox[5], bbox[6], color, thickness)
    cv2.line(draw_img, bbox[6], bbox[7], color, thickness)
    cv2.line(draw_img, bbox[4], bbox[7], color, thickness)
    # sides
    cv2.line(draw_img, bbox[0], bbox[4], color, thickness)
    cv2.line(draw_img, bbox[1], bbox[5], color, thickness)
    cv2.line(draw_img, bbox[2], bbox[6], color, thickness)
    cv2.line(draw_img, bbox[3], bbox[7], color, thickness)

    # check if centerpoint is also available to draw
    if len(bbox) == 9:
        # draw centerpoint
        cv2.circle(draw_img, bbox[8], 3, color, -1)