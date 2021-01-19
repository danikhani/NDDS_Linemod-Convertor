import numpy as np
import io
import yaml
from plyfile import PlyData


def load_model_ply(path_to_ply_file):
    """
   Loads a 3D model from a plyfile
    Args:
        path_to_ply_file: Path to the ply file containing the object's 3D model
    Returns:
        points_3d: numpy array with shape (num_3D_points, 3) containing the x-, y- and z-coordinates of all 3D model points

    """
    model_data = PlyData.read(path_to_ply_file)

    vertex = model_data['vertex']
    points_3d = np.stack([vertex[:]['x'], vertex[:]['y'], vertex[:]['z']], axis=-1)

    return points_3d

# return the corners of the model
def get_model_corners(points_3d):
    min_x, max_x = np.min(points_3d[:, 0]), np.max(points_3d[:, 0])
    min_y, max_y = np.min(points_3d[:, 1]), np.max(points_3d[:, 1])
    min_z, max_z = np.min(points_3d[:, 2]), np.max(points_3d[:, 2])
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

# load the complete 3d model from the ply file
def get_model_distances(corners_3d):
    # xyz distances
    distances = np.max(corners_3d, 0) - np.min(corners_3d, 0)
    return distances

def get_model_centers(corners_3d):
    # xyz center
    center_3d = (np.max(corners_3d, 0) + np.min(corners_3d, 0)) / 2
    return center_3d

def get_model_diamater(corners_3d):
    # more on diamater:  https://github.com/thodan/sixd_toolkit/blob/96bb268e1fb5ebd82ca1b8d352e3263561ba6f5c/pysixd/misc.py
    diameter = np.linalg.norm(np.max(corners_3d, 0) - np.min(corners_3d, 0))
    return diameter


def read_ply_info(ply_path):
    model_3d_points = load_model_ply(ply_path)

    model_corner = get_model_corners(model_3d_points)
    model_distances = get_model_distances(model_corner)
    model_diameter = get_model_diamater(model_corner)

    return model_corner, model_distances ,model_diameter


def export_model_para_yml(file_path, model_number, model_corners, model_distances,model_diameter):
    model_paras = {
        model_number: {
            'diameter': round(model_diameter.tolist(),4),
            'min_x': round(model_corners[0][0].tolist(),4),
            'min_y': round(model_corners[0][1].tolist(),4),
            'min_z': round(model_corners[0][2].tolist(),4),
            'size_x': round(model_distances[0].tolist(),4),
            'size_y': round(model_distances[1].tolist(),4),
            'size_z': round(model_distances[2].tolist(),4),
        }
    }

    with io.open(file_path, 'w', encoding='utf8') as outfile:
        yaml.dump(model_paras, outfile, default_flow_style=None,width=1000)