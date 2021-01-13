import lib.utils.ply_utils as ply

def read_ply_info(ply_path):
    model_3d_points = ply.load_model_ply(ply_path)

    model_corner = ply.get_model_corners(model_3d_points)
    models_distances = ply.get_model_distances(model_corner)
    models_diameter = ply.get_model_diamater(model_corner)

    print(models_diameter)
    return model_corner, models_distances,models_diameter