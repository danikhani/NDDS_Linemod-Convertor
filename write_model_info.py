import lib.utils.ply_utils as ply


def export_model_info(yaml_path, ply_path, model_number):
    model_corner, model_distances, model_diameter = ply.read_ply_info(ply_path)
    ply.export_model_para_yml(yaml_path, model_number, model_corner, model_distances,model_diameter)