import tqdm
import os
from PIL import Image
import numpy as np
import json

from lib.utils import base_utils
from lib.csrc.fps import fps_utils
from lib.utils.linemod.opengl_renderer import OpenGLRenderer


def read_ply_points(ply_path):
    ply = PlyData.read(ply_path)
    data = ply.elements[0].data
    points = np.stack([data['x'], data['y'], data['z']], axis=1)
    return points


def sample_fps_points(data_root):
    ply_path = os.path.join(data_root, 'model.ply')
    ply_points = read_ply_points(ply_path)
    fps_points = fps_utils.farthest_point_sampling(ply_points, 8, True)
    np.savetxt(os.path.join(data_root, 'fps.txt'), fps_points)


def get_model_corners(model):
    min_x, max_x = np.min(model[:, 0]), np.max(model[:, 0])
    min_y, max_y = np.min(model[:, 1]), np.max(model[:, 1])
    min_z, max_z = np.min(model[:, 2]), np.max(model[:, 2])
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

def record_ann(model_meta, img_id, ann_id, images, annotations, cls_type):
    data_root = model_meta['data_root']
    corner_3d = model_meta['corner_3d']
    center_3d = model_meta['center_3d']
    fps_3d = model_meta['fps_3d']
    K = model_meta['K']

    length = int(input("How many pictures do you want to use? "))
    inds = range(length)

    for ind in tqdm.tqdm(inds):
        desired_class = cls_type  # name of the objectclass, on which you want to train

        number = str(ind)
        number = number.zfill(6)

        # getting rgb
        datei = number + '.png'
        rgb_path = os.path.join(data_root, datei)

        rgb = Image.open(rgb_path).convert('RGB')
        img_size = rgb.size
        img_id += 1
        info = {'file_name': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': img_id}
        images.append(info)

        # path to annotations from ndds
        datei = number + '.json'
        pose_path = os.path.join(data_root, datei)

        # getting pose of annotations from ndds

        with open(pose_path, 'r') as file:
            annotation = json.loads(file.read())

        object_from_annotation = annotation['objects']
        object_class = object_from_annotation[0]["class"]

        if desired_class in object_class:

            # translation
            translation = np.array(object_from_annotation[0]['location']) * 10

            # rotation
            rotation = np.asarray(object_from_annotation[0]['pose_transform'])[0:3, 0:3]
            rotation = np.dot(rotation, np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]))
            rotation = np.dot(rotation.T, np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))

            # pose
            pose = np.column_stack((rotation, translation))
        else:
            print("No such class in annotations!")
            pass

        corner_2d = base_utils.project(corner_3d, K, pose)
        center_2d = base_utils.project(center_3d[None], K, pose)[0]

        fps_2d = base_utils.project(fps_3d, K, pose)

        # getting segmentation-mask
        datei = number + '.cs.png'
        mask_path = os.path.join(data_root, datei)

        ann_id += 1
        anno = {'mask_path': mask_path, 'image_id': img_id, 'category_id': 1, 'id': ann_id}
        anno.update({'corner_3d': corner_3d.tolist(), 'corner_2d': corner_2d.tolist()})
        anno.update({'center_3d': center_3d.tolist(), 'center_2d': center_2d.tolist()})
        anno.update({'fps_3d': fps_3d.tolist(), 'fps_2d': fps_2d.tolist()})
        anno.update({'K': K.tolist(), 'pose': pose.tolist()})

        anno.update({'data_root': data_root})

        anno.update({'type': 'real', 'cls': cls_type})
        annotations.append(anno)

    return img_id, ann_id


def custom_to_coco(data_root):
    model_path = os.path.join(data_root, 'model.ply')

    cls_type = input("On which class do you want to train? ")

    renderer = OpenGLRenderer(model_path)
    K = np.loadtxt(os.path.join(data_root, 'camera.txt'))

    model = renderer.model['pts'] / 1000
    corner_3d = get_model_corners(model)
    print("corner_3d:")
    print(corner_3d)
    center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
    print("center_3d:")
    print(center_3d)
    fps_3d = np.loadtxt(os.path.join(data_root, 'fps.txt'))

    model_meta = {
        'K': K,
        'corner_3d': corner_3d,
        'center_3d': center_3d,
        'fps_3d': fps_3d,
        'data_root': data_root,
    }

    img_id = 0
    ann_id = 0
    images = []
    annotations = []

    img_id, ann_id = record_ann(model_meta, img_id, ann_id, images, annotations, cls_type)

    categories = [{'supercategory': 'none', 'id': 1, 'name': cls_type}]
    instance = {'images': images, 'annotations': annotations, 'categories': categories}

    anno_path = os.path.join(data_root, 'train.json')
    with open(anno_path, 'w') as f:
        json.dump(instance, f)