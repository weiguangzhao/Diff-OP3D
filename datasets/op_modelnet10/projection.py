import os, sys, glob
import numpy as np
import skimage.feature
import torch
sys.path.append('..')
from dataset_utils import rotate_point_cloud, get_npy_from_off, determine_best_view, normalize_point, make_front_faces,\
                          get_data_from_off, init_render, render_object, get_cycle_position

import matplotlib.pyplot as plt
import torch.nn.functional as F

# ####import pytorch3d
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (look_at_view_transform, PerspectiveCameras, RasterizationSettings, MeshRasterizer,
                                TexturesVertex, PointLights, MeshRenderer, HardPhongShader, Materials, HardFlatShader)
from pytorch3d.renderer.mesh.shader import (BlendParams)


# ####projection parameter setting
r_p = 1.6
phi_1=30
camera_dict_six = {'up': [[0, 0, r_p], [0.0, 1.0, 0.0]],
                   'down': [[0, 0, -r_p], [0.0, 1.0, 0.0]],
                   'front': [[r_p, 0, 0], [0.0, 0.0, 1.0]],
                   'back': [[-r_p, 0, 0], [0.0, 0.0, 1.0]],
                   'right': [[0, r_p, 0], [0.0, 0.0, 1.0]],
                   'left': [[0, -r_p, 0], [0.0, 0.0, 1.0]]}
camera_dict_cycle = get_cycle_position(r_p, phi_1)


def get_render_data(test_file_list, save_npy_dir, save_pic_path):
    for f_i, f_path in enumerate(test_file_list):
        file_name = f_path.split('/')[-1].split('.')[0]
        # ####get base information for pointcloud
        xyz_raw = np.load(save_npy_dir + file_name + '_xyz.npy')
        faces = np.load(save_npy_dir + file_name + '_faces.npy')
        verts = torch.from_numpy(xyz_raw).type(torch.float32)
        faces = torch.from_numpy(faces).type(torch.int64)
        verts_rgb = torch.ones_like(verts)[None] * torch.tensor([1.0, 1.0, 1.0], device=verts.device)
        textures = TexturesVertex(verts_features=verts_rgb)

        # ####trans to pytorch3d input
        meshes = Meshes(verts=[verts], faces=[faces], textures=textures).cuda()
        meshes._compute_face_areas_normals()
        meshes = make_front_faces(meshes)

        for cur_td in camera_dict_cycle:
            camera_position = camera_dict_cycle[cur_td][0]
            look_up = camera_dict_cycle[cur_td][1]
            camera_position = torch.tensor(camera_position).type(torch.float32)
            look_up = torch.tensor(look_up).type(torch.float32)
            rasterizer, renderer = init_render(camera_position, look_up)
            render_object(file_name, rasterizer, renderer, meshes, save_npy_dir, save_pic_path, toward=cur_td)
        print('complete {}/{}: {}'.format(f_i, len(test_file_list), file_name))


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "7"
    test_file_list = glob.glob('./ModelNet10/*/test/*.off')
    test_file_list.sort()
    save_npy_dir = './npy_render/'
    save_pic_path = './pic_render/'
    os.makedirs(save_pic_path, exist_ok=True)
    os.makedirs(save_npy_dir, exist_ok=True)

    # ####generate data for only first time
    get_data_from_off(test_file_list, save_npy_dir)


    # ####generate the render img
    get_render_data(test_file_list, save_npy_dir, save_pic_path)

