import os
import math
import torch
import numpy as np
import skimage.feature
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ####import pytorch3d
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (look_at_view_transform, PerspectiveCameras, RasterizationSettings, MeshRasterizer,
                                TexturesVertex, PointLights, MeshRenderer, HardPhongShader, Materials, HardFlatShader)
from pytorch3d.renderer.mesh.shader import (BlendParams)


def calculate_target_coordinates(r, phi_1, phi_2):
    phi_1 = np.deg2rad(phi_1)
    phi_2 = np.deg2rad(phi_2)
    x = r * math.sin(phi_1)*math.cos(phi_2)
    y = r * math.sin(phi_1)*math.sin(phi_2)
    z = r * math.cos(phi_1)
    return x, y, z


def get_cycle_position(r, phi_1,):
    camera_dict_cycle = {}
    for c_i in range(24):
        angle = (360./24.0)*c_i
        if angle%10==0: continue
        x, y, z = calculate_target_coordinates(r, phi_1, angle)
        cur_position = [x, y, z]
        camera_dict_cycle['R{}'.format(int(angle))] = [cur_position, [0.0, 0.0, 1.0]]
    return camera_dict_cycle


# ####rotate point cloud
def rotate_point_cloud(point_cloud, angle_x, angle_y, angle_z):
    # change angle to radius
    angle_x = np.deg2rad(angle_x)
    angle_y = np.deg2rad(angle_y)
    angle_z = np.deg2rad(angle_z)

    # rotation matrix
    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, np.cos(angle_x), -np.sin(angle_x)],
                                  [0, np.sin(angle_x), np.cos(angle_x)]])
    rotation_matrix_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                                  [0, 1, 0],
                                  [-np.sin(angle_y), 0, np.cos(angle_y)]])
    rotation_matrix_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                                  [np.sin(angle_z), np.cos(angle_z), 0],
                                  [0, 0, 1]])

    # transpose
    point_cloud_T = point_cloud.T

    # rotate point cloud
    point_cloud_T = rotation_matrix_x.dot(point_cloud_T)
    point_cloud_T = rotation_matrix_y.dot(point_cloud_T)
    point_cloud_T = rotation_matrix_z.dot(point_cloud_T)

    # transpose
    rotated_point_cloud = point_cloud_T.T

    return rotated_point_cloud


# ###make all face towards consistency
def make_front_faces(mesh_pytorch3d):

    # compute the normals for the mesh
    verts = mesh_pytorch3d.verts_packed()
    faces = mesh_pytorch3d.faces_packed().float()
    textures = mesh_pytorch3d.textures
    face_normals = mesh_pytorch3d._faces_normals_packed

    # convert the normals to face normals
    # face_normals = normals.view(num_faces, 3, 3).mean(dim=1)

    # get the dot product between the normals and the faces
    dot_product = (face_normals * faces.norm(dim=1, keepdim=True)).sum(dim=1)

    # create a mask for the front-facing and back-facing triangles
    back_facing_mask = dot_product < 0

    # reverse the order of vertices for the back-facing triangles
    faces[back_facing_mask] = torch.flip(faces[back_facing_mask], [1])
    faces = faces.long()

    # update the mesh with the new faces
    mesh_pytorch3d = Meshes(verts=[verts], faces=[faces], textures=textures)

    # return the mesh with front-facing triangles
    return mesh_pytorch3d


# ### PCA view
def determine_best_view(xyz):
    pca = PCA(n_components=3)
    pca.fit(xyz)
    # ##pca transform
    # rotated_vertices = pca.transform(xyz)
    # ##maximum direction [1, 0, 0]
    # rotated_vertices= normalize_point(rotated_vertices)
    views = pca.components_[-1]

    # test_pca = PCA(n_components=3)
    # test_pca.fit(rotated_vertices)

    return views

def xyz2ball(view):
    x, y, z= view[0], view[1], view[2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi_1 = np.arccos(z / r)
    phi_2 = np.arctan2(y, x)

    phi_1 = np.degrees(phi_1)
    phi_2 = np.degrees(phi_2)
    phi= torch.tensor([phi_1, phi_2])
    return phi


# ####nomoralize the point cloud
def normalize_point(points):
    # ####normalize point xyz to [0, 1]
    center = np.mean([np.min(points, axis=0), np.max(points, axis=0)], axis=0)
    dists = np.sqrt(np.sum((points - center) ** 2, axis=1))
    max_dist = np.max(dists)
    points /= max_dist

    # ####move the center points to the origin
    center = np.mean([np.min(points, axis=0), np.max(points, axis=0)], axis=0)
    points = points - center
    return points


def get_npy_from_off(off_path, save_path):
    with open(off_path, 'r') as f:
        f.readline()  # skip the file head
        num_line =  f.readline()
        num_vert, num_face, _ = map(int, num_line.split())

        points = []
        faces = []
        # ####read location
        for l_i in range(num_vert):
            cur_line = f.readline()
            x, y, z = map(float, cur_line.split())
            points.append([x, y, z])

        # ####read faces
        for l_i in range(num_face):
            cur_line = f.readline()
            _, face_0, face_1, face_2 = map(int, cur_line.split())
            faces.append([face_0, face_1, face_2])

        points = np.array(points)
        faces = np.array(faces)
        points = normalize_point(points)
        f.close()
    np.save(save_path+'_xyz.npy', points)
    np.save(save_path + '_faces.npy', faces)
    pass


# #### get data from off file for ModelNet10
def get_data_from_off(test_file_list, save_dir):
    for f_i, f_path in enumerate(test_file_list):
        file_name = f_path.split('/')[-1].split('.')[0]
        save_path = save_dir + file_name
        get_npy_from_off(f_path, save_path)
        print('complete {}/{}: {}'.format(f_i, len(test_file_list), file_name))
    pass


def init_render(camera_position, look_up):
    lights = PointLights(location=torch.tensor([[2.0, 2.0, 2.0]])).cuda()

    # creat PerspectiveCameras
    half_half = (512.0 / 2.0, 512.0 / 2.0)

    R, T = look_at_view_transform(eye=camera_position.view(1, 3), at=torch.tensor([[0.0, 0.0, 0.0]]),
                                  up=look_up.view(1, 3))
    camera = PerspectiveCameras(device=lights.device, R=R, T=T, principal_point=(half_half,), focal_length=(half_half,),
                                image_size=((512, 512),), in_ndc=False)

    # ####define shader
    blend_params = BlendParams(0.5, 1e-4, (0, 0, 0))
    materials = Materials(device=lights.device)
    shader = HardPhongShader(lights=lights, cameras=camera, materials=materials, blend_params=blend_params)

    # ####define raster
    raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1,
                                            cull_backfaces=False)
    rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings)

    # ####define render
    renderer = MeshRenderer(rasterizer=MeshRasterizer(raster_settings=raster_settings, cameras=camera), shader=shader)

    return rasterizer, renderer


def render_object(file_name, rasterizer, renderer, meshes, save_npy_dir, save_pic_path, toward='up'):
    # ####render image
    fragments = rasterizer(meshes)
    image = renderer(meshes)
    zbuf = fragments.zbuf *1.0
    render_image = image[..., :3].squeeze().detach().cpu().numpy()
    render_mask = image[..., 3].squeeze().detach().cpu().numpy()

    # ####get depth map
    depth_base = 0.0
    zbuf = zbuf.squeeze().detach().cpu().numpy()
    zbuf = 1.0 - (zbuf - np.min(zbuf[zbuf != -1])) / (np.max(zbuf) - np.min(zbuf[zbuf != -1]))
    zbuf = np.clip(zbuf + depth_base, a_min=0.0, a_max=1.0)
    zbuf[render_mask == 0] = 0
    zbuf_rgb = np.broadcast_to(zbuf, (3, zbuf.shape[0], zbuf.shape[1])).transpose((1, 2, 0))

    # ####get edge image
    edge = skimage.feature.canny(skimage.color.rgb2gray(render_image)).astype(np.float32)
    edge_rgb = np.broadcast_to(edge, (3, edge.shape[0], edge.shape[1])).transpose((1, 2, 0))

    # # # #####plot image
    # plt.imshow(render_image)
    # plt.show()
    # plt.imshow(render_mask, cmap='gray')
    # plt.show()
    # plt.imshow(zbuf, cmap='gray')
    # plt.show()
    # plt.imshow(edge, cmap='gray')
    # plt.show()
    # # ####plot rgb image
    # plt.imshow(zbuf_rgb)
    # plt.show()
    # plt.imshow(edge_rgb)
    # plt.show()

    # ######save pic
    os.makedirs('{}{}/render_img/'.format(save_pic_path, toward), exist_ok=True)
    os.makedirs('{}{}/depth_img/'.format(save_pic_path, toward), exist_ok=True)
    os.makedirs('{}{}/edge_img/'.format(save_pic_path, toward), exist_ok=True)
    plt.imshow(render_image)
    plt.savefig('{}{}/render_img/{}.png'.format(save_pic_path, toward, file_name))
    plt.close()
    plt.imshow(zbuf_rgb)
    plt.savefig('{}{}/depth_img/{}.png'.format(save_pic_path, toward, file_name))
    plt.close()
    plt.imshow(edge_rgb)
    plt.savefig('{}{}/edge_img/{}.png'.format(save_pic_path, toward, file_name))
    plt.close()
    #
    # # #####save npy
    os.makedirs('{}{}/'.format(save_npy_dir, toward), exist_ok=True)
    np.save('{}{}/{}_render.npy'.format(save_npy_dir, toward, file_name), render_image)
    np.save('{}{}/{}_depth.npy'.format(save_npy_dir, toward, file_name), zbuf_rgb)
    np.save('{}{}/{}_edge.npy'.format(save_npy_dir, toward, file_name), edge_rgb)
    np.save('{}{}/{}_mask.npy'.format(save_npy_dir, toward, file_name), render_mask)

    pass


