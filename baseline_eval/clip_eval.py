import os
import glob
import time
import torch
import skimage
import argparse
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import clip
from pytorch_lightning import seed_everything
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from datasets.dataset_utils import rotate_point_cloud, determine_best_view, normalize_point
from utils.classify_util import chunk, get_mesh, init_render_with_pca, render_object_style, class_accuracy
from utils.projection_util import Realistic_Projection


# #### test class list map
class_map = {'ant': 0, 'bird': 1, 'crab': 2, 'dinosaur': 3, 'dolphin': 4, 'fish': 5, 'hand': 6, 'octopus': 7,
             'pliers': 8, 'quadruped': 9, 'snake': 10, 'spectacle': 11, 'spider': 12, 'teddy': 13}

test_class = ['ant', 'bird', 'crab', 'dinosaur', 'dolphin', 'fish', 'hand', 'octopus', 'pliers', 'quadruped', 'snake',
              'spectacle', 'spider', 'teddy']

def rot_img(x, degree, dtype):
    # plt.imshow(x.squeeze().permute(1, 2, 0).detach().cpu().numpy())
    # plt.show()

    theta = torch.deg2rad(degree)
    rot_mat = torch.zeros(2, 3)
    rot_mat[0, 0] = torch.cos(theta)
    rot_mat[0, 1] = -torch.sin(theta)
    rot_mat[1, 0] = torch.sin(theta)
    rot_mat[1, 1] = torch.cos(theta)
    rot_mat = rot_mat[None, ...].type(dtype).repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid.cuda(), padding_mode='border')

    # plt.imshow(x.squeeze().permute(1,2,0).detach().cpu().numpy())
    # plt.show()
    return x.squeeze()   # c w h

def real_proj(pc, imsize=224):
    pc_views = Realistic_Projection()
    img = pc_views.get_img(pc).cuda()
    img = torch.nn.functional.interpolate(img, size=(imsize, imsize), mode='bilinear', align_corners=True)
    return img

def main():
    parser = argparse.ArgumentParser()
    # ####set stable diffusion parameter
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible sampling)")

    # ##### set the dataset parameter
    parser.add_argument("--data_path", type=str, default='datasets/op_mcgill/', help="npy data path")
    parser.add_argument("--style", type=str, default='edge', help="render, depth or edge")
    parser.add_argument("--bg_color", type=str, default='white', help="black or white")
    parser.add_argument("--PL_th", type=int, default=9, help="nth prompt list file")
    parser.add_argument("--phi_1", type=float, default=60, help="angle for rotation")
    parser.add_argument("--phi_2", type=float, default=210, help="angle for rotation")
    parser.add_argument("--r_p", type=float, default=2.2, help="distance from camera")

    parser.add_argument("--grad_step_num", type=int, default=3, help="k-step")
    parser.add_argument("--step_value", type=float, default=5, help="distance from camera")
    parser.add_argument("--step_reduce", type=float, default=1, help="distance from camera")

    opt = parser.parse_args()
    seed_everything(opt.seed)

    r_p = opt.r_p
    phi_1 = opt.phi_1
    phi_2 = opt.phi_2

    # ####get prompt list from file
    prompt_path = '{}PL_{}/PL{}.txt'.format(opt.data_path, opt.style, str(opt.PL_th))
    print('get prompts from{}'.format(prompt_path))
    with open(prompt_path, "r") as f:
        data = f.read().splitlines()
    text = clip.tokenize(data).cuda()

    pretrain_model = "ViT-B/16"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(pretrain_model, device=device) # ViT-B/32
    img_norm = preprocess.transforms[4]

    # #### get prompt embedding
    model = model.cuda()
    label_list = []
    pred_list = []
    probs_list = []
    time_list = []
    Tp = 0

    # ####output path
    output_path = '{}results_clip/'.format(opt.data_path)
    os.makedirs(output_path, exist_ok=True)
    test_file_list = glob.glob('datasets/op_mcgill/mcgill_op_npy/*_xyz.npy')
    test_file_list.sort()

    for f_i, f_path in enumerate(test_file_list):
        start_time = time.time()
        class_name = f_path.split('/')[-1].split('_')[0]
        file_name = f_path.split('/')[-1].split('.')[0][:-4]
        cur_label = class_map[class_name]
        label_list.append(cur_label)

        # ####get base information for pointcloud
        xyz_op = np.load('{}/mcgill_op_npy/{}_xyz.npy'.format(opt.data_path, file_name))
        # faces = np.load('{}/mcgill_op_npy/{}_faces.npy'.format(opt.data_path, file_name))

        init_image = real_proj(torch.from_numpy(xyz_op).unsqueeze(0).type(torch.float32).cuda())
        # init_image = init_image.permute(0, 2, 3, 1).squeeze()
        init_image = init_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        edge = skimage.feature.canny(skimage.color.rgb2gray(init_image)).astype(np.float32)
        edge_rgb = np.broadcast_to(edge, (3, edge.shape[0], edge.shape[1])).transpose((1, 2, 0))
        init_image = torch.from_numpy(1 - edge_rgb).cuda()

        # plt.imshow(init_image.detach().cpu().numpy())
        # plt.show()

        # if opt.bg_color == 'white':
        #     if opt.style == 'edge':
        #         init_image = 1 - init_image
        #     else:
        #         init_image[image_mask == 0, ...] = 1
        # plt.imshow(init_image)
        # plt.show()

        # ####
        with torch.no_grad():
            rotate_list = []
            indice_list = []
            # init_image = torch.from_numpy(init_image).cuda()

            # ######################rotate 0
            init_image_rotated = rot_img(init_image.unsqueeze(0).permute(0, 3, 1, 2), torch.tensor([0.0]).cuda(),
                                         dtype=torch.float32)
            init_image_rotated = img_norm(init_image_rotated).unsqueeze(0)
            logits_per_image, logits_per_text = model(init_image_rotated, text)
            probs_0 = logits_per_image.squeeze()

            _, indices = torch.topk(probs_0, k=14, largest=True)
            for id_i, ind in enumerate(indices):
                ind_k = ind % 14
                if ind_k in indice_list or len(indice_list) >= 3:
                    continue
                else:
                    indice_list.append(ind_k.item())
                    rotate_list.append(ind // 14)

            indices = torch.tensor(indice_list).type(torch.int64).cuda()
            rotate_list = torch.tensor(rotate_list).type(torch.int64).cuda()
            print('the proposed idx: {}'.format(indices))

        # ########=============================================GRAD Top3=====================================================##########
        # ####rotate image
        init_angle_raw = torch.tensor([0.0]).cuda()
        k_text_probs = torch.zeros(text[indices].shape[0])
        torch.cuda.empty_cache()

        for p_i in range(k_text_probs.shape[0]):
            grad_step_num = opt.grad_step_num
            step_value = opt.step_value
            step_reduce = opt.step_reduce
            probs_rotate = torch.zeros(grad_step_num)
            angle_rotate = init_angle_raw * 1.0
            angle_rotate += rotate_list[p_i] * 90
            angle_rotate.requires_grad = True
            print('init rotate angle {}'.format(angle_rotate))

            for r_i in range(grad_step_num):
                cur_step_value = step_value - step_reduce * r_i

                init_image_rotated = rot_img(init_image.unsqueeze(0).permute(0, 3, 1, 2),
                                             angle_rotate,
                                             dtype=torch.float32)
                init_image_rotated = img_norm(init_image_rotated).unsqueeze(0)
                logits_per_image, logits_per_text = model(init_image_rotated, text[indices[p_i]].view(-1,77))
                probs = logits_per_image.type(torch.float32)

                grad = torch.autograd.grad(outputs=probs, inputs=angle_rotate, only_inputs=True)[0]
                with torch.no_grad():
                    if grad >= 0:
                        angle_rotate = angle_rotate + cur_step_value
                    else:
                        angle_rotate = angle_rotate - cur_step_value

                angle_rotate.requires_grad = True

                print('cur_prompt: {}, cur_step: {}, cur_grad: {}, cur_probs: {}, next angle: {}'
                      .format(indices[p_i], r_i, grad, probs, angle_rotate))
                probs_rotate[r_i] = probs
            k_text_probs[p_i] = probs_rotate.max()
        pred_cls = indices[torch.argmax(k_text_probs)]
        probs = probs.detach().cpu().numpy()
        pred_cls = pred_cls.detach().cpu().numpy()
        if pred_cls == cur_label:
            Tp += 1
        acc = Tp / (f_i + 1)
        pred_list.append(pred_cls)
        probs_list.append(probs)
        time_list.append(time.time() - start_time)
        print('========complete: {}/{} samples, current:{}  label/pred: {}/{}  all acc: {}============'.
          format(f_i, len(test_file_list), file_name, cur_label, pred_cls, acc))
    np.save(output_path + 'grad_{}_{}_{}_{}_{}_{}_PL{}_CH{}_label.npy'.format(pretrain_model[-2:], opt.style, opt.bg_color,
                                                                         str(opt.seed),
                                                                         str(opt.phi_1), str(opt.phi_2), str(opt.PL_th),
                                                                         str(int(opt.r_p * 10))), np.array(label_list))
    np.save(output_path + 'grad_{}_{}_{}_{}_{}_{}_PL{}_CH{}_pred.npy'.format(pretrain_model[-2:], opt.style, opt.bg_color,
                                                                        str(opt.seed),
                                                                        str(opt.phi_1), str(opt.phi_2), str(opt.PL_th),
                                                                        str(int(opt.r_p * 10))), np.array(pred_list))
    np.save(output_path + 'grad_{}_{}_{}_{}_{}_{}_PL{}_CH{}_prob.npy'.format(pretrain_model[-2:], opt.style, opt.bg_color,
                                                                        str(opt.seed),
                                                                        str(opt.phi_1), str(opt.phi_2), str(opt.PL_th),
                                                                        str(int(opt.r_p * 10))), np.array(probs_list))
    per_accuracy = class_accuracy(labels=np.array(label_list), predictions=np.array(pred_list), test_class=test_class)
    print('average time for each sample is {}'.format(np.mean(time_list)))
    pass


if __name__ == "__main__":
    main()

