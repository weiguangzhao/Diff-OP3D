import os
import glob
import time
import torch
import argparse
import numpy as np
import skimage
import torch.nn.functional as F

from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from scipy.spatial.transform import Rotation

from datasets.dataset_utils import rotate_point_cloud, determine_best_view
from utils.classify_util import load_model_from_config, chunk, DDIMSampler, class_accuracy
from utils.grad_sample import DDIMSampler_grad
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

def real_proj(pc, imsize=512):
    pc_views = Realistic_Projection()
    img = pc_views.get_img(pc).cuda()
    img = torch.nn.functional.interpolate(img, size=(imsize, imsize), mode='bilinear', align_corners=True)
    return img

def main():
    parser = argparse.ArgumentParser()
    # ####set stable diffusion parameter
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible sampling)")
    parser.add_argument("--e_num", type=int, default=30, help="the number of noise e_init (shape[0])")
    parser.add_argument("--step", type=int, default=600, help="the timestep for stable diffusion")
    parser.add_argument("--n_samples", type=int, default=14, help="how many samples to produce for each given prompt. A.k.a. batch size")
    parser.add_argument("--scale", type=float, default=1.00, help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")

    # ##### set the dataset pa rameter
    parser.add_argument("--data_path", type=str, default='datasets/op_mcgill/', help="npy data path")
    parser.add_argument("--style", type=str, default='edge', help="render, depth or edge")
    parser.add_argument("--bg_color", type=str, default='white', help="black or white")
    parser.add_argument("--PL_th", type=int, default=6, help="nth prompt list file")
    parser.add_argument("--r_p", type=float, default=-2.2, help="distance from camera")

    parser.add_argument("--grad_step_num", type=int, default=3, help="distance from camera")
    parser.add_argument("--step_value", type=float, default=5, help="distance from camera")
    parser.add_argument("--step_reduce", type=float, default=2, help="distance from camera")

    # #####defau config
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml", help="path to config which constructs model", )
    parser.add_argument("--ckpt", type=str, default="models/ldm/stable-diffusion-v1/model.ckpt", help="path to checkpoint of model")
    parser.add_argument("--ddim_steps", type=int, default=1000, help="number of ddim sampling steps")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="ddim eta (eta=0.0 corresponds to deterministic sampling")
    parser.add_argument("--H", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--W", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--C", type=int, default=4, help="latent channels")
    parser.add_argument("--f", type=int, default=8, help="downsampling factor")
    opt = parser.parse_args()

    batch_size = opt.n_samples
    step_chose = opt.step
    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    r_p = opt.r_p

    # ####get prompt list from file
    prompt_path = '{}PL_{}/PL{}.txt'.format(opt.data_path, opt.style, str(opt.PL_th))
    print('get prompts from{}'.format(prompt_path))
    with open(prompt_path, "r") as f:
        data = f.read().splitlines()
        data = list(chunk(data, batch_size))

    # #### get prompt embedding
    model = model.cuda()
    prompts_embedding = []
    uc = model.get_learned_conditioning(batch_size * [""])
    uc = torch.repeat_interleave(uc, repeats=batch_size, dim=0)
    for prompts in data:
        prompts = list(prompts)
        prompts_embedding.append(model.get_learned_conditioning(prompts))
    prompts_embedding = torch.cat(prompts_embedding, dim=0)

    label_list = []
    pred_list = []
    mse_list = []
    ind_list = []
    Tp = 0

    # ####output path
    output_path = '{}final_eval/op_mcgill/'.format(opt.data_path)
    os.makedirs(output_path, exist_ok=True)
    test_file_list = glob.glob('datasets/op_mcgill/npy/*_xyz.npy')
    test_file_list.sort()
    random_angle = np.load('datasets/op_mcgill/random_angle.npy')
    ##161
    for f_i, f_path in enumerate(test_file_list):
        start_time = time.time()
        class_name = f_path.split('/')[-1].split('_')[0]
        file_name = f_path.split('/')[-1].split('.')[0][:-4]
        cur_label = class_map[class_name]
        # label_list.append(cur_label)

        # ####get base information for pointcloud
        xyz_raw = np.load('{}/npy/{}_xyz.npy'.format(opt.data_path, file_name))
        faces = np.load('{}/npy/{}_faces.npy'.format(opt.data_path, file_name))
        # ####rotate point cloud
        cur_random_angle = random_angle[f_i, :]
        xyz_rotation = rotate_point_cloud(xyz_raw, cur_random_angle[0], cur_random_angle[1], cur_random_angle[2])

        views = determine_best_view(xyz_rotation)
        rotation = Rotation.align_vectors(views.reshape(-1, 3), np.array([0.0, 0.0, 1.0]).reshape(-1, 3))
        rotation_matrix = rotation[0].as_matrix()
        rotated_vector = np.dot(xyz_raw, rotation_matrix)
        init_image = real_proj(torch.from_numpy(rotated_vector).unsqueeze(0).type(torch.float32).cuda())
        init_image = init_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        edge = skimage.feature.canny(skimage.color.rgb2gray(init_image)).astype(np.float32)
        edge_rgb = np.broadcast_to(edge, (3, edge.shape[0], edge.shape[1])).transpose((1, 2, 0))
        init_image = torch.from_numpy(1 - edge_rgb).cuda()

        # if opt.bg_color == 'white':
        #     if opt.style == 'edge':
        #         init_image = 1 - init_image
        #     else:
        #         init_image[image_mask == 0, ...] = 1
        # plt.imshow(init_image)
        # plt.show()

        with torch.no_grad():
            with model.ema_scope():
                rotate_list = []
                indice_list = []
                # init_image = torch.from_numpy(init_image).cuda()
                # ####latent space
                init_image_rotated = rot_img(init_image.unsqueeze(0).permute(0, 3, 1, 2), torch.tensor([0.0]).cuda(), dtype=torch.float32)
                e_init = torch.randn([opt.e_num, 4, 64, 64]).cuda()
                init_image_latent = 2 * init_image_rotated - 1.0  # ### c w h
                init_image_latent= init_image_latent.unsqueeze(0)
                init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image_latent))
                init_latent = torch.repeat_interleave(init_latent, repeats=batch_size, dim=0)

                sampler = DDIMSampler(model)
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                mse_t_0 = sampler.sample(S=opt.ddim_steps, conditioning=prompts_embedding, batch_size=opt.n_samples,
                                       shape=shape, verbose=False,
                                       unconditional_guidance_scale=opt.scale,
                                       unconditional_conditioning=uc, eta=opt.ddim_eta,
                                       x_init=init_latent, e_init=e_init, step_chose=step_chose)

                _, indices = torch.topk(mse_t_0, k=3, largest=False)
                for id_i, ind in enumerate(indices):
                    ind_k = ind % 14
                    if ind_k in indice_list or len(indice_list) >= 3: continue
                    else:
                        indice_list.append(ind_k.item())
                        rotate_list.append(ind//14)
        indices = torch.tensor(indice_list).type(torch.int64).cuda()
        ind_list.append(indices.detach().cpu().numpy())
        rotate_list = torch.tensor(rotate_list).type(torch.int64).cuda()
        print('the proposed idx: {}'.format(indices))

        if cur_label not in indices:
            pred_list.append(indices[0].detach().cpu().numpy())
            label_list.append(cur_label)
            mse_list.append(mse_t_0[indices].detach().cpu().numpy())
            continue
        label_list.append(cur_label)
        # ########=============================================GRAD Top3=====================================================##########
        # ####rotate image
        init_angle_raw = torch.tensor([0.0]).cuda()
        k_prompts_mse = torch.zeros(prompts_embedding[indices].shape[0])
        sampler = DDIMSampler_grad(model)
        torch.cuda.empty_cache()

        for p_i in range(prompts_embedding[indices].shape[0]):
            grad_step_num = opt.grad_step_num
            step_value = opt.step_value
            step_reduce = opt.step_reduce
            mse_rotate = torch.zeros(grad_step_num)
            angle_rotate = init_angle_raw * 1.0
            angle_rotate += rotate_list[p_i]*90
            angle_rotate.requires_grad = True
            print('init rotate angle {}'.format(angle_rotate))

            for r_i in range(grad_step_num):

                cur_step_value = step_value - step_reduce * r_i

                grad = 0
                mse_t = 0
                for e_i in range(3):
                    # #####rotate image
                    init_image_rotated = rot_img(init_image.unsqueeze(0).permute(0, 3, 1, 2), angle_rotate,
                                                 dtype=torch.float32)
                    # ####latent space
                    init_image_rotated = 2 * init_image_rotated - 1.0  # ### c w h
                    init_image_rotated = init_image_rotated.unsqueeze(0)
                    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image_rotated))
                    init_latent = torch.repeat_interleave(init_latent, repeats=1, dim=0)

                    e_init_range = e_init[e_i*10:(e_i+1)*10]
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    cur_mse = sampler.sample(S=opt.ddim_steps, conditioning=prompts_embedding[indices[p_i]].unsqueeze(0),
                                           batch_size=1, shape=shape, verbose=False,
                                           unconditional_guidance_scale=opt.scale,
                                           unconditional_conditioning=uc[indices[p_i]].unsqueeze(0), eta=opt.ddim_eta,
                                           x_init=init_latent, e_init=e_init_range, step_chose=step_chose)

                    cur_grad = torch.autograd.grad(outputs=cur_mse, inputs=angle_rotate, only_inputs=True)[0]
                    grad += cur_grad
                    mse_t +=cur_mse

                with torch.no_grad():
                    if grad >= 0:
                        angle_rotate = angle_rotate - cur_step_value
                    else:
                        angle_rotate = angle_rotate + cur_step_value
                    mse_t = mse_t/3.0

                angle_rotate.requires_grad = True

                print('cur_prompt: {}, cur_step: {}, cur_grad: {}, cur_mse: {}, next angle: {}'
                      .format(indices[p_i], r_i, grad, mse_t, angle_rotate))
                mse_rotate[r_i] = mse_t
            k_prompts_mse[p_i] = mse_rotate.min()
        pred_cls = indices[torch.argmin(k_prompts_mse)]

        pred_cls = pred_cls.detach().cpu().numpy()
        k_prompts_mse = k_prompts_mse.detach().cpu().numpy()
        if pred_cls == cur_label:
            Tp += 1
        acc = Tp / (f_i + 1)
        pred_list.append(pred_cls)
        mse_list.append(k_prompts_mse)
        print('========complete: {}/{} samples, current:{}  label/pred: {}/{}  all acc: {}============'.
              format(f_i, len(test_file_list), file_name, cur_label, pred_cls, acc))
    np.save(output_path + 'PCA_four_grad_{}_{}_{}_{}_{}_PL{}_CH{}_label.npy'.format(opt.style, opt.bg_color, str(opt.seed),
                                                                               str(opt.e_num), str(opt.step),
                                                                               str(opt.PL_th),
                                                                               str(int(opt.r_p * 10))),
            np.array(label_list))
    np.save(output_path + 'PCA_four_grad_{}_{}_{}_{}_{}_PL{}_CH{}_pred.npy'.format(opt.style, opt.bg_color, str(opt.seed),
                                                                              str(opt.e_num), str(opt.step),
                                                                              str(opt.PL_th),
                                                                              str(int(opt.r_p * 10))),
            np.array(pred_list))
    np.save(output_path + 'PCA_four_grad_{}_{}_{}_{}_{}_PL{}_CH{}_mse.npy'.format(opt.style, opt.bg_color, str(opt.seed),
                                                                             str(opt.e_num), str(opt.step),
                                                                             str(opt.PL_th),
                                                                             str(int(opt.r_p * 10))),
            np.array(mse_list))
    np.save(output_path + 'PCA_four_grad_{}_{}_{}_{}_{}_PL{}_CH{}_ind.npy'.format(opt.style, opt.bg_color, str(opt.seed),
                                                                             str(opt.e_num), str(opt.step),
                                                                             str(opt.PL_th),
                                                                             str(int(opt.r_p * 10))),
            np.array(ind_list))
    per_accuracy = class_accuracy(labels=np.array(label_list), predictions=np.array(pred_list), test_class=test_class)
    pass


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    main()

