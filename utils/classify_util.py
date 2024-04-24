import torch
import math
import numpy as np
import skimage
from itertools import islice

# ####ldm lib
from ldm.util import instantiate_from_config
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps

# ####import pytorch3d
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (look_at_view_transform, PerspectiveCameras, RasterizationSettings, MeshRasterizer,
                                TexturesVertex, PointLights, MeshRenderer, HardPhongShader, Materials, HardFlatShader)
from pytorch3d.renderer.mesh.shader import (BlendParams)
from datasets.dataset_utils import make_front_faces, calculate_target_coordinates


# ####split data
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

# ####load model config
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model



# ####calculate class result
def class_accuracy(labels, predictions, test_class):
    num_classes = len(test_class)
    class_acc = np.zeros(num_classes)
    for c in range(num_classes):
        class_mask = (labels == c)
        class_preds = predictions[class_mask]
        class_labels = labels[class_mask]
        class_acc[c] = np.mean(class_preds == class_labels)
        print("{:<20s} {:.2%}".format(test_class[c], class_acc[c]))
    print("Average accuracy: {:.2%}".format(class_acc.sum() / num_classes))
    return class_acc


def get_mesh(verts, faces):
    verts_rgb = torch.ones_like(verts)[None] * torch.tensor([1.0, 1.0, 1.0], device=verts.device)
    textures = TexturesVertex(verts_features=verts_rgb)
    meshes = Meshes(verts=[verts], faces=[faces], textures=textures).cuda()
    meshes._compute_face_areas_normals()
    meshes = make_front_faces(meshes)
    return meshes


def init_render_with_pca(r_p, views, img_size=512.0):
    x, y, z = views[0]*r_p,   views[1]*r_p,  views[2]*r_p
    camera_position = torch.tensor([x, y, z]).type(torch.float32)
    # look_up = torch.tensor([views[0], views[1], views[2]]).type(torch.float32)
    look_up = torch.tensor([0.0, 0.0, 1.0]).type(torch.float32)

    lights = PointLights(location=torch.tensor([[2.0, 2.0, 2.0]],dtype=torch.float32)).cuda()

    # creat PerspectiveCameras
    half_half = (img_size / 2.0, img_size / 2.0)

    R, T = look_at_view_transform(eye=camera_position.view(1, 3), at=torch.tensor([[0.0, 0.0, 0.0]]),
                                  up=look_up.view(1, 3))
    camera = PerspectiveCameras(device=lights.device, R=R, T=T, principal_point=(half_half,), focal_length=(half_half,),
                                image_size=((img_size, img_size),), in_ndc=False)

    # ####define shader
    blend_params = BlendParams(0.5, 1e-4, (0, 0, 0))
    materials = Materials(device=lights.device)
    shader = HardPhongShader(lights=lights, cameras=camera, materials=materials, blend_params=blend_params)

    # ####define raster
    raster_settings = RasterizationSettings(image_size=int(img_size), blur_radius=0.0, faces_per_pixel=1,
                                            cull_backfaces=False)
    rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings)

    # ####define render
    renderer = MeshRenderer(rasterizer=MeshRasterizer(raster_settings=raster_settings, cameras=camera), shader=shader)

    return rasterizer, renderer


def render_object_style(rasterizer, renderer, mesh, img_style):
    # ####render image
    fragments = rasterizer(mesh)
    image = renderer(mesh)
    zbuf = fragments.zbuf * 1.0
    render_image = image[..., :3].squeeze().detach().cpu().numpy()
    render_mask = image[..., 3].squeeze().detach().cpu().numpy()

    # ####get depth map
    if img_style=='depth':
        depth_base = 0.0
        zbuf = zbuf.squeeze().detach().cpu().numpy()
        zbuf = 1.0 - (zbuf - np.min(zbuf[zbuf != -1])) / (np.max(zbuf) - np.min(zbuf[zbuf != -1]))
        zbuf = np.clip(zbuf + depth_base, a_min=0.0, a_max=1.0)
        zbuf[render_mask == 0] = 0
        zbuf_rgb = np.broadcast_to(zbuf, (3, zbuf.shape[0], zbuf.shape[1])).transpose((1, 2, 0))
        return zbuf_rgb.copy(), render_mask

    # ####get edge image
    if img_style == 'edge':
        depth_base = 0.0
        zbuf = zbuf.squeeze().detach().cpu().numpy()
        zbuf = 1.0 - (zbuf - np.min(zbuf[zbuf != -1])) / (np.max(zbuf) - np.min(zbuf[zbuf != -1]))
        zbuf = np.clip(zbuf + depth_base, a_min=0.0, a_max=1.0)
        zbuf[render_mask == 0] = 0
        zbuf_rgb = np.broadcast_to(zbuf, (3, zbuf.shape[0], zbuf.shape[1])).transpose((1, 2, 0))

        edge = skimage.feature.canny(skimage.color.rgb2gray(zbuf_rgb)).astype(np.float32)
        edge_rgb = np.broadcast_to(edge, (3, edge.shape[0], edge.shape[1])).transpose((1, 2, 0))
        return edge_rgb, None

    return render_image, render_mask



# #### DDIM Sample
class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps  = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        self.ddim_timesteps = self.ddim_timesteps  - 1
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def sample(self, S, batch_size, shape,  conditioning=None, callback=None, normals_sequence=None, img_callback=None,
               quantize_x0=False,  eta=0., mask=None, x0=None, temperature=1., noise_dropout=0., score_corrector=None,
               corrector_kwargs=None, verbose=True, x_init=None,  log_every_t=100,  unconditional_guidance_scale=1.,
               unconditional_conditioning=None, e_init=None, step_chose=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning.shape[0] != batch_size:
            print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        # print(f'Data shape for DDIM sampling is {size}, eta {eta}')
        mse_t_sum = torch.zeros(batch_size).cuda()
        for e_i in range(e_init.shape[0]):
            cur_e_init = e_init[e_i, ...].unsqueeze(0)
            cur_e_init = torch.repeat_interleave(cur_e_init, repeats=batch_size, dim=0)
            mse_t = self.ddim_sampling(conditioning, size, callback=callback, img_callback=img_callback,
                                       quantize_denoised=quantize_x0, mask=mask, x0=x0,
                                       ddim_use_original_steps=False, noise_dropout=noise_dropout,
                                       temperature=temperature, score_corrector=score_corrector,
                                       corrector_kwargs=corrector_kwargs, x_init=x_init, log_every_t=log_every_t,
                                       unconditional_guidance_scale=unconditional_guidance_scale,
                                       unconditional_conditioning=unconditional_conditioning, e_init=cur_e_init,
                                       step_chose=step_chose
                                       )
            mse_t_sum += mse_t
        return mse_t_sum/e_init.shape[0]


    @torch.no_grad()
    def ddim_sampling(self, cond, shape, x_init=None, ddim_use_original_steps=False, callback=None, timesteps=None,
                      quantize_denoised=False, mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,  e_init=None,
                      step_chose=None):
        device = self.model.betas.device
        b = shape[0]
        img = x_init

        # ####use fixed idx
        index = step_chose
        step = step_chose
        ts = torch.full((b,), step, device=device, dtype=torch.long)
        mse_t = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                  quantize_denoised=quantize_denoised, temperature=temperature,
                                  noise_dropout=noise_dropout, score_corrector=score_corrector,
                                  corrector_kwargs=corrector_kwargs,
                                  unconditional_guidance_scale=unconditional_guidance_scale,
                                  unconditional_conditioning=unconditional_conditioning, e_init=e_init)
        return mse_t



    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, e_init=None):
        b, *_, device = *x.shape, x.device
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        x_t = x * a_t.sqrt() + sqrt_one_minus_at * e_init

        #  #####set mini-batch to suppression cuda room boom
        mini_batch = 5
        et_list = []
        for m_i in range(math.ceil(b/mini_batch)):
            min_x_t = x_t[m_i*mini_batch:min((m_i+1)*mini_batch, b), ...]
            min_t = t[m_i*mini_batch:min((m_i+1)*mini_batch, b), ...]
            min_uc = unconditional_conditioning[m_i*mini_batch:min((m_i+1)*mini_batch, b), ...]
            min_c = c[m_i*mini_batch:min((m_i+1)*mini_batch, b), ...]
            x_in = torch.cat([min_x_t] * 2)
            t_in = torch.cat([min_t] * 2)
            c_in = torch.cat([min_uc, min_c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)  # (1, 4,64 ,64)
            et_list.append(e_t)
        e_t = torch.cat(et_list, dim=0)
        mse_t = torch.sum(torch.square(e_init - e_t), dim=(1, 2, 3))
        return mse_t