import torch
import numpy as np

# ####ldm lib
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps

# #### DDIM Sample
class DDIMSampler_grad(object):
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
        # x_prev, pred_x0, mse_t = outs
        return mse_t




    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, e_init=None):
        b, *_, device = *x.shape, x.device
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        x_t = x * a_t.sqrt() + sqrt_one_minus_at * e_init

        x_in = torch.cat([x_t] * 2)
        t_in = torch.cat([t] * 2)
        c_in = torch.cat([unconditional_conditioning, c])
        e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
        e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)  # (1, 4,64 ,64)
        mse_t = torch.sum(torch.square(e_init - e_t), dim=(1, 2, 3))

        # # current prediction for x_0
        # pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        # # direction pointing to x_t
        # dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        # noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        # x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return mse_t