import torch
from tqdm import tqdm


class DiffusionImageMechanism:

    def __init__(self, model, guidance_scale):
        self.model = model
        self.guidance_scale = guidance_scale

    def ddim_inversion(self, image, cond):
        batch_size = image.shape[0]
        xt = image.clone()
        for _t in tqdm(range(self.model.sampler.pred_timesteps), desc='DDIM Inversion'):
            t = torch.ones(batch_size, dtype=torch.long, device="cuda") * _t
            e = self.model.model.unet(xt, t, cond)
            x0_t = (
                torch.sqrt(1.0 / self.model.sampler.alphas_cumprod[t])[:, None, None, None] * xt
                - torch.sqrt(1.0 / self.model.sampler.alphas_cumprod[t] - 1)[:, None, None, None] * e
            ).clamp(-1, 1)
            e = (
                (torch.sqrt(1.0 / self.model.sampler.alphas_cumprod[t])[:, None, None, None] * xt - x0_t)
                / (torch.sqrt(1.0 / self.model.sampler.alphas_cumprod[t] - 1)[:, None, None, None])
            )
            xt = (
                torch.sqrt(self.model.sampler.alphas_cumprod_next[t])[:, None, None, None] * x0_t
                + torch.sqrt(1 - self.model.sampler.alphas_cumprod_next[t])[:, None, None, None] * e
            )
        return xt
        
    def ddim(self, xt, cond):
        batch_size = xt.shape[0]
        xt = xt.clone()
        for _t in tqdm(
            reversed(range(self.model.sampler.pred_timesteps)),
            desc='DDIM',
            total=self.model.sampler.pred_timesteps,
        ):
            t = torch.ones(batch_size, dtype=torch.long, device="cuda") * _t
            if self.model.model.p_uncond >= 0:
                e_uncond = self.model.model.unet(xt, t, self.model.model.null_token)
                e_cond = self.model.model.unet(xt, t, cond)
                e = e_uncond + self.guidance_scale * (e_cond - e_uncond)
            else:
                e = self.model.model.unet(xt, t, cond)
            x0_t = (
                torch.sqrt(1.0 / self.model.sampler.alphas_cumprod[t])[:, None, None, None] * xt
                - torch.sqrt(1.0 / self.model.sampler.alphas_cumprod[t] - 1)[:, None, None, None] * e
            ).clamp(-1, 1)
            e = (
                (torch.sqrt(1.0 / self.model.sampler.alphas_cumprod[t])[:, None, None, None] * xt - x0_t)
                / (torch.sqrt(1.0 / self.model.sampler.alphas_cumprod[t] - 1)[:, None, None, None])
            )
            xt = (
                torch.sqrt(self.model.sampler.alphas_cumprod_prev[t])[:, None, None, None] * x0_t
                + torch.sqrt(1 - self.model.sampler.alphas_cumprod_prev[t])[:, None, None, None] * e
            )
        return xt
    
    def abduct(self, obs):
        noise = {}
        image, cond = obs["image"], obs["metadata"].float()
        if self.model.model.use_semantic_encoder:
            if self.model.model.gaussian_encoder:
                z, _ = self.model.model.encoder(image, cond, use_smoothing=False)
            else:
                z = self.model.model.encoder(image, cond, use_smoothing=False)
            noise["z"] = z
            cond = torch.cat([z, cond], dim=1)
        u = self.ddim_inversion(image, cond)
        noise["u"] = u
        return noise
    
    def predict(self, noise, cond):
        cond = cond["metadata"].float()
        if self.model.model.use_semantic_encoder:  # and not self.model.model.cond_semantic_encoder:
            cond = torch.cat([noise["z"], cond], dim=1)
        image = self.ddim(noise["u"], cond)
        return {"image": image}

    # def counterfactual(self, image, cond, interv):
    #     return self.predict(self.abduct(image, cond), interv)