import torch
from tqdm import tqdm

import torch.nn.functional as F


class COImageMechanism:

    def __init__(self, model, guidance_scale, cta_steps, learning_rate):
        self.model = model
        if self.model.model.p_uncond < 0:
            raise RuntimeError("Can only be performed with CFG model")
        self.guidance_scale = guidance_scale
        self.cta_steps = cta_steps
        self.learning_rate = learning_rate

    @torch.no_grad()
    def _ddim_inversion(self, image, cond):
        xt = image.clone()
        xts = [xt]
        for _t in tqdm(range(self.model.sampler.pred_timesteps), desc='DDIM Inversion'):
            t = torch.ones(1, dtype=torch.long, device="cuda") * _t
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
            xts.append(xt)
        return xts

    def _fit(self, xts, cond):
        assert len(xts) == self.model.sampler.pred_timesteps + 1

        optimised_null_tokens = []
        xt = xts[-1].clone()
        
        # Reverse the intermediates returned from _ddim_inversion
        xts = list(reversed(xts))

        # Initialise null token
        null_token = self.model.model.null_token.clone().requires_grad_(True)
        # print("nt:", null_token.shape)
        # null_token.requires_grad_(True)
        
        # Reverse process from T -> 1
        for _t in tqdm(
            reversed(range(self.model.sampler.pred_timesteps)),
            desc='Fitting',
            total=self.model.sampler.pred_timesteps,
        ):
            # print(_t)
            
            # Clone optimised null-token from "next" step
            null_token = null_token.detach().clone().requires_grad_(True)
            # print(_t, null_token.mean())

            # CTA target
            xtp1 = xts[_t + 1].detach().clone().requires_grad_(False)  # (False)

            # Initialise optimiser
            lr_scale_factor = 1. - _t / (self.model.sampler.pred_timesteps * 2)
            lr = self.learning_rate * lr_scale_factor
            optim = torch.optim.Adam([null_token], lr=lr)
            torch.autograd.set_detect_anomaly(True)

            # Eps prediction
            t = torch.ones(1, dtype=torch.long, device="cuda") * _t

            # CTA
            for _ in range(self.cta_steps):
                # Guided noise prediction
                e_uncond = self.model.model.unet(xt, t, null_token)
                e_cond = self.model.model.unet(xt, t, cond)
                e = e_uncond + self.guidance_scale * (e_cond - e_uncond)
    
                # Prev step
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

                # Loss
                loss = F.mse_loss(xtp1, xt).unsqueeze(0)

                # Optimise null token
                optim.zero_grad()
                loss.backward(retain_graph=False)
                optim.step()

                # Early stopping
                if loss < 1e-5:
                    break

                xt.detach_()

            # Gather optimised null tokens
            optimised_null_token = null_token.detach().clone()
            optimised_null_tokens.append(optimised_null_token)

            # Prev step
            with torch.no_grad():
                # Guided noise prediction with optimised token
                e_uncond = self.model.model.unet(xt, t, optimised_null_token)
                e_cond = self.model.model.unet(xt, t, cond)
                e = e_uncond + self.guidance_scale * (e_cond - e_uncond)

                # CO optimised prev step
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

        assert len(optimised_null_tokens) == self.model.sampler.pred_timesteps
        return optimised_null_tokens

    @torch.no_grad()
    def _guided_ddim(self, u, nts, cond):
        xt = u.clone()
        for _t in tqdm(reversed(range(self.model.sampler.pred_timesteps)), desc='DDIM'):
            t = torch.ones(1, dtype=torch.long, device="cuda") * _t
            e_uncond = self.model.model.unet(xt, t, nts[_t])
            e_cond = self.model.model.unet(xt, t, cond)
            e = e_uncond + self.guidance_scale * (e_cond - e_uncond)
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
        image, cond = obs["image"], obs["metadata"].float()
        assert image.shape[0] == cond.shape[0] == 1
        noise = {}
        if self.model.model.use_semantic_encoder:
            if self.model.model.gaussian_encoder:
                z, _ = self.model.model.encoder(image)
            else:
                z = self.model.model.encoder(image)
            noise["z"] = z
            cond = torch.cat([z, cond], dim=1)
        xts = self._ddim_inversion(image, cond)
        noise["u"] = xts[-1]
        optimised_null_tokens = self._fit(xts, cond)
        noise["n"] = optimised_null_tokens
        # recon = self._guided_ddim(noise["u"], noise["n"], cond)
        return noise

    @torch.no_grad()
    def predict(self, noise, cond):
        cond = cond["metadata"].float()
        assert noise["u"].shape[0] == cond.shape[0] == 1
        if "z" in noise:
            cond = torch.cat([noise["z"], cond], dim=1)
        image = self._guided_ddim(noise["u"], noise["n"], cond)
        return {"image": image}
