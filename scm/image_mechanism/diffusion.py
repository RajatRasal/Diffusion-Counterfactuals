from typing import Dict, List, Literal, Optional

import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from counterfactuals.scm.image_mechanism.base import ImageMechanism
from ddpm.training.mnist_conditional import GuidedDiffusionLightningModule
from models.ddpm.callbacks import DiffusionCallback
from models.ddpm.null_token_optimisation import NullTokenOptimisation


class DiffusionImageMechanism(ImageMechanism):

    def __init__(
        self,
        model: GuidedDiffusionLightningModule,
        guidance_scale: float,
        timesteps: int,
        inversion_method: Literal["ddim", "cfg", "cfg++"] = "ddim",
        generation_method: Literal["ddim", "cfg", "cfg++"] = "cfg",
        include_d: bool = False,
        deterministic: bool = True,
        inversion_callbacks: Optional[List[DiffusionCallback]] = None,
        generation_callbacks: Optional[List[DiffusionCallback]] = None,
    ):
        self.model = model
        self.diffusion = self.model._get_diffusion()
        self.guidance_scale = guidance_scale
        self.timesteps = timesteps
        self.inversion_method = inversion_method
        self.generation_method = generation_method
        self.include_d = include_d
        self.deterministic = deterministic
        self.inversion_callbacks = inversion_callbacks
        self.generation_callbacks = generation_callbacks
        # self.u_noise = MultivariateNormal() # each value between -1 and 1
        # self.z_noise = MultivariateNormal() # diagonal covariance -1 and 1

    def _sample(
        self,
        u: torch.FloatTensor,
        null_token: torch.FloatTensor,
        cond: Dict,
        guidance_scale: float,
        callbacks: Optional[List[DiffusionCallback]] = None,
    ) -> torch.FloatTensor:
        return self.diffusion.sample(
            xT=u,
            null_token=null_token,
            conditions=cond,
            method=self.generation_method,
            guidance_scale=guidance_scale,
            deterministic=True,
            timesteps=self.timesteps,
            scheduler_step_kwargs={"eta": 0, "use_clipped_model_output": True},
            disable_progress_bar=True,
            callbacks=callbacks,
        )
    
    def abduct(self, cond: Dict) -> Dict:
        if self.inversion_callbacks is not None:
            for c in self.inversion_callbacks:
                c.reset()
        images = cond["image"]
        mu, logvar, reparam = self.model.encode_image(images)
        z = mu if self.deterministic else reparam
        cond = self.model._get_condition(cond, z)
        null_token = self.model._null_token.repeat(images.shape[0], 1)
        u = self.diffusion.ddim_inversion(
            images,
            null_token=null_token,
            conditions=cond,
            timesteps=self.timesteps,
            method=self.inversion_method,
            use_clipping=True,
            disable_progress_bar=True,
            callbacks=self.inversion_callbacks,
        )
        recons = self._sample(u, null_token, cond, 1).clamp(-1, 1)
        noise = {"z": z, "u": u}
        if self.include_d:
            noise["d"] = images - recons
        return noise

    def predict(self, noise: Dict, cond: Dict) -> Dict:
        if self.generation_callbacks is not None:
            for c in self.generation_callbacks:
                c.reset()
        z, u = noise["z"], noise["u"]
        cond = self.model._get_condition(cond, z)
        null_token = self.model._null_token.repeat(u.shape[0], 1)
        cfs = self._sample(
            u, null_token, cond, self.guidance_scale, self.generation_callbacks,
        )
        if self.include_d:
            cfs += noise["d"]
        return {"image": cfs.clamp(-1, 1)}


class NTOImageMechanism(ImageMechanism):

    def __init__(
        self,
        model: GuidedDiffusionLightningModule,
        guidance_scale: float,
        timesteps: int = 50,
        nti_steps: int = 10,
        learning_rate: float = 1e-2,
        deterministic: bool = True,
        seed: int = 0,
        reset_null_token: bool = False,
        include_d: bool = False,
        inversion_callbacks: Optional[List[DiffusionCallback]] = None,
        generation_callbacks: Optional[List[DiffusionCallback]] = None,
    ):
        self.model = model
        self.diffusion = self.model._get_diffusion()
        self.guidance_scale = guidance_scale
        self.timesteps = timesteps
        self.nti_steps = nti_steps
        self.learning_rate = learning_rate
        self.deterministic = deterministic
        self.null_token = self.model._null_token.clone().repeat(1, 1)
        self.seed = seed
        self.reset_null_token = reset_null_token
        self.include_d = include_d
        self.inversion_callbacks = inversion_callbacks
        self.generation_callbacks = generation_callbacks
        self.nto = NullTokenOptimisation(
            self.diffusion,
            self.null_token,
            self.nti_steps,
            self.learning_rate,
            self.guidance_scale,
            self.timesteps,
            self.deterministic,
            {"eta": 0, "use_clipped_model_output": True},
            self.reset_null_token,
        )
    
    def abduct(self, cond: Dict) -> Dict:
        if self.inversion_callbacks is not None:
            for c in self.inversion_callbacks:
                c.reset()
        image = cond["image"]
        _, _, z = self.model.encode_image(image)
        cond = self.model._get_condition(cond, z)
        generator = torch.cuda.manual_seed(self.seed)
        u = self.nto.fit(
            image,
            cond,
            disable_progress_bar=True,
            generator=generator,
            callbacks=self.inversion_callbacks,
        )
        recon = self.nto.generate(
            u,
            cond,
            self.nto.optimised_null_tokens,
            deterministic=True,
            disable_progress_bar=True,
            generator=generator,
        ).clamp(-1, 1)
        noise = {"z": z, "u": u, "null_tokens": self.nto.optimised_null_tokens}
        if self.include_d:
            noise["d"] = image - recon
        return noise

    def predict(self, noise: Dict, cond: Dict) -> Dict:
        if self.generation_callbacks is not None:
            for c in self.generation_callbacks:
                c.reset()
        z, u, null_tokens = noise["z"], noise["u"], noise["null_tokens"]
        cond = self.model._get_condition(cond, noise["z"])
        generator = torch.cuda.manual_seed(self.seed)
        cf = self.nto.generate(
            u,
            cond,
            null_tokens,
            deterministic=True,
            disable_progress_bar=True,
            generator=generator,
            callbacks=self.generation_callbacks,
        )
        if self.include_d:
            cf += noise["d"]
        return {"image": cf.clamp(-1, 1)}
