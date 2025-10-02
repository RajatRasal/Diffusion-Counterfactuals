"""
The codes are modified.

Link:
    - [Trainer] https://github.com/Megvii-BaseDetection/YOLOX/
      blob/a5bb5ab12a61b8a25a5c3c11ae6f06397eb9b296/yolox/core/trainer.py#L36-L382
"""
from pathlib import Path
from time import time

import torch
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .models.loss import SimpleLoss
from .utils import Meter, TimestepSampler, get_betas, seed_everything, training_reproducibility_cudnn
from .utils.ema import EMA


class Trainer:

    def __init__(self, model, cfg, output_dir, train_dataset):
        self.model = model
        self.cfg = cfg
        self.output_dir = output_dir
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(self.train_dataset, **self.cfg['train']['dataloader'])

        self.device = self.cfg['general']['device']
        self.model.to(self.device)

        self.ckpt_dir = Path(self.output_dir / 'ckpt')
        self.ckpt_dir.mkdir(exist_ok=True)
        logger.info(f'Checkpoints are saved in {self.ckpt_dir}')

        self.tblogger = SummaryWriter(self.output_dir / 'tensorboard')
        logger.info(f'Create a new event in {self.output_dir / "tensorboard"}')

        seed_everything(cfg['general']['seed'])
        training_reproducibility_cudnn()

        self.use_attr = cfg['model']['conditioning']['use_attr']

        self.log_interval = cfg['train']['log_interval']
        self.save_interval = cfg['train']['save_interval']
        logger.info(f'Output a log for every {self.log_interval} iteration')
        logger.info(f'Save checkpoint every {self.save_interval} epoch')

        self.optimizer = self.get_optimizer()
        self.criterion = SimpleLoss(self.cfg)

        self.fp16 = cfg['train']['fp16']
        self.grad_accum_steps = cfg['train']['grad_accum_steps']

        self.clip_grad_norm = cfg['train']['clip_grad_norm']

        self.num_timesteps = cfg['model']['timesteps']['num']
        self.timestep_sampler = TimestepSampler(cfg)

        self.betas = get_betas(cfg)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = self.alphas.cumprod(dim=0)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)

        self.use_ema = cfg['train']['ema']['use_ema']
        if self.use_ema:
            self.ema = EMA(
                model=self.model,
                beta=cfg['train']['ema']['decay'],
                update_every=cfg['train']['ema']['update_every_steps'],
                update_after_step=cfg['train']['ema']['skip_steps'],
            )

    def get_optimizer(self):
        optimizer_cfg = self.cfg['train']['optimizer']
        optimizer_cls = getattr(torch.optim, optimizer_cfg['name'])
        optimizer = optimizer_cls(self.model.parameters(), **optimizer_cfg['params'])
        logger.info(f'Use {optimizer_cfg["name"]} optimizer')
        return optimizer

    def train(self):
        self.before_train()
        self.train_in_epoch()
        self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.cfg['train']['epoch']):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter, batch in enumerate(self.train_loader):
            self.before_iter()
            self.train_one_iter(batch)
            self.after_iter()

    def train_one_iter(self, batch):
        with torch.cuda.amp.autocast(enabled=self.fp16):
            x0, y = batch
            x0 = x0.to(self.device)
            y = y.to(self.device) if self.use_attr else None

            batch_size = x0.shape[0]
            t = self.timestep_sampler.sample(batch_size)

            noise = torch.randn_like(x0, device=self.device)
            alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1.0 - alpha_t) * noise

            out, z, mean, logvar = self.model(x0, xt, t.float(), y)
            loss, recon, kl = self.criterion(out, noise, mean, logvar)
            loss /= self.grad_accum_steps

        self.scaler.scale(loss).backward()
        self.train_loss_meter.update(loss.item())
        self.recon_loss_meter.update(recon.item())
        self.kl_loss_meter.update(0 if kl is None else kl.item())

        if (self.iter + 1) % self.grad_accum_steps == 0:
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

    def before_train(self):
        self.train_loss_meter = Meter()
        self.recon_loss_meter = Meter()
        self.kl_loss_meter = Meter()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        logger.info('Training start ...')

    def after_train(self):
        logger.info('Training done')

    def before_epoch(self):
        self.model.train()
        self.epoch_start_time = time()
        logger.info(f'---> Start train epoch {self.epoch + 1}')

    def after_epoch(self):
        self.save_ckpt(name='last_ckpt.pth')
        epoch_elapsed_time = time() - self.epoch_start_time
        logger.info(f'Epoch {self.epoch + 1} done. ({epoch_elapsed_time:.1f} sec)')

        if (self.epoch + 1) % self.save_interval == 0:
            self.save_ckpt(name=f'epoch_{self.epoch + 1}_ckpt.pth')

    def before_iter(self):
        pass

    def after_iter(self):
        if (self.iter + 1) % self.log_interval == 0:
            step = self.epoch * len(self.train_loader) + self.iter + 1
            logger.info(
                'step: {}, epoch: {}/{}, iter: {}/{}, loss: {:.3f}, recon: {:.3f}, kl: {:.3f}'.format(
                    step,
                    self.epoch + 1, self.cfg['train']['epoch'],
                    self.iter + 1, len(self.train_loader),
                    self.train_loss_meter.latest,
                    self.recon_loss_meter.latest,
                    self.kl_loss_meter.latest,
                )
            )
            self.tblogger.add_scalar('train_loss', self.train_loss_meter.latest, step)
            self.tblogger.add_scalar('recon_loss', self.recon_loss_meter.latest, step)
            self.tblogger.add_scalar('kl_loss', self.kl_loss_meter.latest, step)
            self.train_loss_meter.reset()
            self.recon_loss_meter.reset()
            self.kl_loss_meter.reset()

        if self.use_ema:
            self.ema.update()

    def save_ckpt(self, name):
        logger.info(f'Saving checkpoint to {self.ckpt_dir / name}')
        state = {
            'epoch': self.epoch + 1,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if self.use_ema:
            state['ema'] = self.ema.state_dict()
        torch.save(state, self.ckpt_dir / name)
