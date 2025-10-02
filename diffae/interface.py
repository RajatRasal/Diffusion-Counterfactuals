import json
import math
import time
from pathlib import Path

import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader

from .classifier import ClassifierTrainer, LinearClassifier, evaluate_classifier
from .dataset import EmbeddingDataset, get_dataset, get_torchvision_transforms
from .models.model import DiffusionAutoEncoders
from .sampler import Sampler
from .trainer import Trainer
from .utils import get_torchvision_unnormalize
from .utils.ema import EMA


class DiffusionAutoEncodersInterface:
    CFG_DIR = Path('src/models/diffae/diffae/cfg')

    def __init__(self, args, mode):
        """Setting up config, output directory, model, and dataset.

        Args:
            args (dict): A dict of arguments with the following keys,
                mode == 'train':
                    data_name (str): Dataset name.
                    size (int): Image size.
                    expn (str): Experiment name.
                mode in {'test', 'clf_train', 'clf_test', 'infer'}:
                    output (str): Path to output directory.
        """
        assert mode in {'train', 'test', 'clf_train', 'clf_test', 'infer'}
        self.mode = mode
        self.cfg = self._init_config(args)

        if self.mode == 'train':
            self.output_dir = self._init_output_dir(args)
            saved_cfg_file = self.output_dir / 'model.yml'
            with saved_cfg_file.open('w') as fp:
                yaml.safe_dump(self.cfg, fp, sort_keys=False)
        else:
            self.output_dir = Path(args['output'])

        if self.mode.endswith('train'):
            log_path = self.output_dir / f'{self.mode}.txt'
            logger.add(log_path, mode='w')

        model_ckpt_path = self.output_dir / 'ckpt' / args['model_ckpt'] if 'model_ckpt' in args else None
        model_ckpt_path = None if self.mode == 'train' else self.output_dir / 'ckpt' / args['model_ckpt']
        self.model = self._init_model(model_ckpt_path)

        clf_ckpt_path = self.output_dir / 'ckpt' / args['clf_ckpt'] if 'clf_ckpt' in args else None
        self.clf_model = self._init_clf_model(clf_ckpt_path)

        if self.mode in {'test', 'infer'}:
            self.sampler = Sampler(self.model, self.cfg)
            if self.cfg['general']['data_name'] != "embed":
                self.unnormalize = get_torchvision_unnormalize(
                    self.cfg['test']['dataset'][-1]['params']['mean'],
                    self.cfg['test']['dataset'][-1]['params']['std'],
                )

        self._init_dataset()

    def _init_config(self, args):
        """Setting up config dict.
        """
        logger.info('Initializing config...')
        # Load config file.
        if self.mode == 'train':
            cfg_path = self.CFG_DIR / f'{args["size"]}_model.yml'
        else:
            cfg_path = Path(args['output']) / 'model.yml'
        with cfg_path.open('r') as fp:
            cfg = yaml.safe_load(fp)

        # if self.mode == 'train':
        #     cfg['general']['data_name'] = args['data_name']
        if cfg['general']['data_name'] == 'celeba':
            logger.info('For CelebA (not CelabA-HQ) dataset, D2C Crop is applied first.')
            cfg['train']['dataset'].insert(0, {'name': 'D2CCrop', 'params': {}})

        if not torch.cuda.is_available():
            logger.warning('cuda is not available')
            cfg['general']['device'] = 'cpu'

        return cfg

    def _init_output_dir(self, args):
        """Create output directory.
        """
        assert self.mode == 'train'
        logger.info('Initializing output directory...')
        output_root = Path(self.cfg['general']['output_root'])
        output_dir = output_root / time.strftime('%Y%m%d%H%M%S') if args['expn'] is None else output_root / args['expn']
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    def _init_model(self, ckpt_path=None):
        """Setting up DiffusionAutoEncoders module.
        """
        logger.info('Initializing Diffusion Autoencoders...')
        self.cfg["model"]["network"]["unet"]["emb_channels"] = 0
        # TODO: if dataset mnist and class in attributes, add 10 to emb_channels
        if self.cfg["general"]["data_name"] == "mnist" and "class" in self.cfg["general"]["attributes"]:
            self.cfg["model"]["network"]["unet"]["emb_channels"] += 9
        elif self.cfg["general"]["data_name"] == "embed":
            self.cfg["model"]["network"]["unet"]["emb_channels"] = 4
        model = DiffusionAutoEncoders(self.cfg)
        if ckpt_path is not None:
            logger.info('Loading Diff-AE checkpoint...')
            state = torch.load(ckpt_path)
            model.load_state_dict(state['model'], strict=False)
            if 'ema' in state and self.cfg['test']['ema']['use']:
                logger.info('Loading EMA checkpoint...')
                ema = EMA(model)
                ema.load_state_dict(state['ema'], strict=False)
                model = ema.ema_model
        return model

    def _init_clf_model(self, ckpt_path=None):
        """Setting up LinearClassifier module.
        """
        logger.info('Initializing classifier...')
        clf_model = LinearClassifier(self.cfg)
        if ckpt_path is not None:
            logger.info('Loading classifier checkpoint...')
            state = torch.load(ckpt_path)
            clf_model.load_state_dict(state['model'])
        return clf_model

    def _init_dataset(self):
        """Setting up transforms and dataset.
        """
        assert self.mode in {'train', 'test', 'clf_train', 'clf_test', 'infer'}
        logger.info('Initializing dataset...')

        if self.mode in {'train', 'clf_train'}:
            self.transforms = get_torchvision_transforms(self.cfg, 'train')
        else:
            self.transforms = get_torchvision_transforms(self.cfg, 'test')

        data_name = self.cfg['general']['data_name']
        attributes = self.cfg['general'].get('attributes', None)
        if self.mode == 'train':
            self.dataset = get_dataset(name=data_name, split=self.mode, transform=self.transforms, attributes=attributes)
        elif self.mode == 'test':
            self.dataset = get_dataset(name=data_name, split=self.mode, transform=self.transforms, attributes=attributes)
        elif self.mode == 'clf_train':
            image_dataset = get_dataset(name=data_name, split='train', transform=self.transforms, attributes=attributes)
            image_loader = DataLoader(image_dataset, **self.cfg['classifier']['train']['dataloader'])
            self.dataset = EmbeddingDataset(image_loader, self.model.encoder, self.cfg)
        elif self.mode == 'clf_test':
            image_dataset = get_dataset(name=data_name, split='test', transform=self.transforms, attributes=attributes)
            image_loader = DataLoader(image_dataset, **self.cfg['classifier']['test']['dataloader'])
            self.dataset = EmbeddingDataset(image_loader, self.model.encoder, self.cfg)

    def train(self):
        """DiffusionAutoEncoders training.
        """
        assert self.mode == 'train'

        trainer = Trainer(self.model, self.cfg, self.output_dir, self.dataset)
        trainer.train()

    @torch.inference_mode()
    def test(self):
        """DiffusionAutoEncoders evaluation.
        """
        assert self.mode == 'test'

        logger.info('Evaluation start...')
        result = self.sampler.sample_testdata(self.dataset)
        with (self.output_dir / 'test.json').open('w') as fp:
            json.dump(result, fp, indent=4, sort_keys=True)

    def clf_train(self):
        """LinearClassifier training.
        """
        self.clf_model.set_norm_params(self.dataset)
        self.clf_trainer = ClassifierTrainer(self.clf_model, self.cfg, self.output_dir, self.dataset)
        self.clf_trainer.train()

    @torch.inference_mode()
    def clf_test(self):
        """LinearClassifier evaluation.
        """
        logger.info('Classifier evaluation start...')
        result = evaluate_classifier(self.clf_model, self.cfg, self.dataset)
        with (self.output_dir / 'clf_test.json').open('w') as fp:
            json.dump(result, fp, indent=4, sort_keys=False)

    @torch.inference_mode()
    def infer(self, image, xt=None, style_emb=None, y=None):
        """Autoencode a single image.

        Args:
            image: (PIL Image): A single PIL Image.
            style_emb (torch.tensor, optional): A tensor of SemanticEncoder embedding.
                You can perform conditional generation with arbitary embedding by passing this argument.

        Returns:
            result (dict): A result of autoencoding which has the following keys,
                input (numpy.ndarray): A input image array.
                output (numpy.ndarray): A output (autoencoded) image array.
                x0_preds (List[numpy.ndarray]): A list of predicted x0 per timestep.
                xt_preds (List[numpy.ndarray]): A list of predicted xt per timestep.
        """
        assert self.mode == 'infer'
        image = self.transforms(image)
        result = self.sampler.sample_one_image(image, y=y, xt=xt, style_emb=style_emb)

        # Unnormalize and to numpy.ndarray
        for k, v in result.items():
            if isinstance(v, list):
                for i, x in enumerate(result[k]):
                    assert torch.is_tensor(x)
                    result[k][i] = self.unnormalize(x).permute(1, 2, 0).cpu().detach().numpy()
            elif torch.is_tensor(v):
                result[k] = self.unnormalize(v).permute(1, 2, 0).cpu().detach().numpy()

        return result

    @torch.inference_mode()
    def infer_manipulate(self, image, target_ids, s=[0.3]):
        """Attribute manipulation using classifier guidance.

        Args:
            image (PIL Image): A single PIL Image.
            target_id (int): Target attribute id.
            s (float or List[float]): Attribute manipulation parameter(s).

        Returns:
            result (dict): A result of autoencoding which has the following keys,
                input (numpy.ndarray): A input image array.
                output (numpy.ndarray): A output (autoencoded) image array.
                x0_preds (List[numpy.ndarray]): A list of predicted x0 per timestep.
                xt_preds (List[numpy.ndarray]): A list of predicted xt per timestep.
        """
        assert self.mode == 'infer'
        device = self.cfg['general']['device']
        # if isinstance(s, (int, float)):
        #     s = [s]

        clf_ckpt_path = self.output_dir / 'ckpt/clf_last_ckpt.pth'
        self.clf_model = self._init_clf_model(clf_ckpt_path)
        self.clf_model.to(device)
        # direction = torch.nn.functional.normalize(self.clf_model.classifier.weight[target_id][None, :], dim=1)

        x0 = self.transforms(image).unsqueeze(dim=0).to(device)
        xt = self.sampler.encode_stochastic(x0)
        style_emb = self.model.encoder(x0)
        style_emb = self.clf_model.normalize(style_emb)
        transferred_style_emb = style_emb

        # results = []
        for i in range(len(s)):
            direction = torch.nn.functional.normalize(self.clf_model.classifier.weight[target_ids[i]][None, :], dim=1)
            transferred_style_emb += s[i] * math.sqrt(style_emb.shape[1]) * direction

        transferred_style_emb = self.clf_model.unnormalize(transferred_style_emb)
        result = self.infer(image, xt=xt, style_emb=transferred_style_emb)
        #    # result = self.infer(image, y=y, xt=xt, style_emb=transferred_style_emb)
        #    results.append(result)

        return result

    @torch.inference_mode()
    def infer_two_manipulate(self, image, target_id, target_id2, s1=0.3, s2=0.3):
        """Attribute manipulation using classifier guidance.

        Args:
            image (PIL Image): A single PIL Image.
            target_id (int): Target attribute id.
            s (float or List[float]): Attribute manipulation parameter(s).

        Returns:
            result (dict): A result of autoencoding which has the following keys,
                input (numpy.ndarray): A input image array.
                output (numpy.ndarray): A output (autoencoded) image array.
                x0_preds (List[numpy.ndarray]): A list of predicted x0 per timestep.
                xt_preds (List[numpy.ndarray]): A list of predicted xt per timestep.
        """
        assert self.mode == 'infer'
        device = self.cfg['general']['device']
        if isinstance(s, (int, float)):
            s = [s]

        clf_ckpt_path = self.output_dir / 'ckpt/clf_last_ckpt.pth'
        self.clf_model = self._init_clf_model(clf_ckpt_path)
        self.clf_model.to(device)
        direction = torch.nn.functional.normalize(self.clf_model.classifier.weight[target_id1][None, :], dim=1)
        direction2 = torch.nn.functional.normalize(self.clf_model.classifier.weight[target_id2][None, :], dim=1)

        x0 = self.transforms(image).unsqueeze(dim=0).to(device)
        xt = self.sampler.encode_stochastic(x0)
        style_emb = self.model.encoder(x0)
        style_emb = self.clf_model.normalize(style_emb)

        results = []
        for i in range(len(s)):
            transferred_style_emb = style_emb + s[i] * math.sqrt(style_emb.shape[1]) * direction
            transferred_style_emb = self.clf_model.unnormalize(transferred_style_emb)

            result = self.infer(image, xt=xt, style_emb=transferred_style_emb)
            # result = self.infer(image, y=y, xt=xt, style_emb=transferred_style_emb)
            results.append(result)

    @torch.inference_mode()
    def infer_interpolate(self, image_1, image_2, alphas):
        """Interpolation of 2 images.

        Args:
            image_1, image_2 (PIL Image): A single PIL Image.
            alphas (float or List[float]): Interpolation parameter(s).

        Returns:
            result (List[dict]): A List of result which has the following keys,
                output (numpy.ndarray): A output (autoencoded) image array.
                x0_preds (List[numpy.ndarray]): A list of predicted x0 per timestep.
                xt_preds (List[numpy.ndarray]): A list of predicted xt per timestep.
        """
        assert self.mode == 'infer'
        device = self.cfg['general']['device']
        if isinstance(alphas, (int, float)):
            alphas = [alphas]

        image_1 = self.transforms(image_1)
        image_2 = self.transforms(image_2)

        x0_1 = image_1.unsqueeze(dim=0).to(device)
        x0_2 = image_2.unsqueeze(dim=0).to(device)

        xt_1 = self.sampler.encode_stochastic(x0_1)
        xt_2 = self.sampler.encode_stochastic(x0_2)

        style_emb_1 = self.model.encoder(x0_1)
        style_emb_2 = self.model.encoder(x0_2)

        results = []
        for alpha in alphas:
            result = self.sampler.interpolate(xt_1, xt_2, style_emb_1, style_emb_2, alpha)

            # Unnormalize and to numpy.ndarray
            for k, v in result.items():
                if isinstance(v, list):
                    for i, x in enumerate(result[k]):
                        assert torch.is_tensor(x)
                        result[k][i] = self.unnormalize(x).permute(1, 2, 0).cpu().detach().numpy()
                elif torch.is_tensor(v):
                    result[k] = self.unnormalize(v).permute(1, 2, 0).cpu().detach().numpy()

            results.append(result)

        return results
