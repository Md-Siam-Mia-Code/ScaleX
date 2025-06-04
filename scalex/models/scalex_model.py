# scalex/models/scalex_model.py (or gfpgan_model.py)
import math
import os.path as osp
import torch
from torch import nn, optim, Tensor
from torch.nn import functional as F
import torch.utils.data as torch_data # <<< CORRECTED IMPORT for DataLoader
from torchvision.ops import roi_align
from collections import OrderedDict
from tqdm import tqdm
from typing import Dict, Any, List, Optional, Tuple, Union

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.losses.gan_loss import r1_penalty
from basicsr.metrics import calculate_metric
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from torch.utils.tensorboard import SummaryWriter as TensorboardLoggerType


@MODEL_REGISTRY.register()
class ScaleXModel(BaseModel): # Renamed from GFPGANModel
    """The ScaleX model (formerly GFPGAN) for real-world blind face restoration."""

    def __init__(self, opt: Dict[str, Any]):
        super().__init__(opt) # Python 3 super()
        self.idx: int = 0  # For saving intermediate data during debugging

        # Define generator network (net_g)
        self.net_g: nn.Module = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # Load pretrained generator if specified
        load_path_g: Optional[str] = self.opt['path'].get('pretrain_network_g')
        if load_path_g is not None:
            param_key_g: str = self.opt['path'].get('param_key_g', 'params')
            strict_load_g: bool = self.opt['path'].get('strict_load_g', True)
            self.load_network(self.net_g, load_path_g, strict_load_g, param_key_g)

        self.log_size: int = int(math.log2(self.opt['network_g']['out_size']))

        # Initialize attributes that will be defined in init_training_settings
        self.net_d: Optional[nn.Module] = None
        self.net_g_ema: Optional[nn.Module] = None
        self.use_facial_disc: bool = False
        self.net_d_left_eye: Optional[nn.Module] = None
        self.net_d_right_eye: Optional[nn.Module] = None
        self.net_d_mouth: Optional[nn.Module] = None
        self.cri_component: Optional[nn.Module] = None
        self.cri_pix: Optional[nn.Module] = None
        self.cri_perceptual: Optional[nn.Module] = None
        self.cri_l1: Optional[nn.Module] = None
        self.cri_gan: Optional[nn.Module] = None
        self.use_identity: bool = False
        self.network_identity: Optional[nn.Module] = None
        self.optimizer_g: Optional[optim.Optimizer] = None
        self.optimizer_d: Optional[optim.Optimizer] = None
        self.optimizer_d_left_eye: Optional[optim.Optimizer] = None
        self.optimizer_d_right_eye: Optional[optim.Optimizer] = None
        self.optimizer_d_mouth: Optional[optim.Optimizer] = None
        
        self.lq: Optional[Tensor] = None
        self.gt: Optional[Tensor] = None
        self.output: Optional[Tensor] = None
        self.loc_left_eyes: Optional[Tensor] = None
        self.loc_right_eyes: Optional[Tensor] = None
        self.loc_mouths: Optional[Tensor] = None
        self.left_eyes_gt: Optional[Tensor] = None
        self.right_eyes_gt: Optional[Tensor] = None
        self.mouths_gt: Optional[Tensor] = None
        self.left_eyes: Optional[Tensor] = None
        self.right_eyes: Optional[Tensor] = None
        self.mouths: Optional[Tensor] = None

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self) -> None:
        train_opt: Dict[str, Any] = self.opt['train']

        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)
        load_path_d: Optional[str] = self.opt['path'].get('pretrain_network_d')
        if load_path_d is not None:
            strict_load_d: bool = self.opt['path'].get('strict_load_d', True)
            self.load_network(self.net_d, load_path_d, strict_load_d)

        self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
        load_path_g_ema: Optional[str] = self.opt['path'].get('pretrain_network_g')
        if load_path_g_ema is not None:
            strict_load_g_ema: bool = self.opt['path'].get('strict_load_g', True)
            self.load_network(self.net_g_ema, load_path_g_ema, strict_load_g_ema, 'params_ema')
        else:
            self.model_ema(0) 

        self.net_g.train()
        if self.net_d: self.net_d.train()
        if self.net_g_ema: self.net_g_ema.eval()

        self.use_facial_disc = ('network_d_left_eye' in self.opt and
                                'network_d_right_eye' in self.opt and
                                'network_d_mouth' in self.opt)

        if self.use_facial_disc:
            component_nets_config = [
                ('network_d_left_eye', 'pretrain_network_d_left_eye', 'net_d_left_eye'),
                ('network_d_right_eye', 'pretrain_network_d_right_eye', 'net_d_right_eye'),
                ('network_d_mouth', 'pretrain_network_d_mouth', 'net_d_mouth'),
            ]
            for net_key, pretrain_key, attr_name in component_nets_config:
                net_module = build_network(self.opt[net_key]) # Use a different variable name
                net_module = self.model_to_device(net_module)
                self.print_network(net_module)
                setattr(self, attr_name, net_module)
                
                load_path_comp: Optional[str] = self.opt['path'].get(pretrain_key)
                if load_path_comp:
                    self.load_network(net_module, load_path_comp, True, 'params')
                net_module.train()
            
            self.cri_component = build_loss(train_opt['gan_component_opt']).to(self.device)

        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        
        l1_opt = train_opt.get('L1_opt')
        if l1_opt:
            self.cri_l1 = build_loss(l1_opt).to(self.device)
        else:
            self.cri_l1 = nn.L1Loss().to(self.device)
            logger = get_root_logger()
            logger.warning("L1_opt not found in training options. Using default nn.L1Loss().")

        self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        self.use_identity = 'network_identity' in self.opt
        if self.use_identity:
            self.network_identity = build_network(self.opt['network_identity'])
            self.network_identity = self.model_to_device(self.network_identity)
            self.print_network(self.network_identity)
            load_path_identity: Optional[str] = self.opt['path'].get('pretrain_network_identity')
            if load_path_identity:
                self.load_network(self.network_identity, load_path_identity, True, param_key=None)
            if self.network_identity is not None: # Ensure it's not None before eval/param manipulation
                self.network_identity.eval()
                for param in self.network_identity.parameters():
                    param.requires_grad = False

        self.r1_reg_weight: float = train_opt['r1_reg_weight']
        self.net_d_iters: int = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters: int = train_opt.get('net_d_init_iters', 0)
        self.net_d_reg_every: int = train_opt['net_d_reg_every']

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self) -> None:
        train_opt: Dict[str, Any] = self.opt['train']
        g_optim_opt = train_opt['optim_g'].copy() 
        g_optim_type = g_optim_opt.pop('type', 'Adam') 
        g_lr_base = g_optim_opt.pop('lr', 1e-4)
        
        net_g_reg_ratio = 1.0 
        g_lr_final = g_lr_base * net_g_reg_ratio
        g_betas = (0.0**net_g_reg_ratio, 0.99**net_g_reg_ratio)
        
        self.optimizer_g = self.get_optimizer(
            g_optim_type, self.net_g.parameters(), g_lr_final, betas=g_betas, **g_optim_opt
        )
        self.optimizers.append(self.optimizer_g)

        if self.net_d:
            d_optim_opt = train_opt['optim_d'].copy()
            d_optim_type = d_optim_opt.pop('type', 'Adam')
            d_lr_base = d_optim_opt.pop('lr', 1e-4)

            net_d_reg_ratio = self.net_d_reg_every / (self.net_d_reg_every + 1.0)
            d_lr_final = d_lr_base * net_d_reg_ratio
            d_betas = (0.0**net_d_reg_ratio, 0.99**net_d_reg_ratio)
            
            self.optimizer_d = self.get_optimizer(
                d_optim_type, self.net_d.parameters(), d_lr_final, betas=d_betas, **d_optim_opt
            )
            self.optimizers.append(self.optimizer_d)

        if self.use_facial_disc and 'optim_component' in train_opt:
            comp_optim_opt = train_opt['optim_component'].copy()
            comp_optim_type = comp_optim_opt.pop('type', 'Adam')
            comp_lr = comp_optim_opt.pop('lr', 1e-4)
            comp_betas = comp_optim_opt.pop('betas', (0.9, 0.99)) 

            if self.net_d_left_eye:
                self.optimizer_d_left_eye = self.get_optimizer(
                    comp_optim_type, self.net_d_left_eye.parameters(), comp_lr, betas=comp_betas, **comp_optim_opt)
                self.optimizers.append(self.optimizer_d_left_eye)
            if self.net_d_right_eye:
                self.optimizer_d_right_eye = self.get_optimizer(
                    comp_optim_type, self.net_d_right_eye.parameters(), comp_lr, betas=comp_betas, **comp_optim_opt)
                self.optimizers.append(self.optimizer_d_right_eye)
            if self.net_d_mouth:
                self.optimizer_d_mouth = self.get_optimizer(
                    comp_optim_type, self.net_d_mouth.parameters(), comp_lr, betas=comp_betas, **comp_optim_opt)
                self.optimizers.append(self.optimizer_d_mouth)

    def feed_data(self, data: Dict[str, Any]) -> None:
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        # crop_components is an opt key for the dataset, not directly in model opt
        # Assuming this logic relies on the dataset providing these keys conditionally
        if 'loc_left_eye' in data: 
            self.loc_left_eyes = data['loc_left_eye'].to(self.device)
            self.loc_right_eyes = data['loc_right_eye'].to(self.device)
            self.loc_mouths = data['loc_mouth'].to(self.device)
        else: 
            self.loc_left_eyes = None
            self.loc_right_eyes = None
            self.loc_mouths = None

    def construct_img_pyramid(self) -> List[Tensor]:
        if self.gt is None:
            raise RuntimeError("self.gt is None. Call feed_data with GT image first.")
        
        pyramid_gt: List[Tensor] = [self.gt]
        down_img: Tensor = self.gt
        num_downsamples = self.log_size - 3 
        for _ in range(num_downsamples):
            down_img = F.interpolate(down_img, scale_factor=0.5, mode='bilinear', align_corners=False)
            pyramid_gt.insert(0, down_img) 
        return pyramid_gt

    def get_roi_regions(self, eye_out_size: int = 80, mouth_out_size: int = 120) -> None:
        if not all([self.loc_left_eyes is not None, self.loc_right_eyes is not None, self.loc_mouths is not None,
                    self.gt is not None, self.output is not None]):
            logger = get_root_logger()
            logger.warning("Cannot get ROI regions: Missing location data, GT, or output tensor.")
            self.left_eyes_gt, self.right_eyes_gt, self.mouths_gt = None, None, None # Ensure these are None
            self.left_eyes, self.right_eyes, self.mouths = None, None, None        # if we return early
            return

        face_ratio: float = self.opt['network_g']['out_size'] / 512.0
        eye_out_size_scaled: int = int(eye_out_size * face_ratio)
        mouth_out_size_scaled: int = int(mouth_out_size * face_ratio)

        rois_eyes_list: List[Tensor] = []
        rois_mouths_list: List[Tensor] = []
        
        batch_size = self.loc_left_eyes.size(0)
        for b_idx in range(batch_size):
            batch_indices_eye = self.loc_left_eyes.new_full((2, 1), float(b_idx)) 
            batch_indices_mouth = self.loc_left_eyes.new_full((1, 1), float(b_idx))
            
            bboxes_eye = torch.stack([self.loc_left_eyes[b_idx, :], self.loc_right_eyes[b_idx, :]], dim=0)
            rois_eye_batch = torch.cat([batch_indices_eye, bboxes_eye], dim=-1) 
            rois_eyes_list.append(rois_eye_batch)

            bboxes_mouth = self.loc_mouths[b_idx:b_idx + 1, :] 
            rois_mouth_batch = torch.cat([batch_indices_mouth, bboxes_mouth], dim=-1) 
            rois_mouths_list.append(rois_mouth_batch)

        rois_eyes_tensor: Tensor = torch.cat(rois_eyes_list, dim=0)
        rois_mouths_tensor: Tensor = torch.cat(rois_mouths_list, dim=0)

        all_eyes_gt = roi_align(self.gt, boxes=rois_eyes_tensor, output_size=eye_out_size_scaled) * face_ratio
        self.left_eyes_gt = all_eyes_gt[0::2, :, :, :] 
        self.right_eyes_gt = all_eyes_gt[1::2, :, :, :] 
        self.mouths_gt = roi_align(self.gt, boxes=rois_mouths_tensor, output_size=mouth_out_size_scaled) * face_ratio

        all_eyes_output = roi_align(self.output, boxes=rois_eyes_tensor, output_size=eye_out_size_scaled) * face_ratio
        self.left_eyes = all_eyes_output[0::2, :, :, :]
        self.right_eyes = all_eyes_output[1::2, :, :, :]
        self.mouths = roi_align(self.output, boxes=rois_mouths_tensor, output_size=mouth_out_size_scaled) * face_ratio

    def _gram_mat(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        features = x.view(n, c, h * w)
        features_t = features.transpose(1, 2) 
        gram = torch.bmm(features, features_t) / (c * h * w) 
        return gram

    def gray_resize_for_identity(self, x: Tensor, target_size: int = 128) -> Tensor:
        out_gray: Tensor = (0.2989 * x[:, 0, :, :] +
                            0.5870 * x[:, 1, :, :] +
                            0.1140 * x[:, 2, :, :])
        out_gray = out_gray.unsqueeze(1) 
        out_gray = F.interpolate(out_gray, (target_size, target_size), mode='bilinear', align_corners=False)
        return out_gray

    def optimize_parameters(self, current_iter: int) -> None:
        if self.net_d is None or self.optimizer_g is None or self.optimizer_d is None or self.lq is None:
            # Log or raise error if essential components for training are missing
            logger = get_root_logger()
            logger.error("Training cannot proceed: net_d, optimizers, or lq data is None.")
            return

        # --- Optimize Generator (net_g) ---
        self.net_d.requires_grad_(False)
        if self.use_facial_disc:
            if self.net_d_left_eye: self.net_d_left_eye.requires_grad_(False)
            if self.net_d_right_eye: self.net_d_right_eye.requires_grad_(False)
            if self.net_d_mouth: self.net_d_mouth.requires_grad_(False)
        
        self.optimizer_g.zero_grad()

        pyramid_loss_weight: float = self.opt['train'].get('pyramid_loss_weight', 0.0)
        if pyramid_loss_weight > 0 and current_iter > self.opt['train'].get('remove_pyramid_loss', float('inf')):
            pyramid_loss_weight = 1e-12

        out_rgbs: Optional[List[Tensor]] = None
        # Ensure net_g has a forward method compatible with these arguments
        if hasattr(self.net_g, 'forward') and callable(getattr(self.net_g, 'forward')):
            if pyramid_loss_weight > 0:
                self.output, out_rgbs = self.net_g(self.lq, return_rgb=True)
            else:
                self.output, _ = self.net_g(self.lq, return_rgb=False)
        else:
            raise AttributeError("self.net_g does not have a callable 'forward' method as expected.")


        if self.use_facial_disc:
            self.get_roi_regions()

        l_g_total: Tensor = torch.tensor(0.0, device=self.device)
        loss_dict: OrderedDict[str, Tensor] = OrderedDict()

        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            if self.cri_pix and self.gt is not None and self.output is not None:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix

            if pyramid_loss_weight > 0 and out_rgbs is not None and self.cri_l1 is not None:
                pyramid_gt = self.construct_img_pyramid()
                for i, rgb_out_level in enumerate(out_rgbs):
                    if i < len(pyramid_gt):
                        l_pyramid = self.cri_l1(rgb_out_level, pyramid_gt[i]) * pyramid_loss_weight
                        l_g_total += l_pyramid
                        loss_dict[f'l_pyramid_{2**(i+3)}'] = l_pyramid

            if self.cri_perceptual and self.gt is not None and self.output is not None:
                percep_out = self.cri_perceptual(self.output, self.gt)
                if isinstance(percep_out, tuple) and len(percep_out) == 2:
                    l_g_percep, l_g_style = percep_out
                    if l_g_percep is not None:
                        l_g_total += l_g_percep
                        loss_dict['l_g_percep'] = l_g_percep
                    if l_g_style is not None:
                        l_g_total += l_g_style
                        loss_dict['l_g_style'] = l_g_style
                elif isinstance(percep_out, Tensor): # If only perceptual loss is returned
                    l_g_percep = percep_out
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep

            if self.output is not None:
                fake_g_pred: Tensor = self.net_d(self.output)
                l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan'] = l_g_gan

            if self.use_facial_disc and self.cri_component is not None:
                comp_loss_items = [
                    ('left_eye', self.left_eyes, self.left_eyes_gt, self.net_d_left_eye),
                    ('right_eye', self.right_eyes, self.right_eyes_gt, self.net_d_right_eye),
                    ('mouth', self.mouths, self.mouths_gt, self.net_d_mouth),
                ]
                comp_style_total_loss: Tensor = torch.tensor(0.0, device=self.device)

                for name, fake_comp, real_comp_gt, net_d_comp_module in comp_loss_items:
                    if fake_comp is None or net_d_comp_module is None: continue
                    
                    fake_comp_pred_tuple = net_d_comp_module(fake_comp, return_feats=True)
                    fake_comp_pred = fake_comp_pred_tuple[0]
                    fake_comp_feats = fake_comp_pred_tuple[1] if len(fake_comp_pred_tuple) > 1 else None

                    l_g_comp_gan = self.cri_component(fake_comp_pred, True, is_disc=False)
                    l_g_total += l_g_comp_gan
                    loss_dict[f'l_g_gan_{name}'] = l_g_comp_gan

                    comp_style_weight = self.opt['train'].get('comp_style_weight', 0.0)
                    if comp_style_weight > 0 and real_comp_gt is not None and self.cri_l1 is not None and fake_comp_feats:
                        real_comp_pred_tuple = net_d_comp_module(real_comp_gt, return_feats=True)
                        real_comp_feats = real_comp_pred_tuple[1] if len(real_comp_pred_tuple) > 1 else None
                        
                        if real_comp_feats and len(fake_comp_feats) == len(real_comp_feats):
                            for feat_f, feat_r in zip(fake_comp_feats, real_comp_feats):
                                comp_style_total_loss += self.cri_l1(self._gram_mat(feat_f), self._gram_mat(feat_r.detach()))
                
                if comp_style_weight > 0 and comp_style_total_loss.nelement() > 0 : # Ensure loss is not empty
                    comp_style_final = comp_style_total_loss * comp_style_weight
                    l_g_total += comp_style_final
                    loss_dict['l_g_comp_style'] = comp_style_final
            
            if self.use_identity and self.network_identity is not None and self.cri_l1 is not None and self.gt is not None and self.output is not None:
                identity_weight: float = self.opt['train'].get('identity_weight', 0.0)
                if identity_weight > 0:
                    out_gray = self.gray_resize_for_identity(self.output)
                    gt_gray = self.gray_resize_for_identity(self.gt)
                    
                    identity_out = self.network_identity(out_gray)
                    identity_gt_feat = self.network_identity(gt_gray).detach()
                    
                    l_identity = self.cri_l1(identity_out, identity_gt_feat) * identity_weight
                    l_g_total += l_identity
                    loss_dict['l_identity'] = l_identity
            
            if l_g_total.nelement() > 0 and l_g_total.requires_grad: # Ensure total loss is valid
                l_g_total.backward()
                self.optimizer_g.step()

        if self.net_g_ema is not None:
            self.model_ema(decay=0.5**(32 / (10 * 1000)))

        # --- Optimize Discriminators ---
        self.net_d.requires_grad_(True)
        self.optimizer_d.zero_grad()

        if self.use_facial_disc:
            comp_nets_optimizers_d = [
                (self.net_d_left_eye, self.optimizer_d_left_eye),
                (self.net_d_right_eye, self.optimizer_d_right_eye),
                (self.net_d_mouth, self.optimizer_d_mouth)
            ]
            for net_module, opt_comp in comp_nets_optimizers_d:
                if net_module and opt_comp:
                    net_module.requires_grad_(True)
                    opt_comp.zero_grad()
        
        if self.output is not None and self.gt is not None:
            fake_d_pred = self.net_d(self.output.detach())
            real_d_pred = self.net_d(self.gt)
            
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            l_d = l_d_real + l_d_fake
            loss_dict['l_d'] = l_d
            loss_dict['real_score'] = real_d_pred.detach().mean()
            loss_dict['fake_score'] = fake_d_pred.detach().mean()
            l_d.backward()

            if current_iter % self.net_d_reg_every == 0:
                self.gt.requires_grad = True
                real_pred_for_r1 = self.net_d(self.gt)
                l_d_r1 = r1_penalty(real_pred_for_r1, self.gt)
                l_d_r1_scaled = (self.r1_reg_weight / 2.0 * l_d_r1 * self.net_d_reg_every) + (0.0 * real_pred_for_r1.mean())
                loss_dict['l_d_r1'] = l_d_r1_scaled.detach()
                l_d_r1_scaled.backward()
                self.gt.requires_grad = False

            self.optimizer_d.step()

        if self.use_facial_disc and self.cri_component is not None:
            for name, fake_comp, real_comp_gt, net_d_comp_module, opt_comp in comp_nets_optimizers_d: # Use the redefined list
                if fake_comp is None or real_comp_gt is None or net_d_comp_module is None or opt_comp is None:
                    continue
                
                fake_comp_pred_d_tuple = net_d_comp_module(fake_comp.detach()) # No return_feats needed for D loss typically
                fake_comp_pred_d = fake_comp_pred_d_tuple[0] if isinstance(fake_comp_pred_d_tuple, tuple) else fake_comp_pred_d_tuple
                
                real_comp_pred_d_tuple = net_d_comp_module(real_comp_gt)
                real_comp_pred_d = real_comp_pred_d_tuple[0] if isinstance(real_comp_pred_d_tuple, tuple) else real_comp_pred_d_tuple

                l_d_comp_real = self.cri_component(real_comp_pred_d, True, is_disc=True)
                l_d_comp_fake = self.cri_component(fake_comp_pred_d, False, is_disc=True)
                l_d_comp = l_d_comp_real + l_d_comp_fake
                loss_dict[f'l_d_{name}'] = l_d_comp
                l_d_comp.backward()
                opt_comp.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self) -> None:
        with torch.no_grad():
            if self.net_g_ema is not None:
                self.net_g_ema.eval()
                if self.lq is not None:
                    # Assuming net_g_ema.forward signature matches net_g
                    self.output, _ = self.net_g_ema(self.lq, return_rgb=False) 
            elif self.net_g is not None:
                logger = get_root_logger()
                logger.warning('net_g_ema not available, using net_g for testing.')
                self.net_g.eval()
                if self.lq is not None:
                    self.output, _ = self.net_g(self.lq, return_rgb=False)
                self.net_g.train()
            else:
                raise RuntimeError("No generator network available for testing (net_g_ema or net_g).")

    def dist_validation(self, dataloader: torch_data.DataLoader, current_iter: int,
                        tb_logger: Optional[TensorboardLoggerType], save_img: bool) -> None:
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader: torch_data.DataLoader, current_iter: int,
                           tb_logger: Optional[TensorboardLoggerType], save_img: bool) -> None:
        dataset_opt = getattr(dataloader.dataset, 'opt', None)
        if dataset_opt is None or 'name' not in dataset_opt:
            # Fallback if opt or name is not directly on dataset object
            dataset_name: str = self.opt['val'].get('name', 'validation_dataset')
        else:
            dataset_name: str = dataset_opt['name']

        with_metrics: bool = self.opt['val'].get('metrics') is not None
        use_pbar: bool = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results: Dict[str, float] = {
                    metric: 0.0 for metric in self.opt['val']['metrics'].keys()
                }
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = {metric: 0.0 for metric in self.metric_results}

        metric_data: Dict[str, np.ndarray] = {}
        prog_bar: Optional[tqdm] = None
        if use_pbar:
            prog_bar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name_no_ext: str = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            if self.output is None: continue

            sr_img_np: np.ndarray = tensor2img(self.output.detach().cpu(), min_max=(-1, 1))
            metric_data['img'] = sr_img_np
            if self.gt is not None:
                gt_img_np: np.ndarray = tensor2img(self.gt.detach().cpu(), min_max=(-1, 1))
                metric_data['img2'] = gt_img_np
            
            del self.lq; self.lq = None
            del self.output; self.output = None
            if self.gt is not None: del self.gt; self.gt = None
            torch.cuda.empty_cache()

            if save_img:
                save_folder = osp.join(self.opt['path']['visualization'], dataset_name)
                if self.opt['is_train']:
                     # During training, save under a subfolder named by iter or img_name
                    save_folder_iter = osp.join(self.opt['path']['visualization'], f'iter_{current_iter}', dataset_name)
                    os.makedirs(save_folder_iter, exist_ok=True)
                    save_img_path = osp.join(save_folder_iter, f'{img_name_no_ext}.png')
                else:
                    os.makedirs(save_folder, exist_ok=True)
                    suffix = self.opt['val'].get('suffix', self.opt.get('name', 'ScaleX'))
                    save_img_path = osp.join(save_folder, f'{img_name_no_ext}_{suffix}.png')
                imwrite(sr_img_np, save_img_path)

            if with_metrics:
                for metric_name, metric_opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[metric_name] += calculate_metric(metric_data, metric_opt_)
            
            if prog_bar:
                prog_bar.update(1)
                prog_bar.set_description(f'Test {img_name_no_ext}')
        
        if prog_bar:
            prog_bar.close()

        if with_metrics:
            num_images = len(dataloader)
            if num_images > 0:
                for metric_key in self.metric_results:
                    self.metric_results[metric_key] /= num_images
                
                # Example: use first metric for best tracking or a specifically named one
                first_metric_key = list(self.opt['val']['metrics'].keys())[0]
                self._update_best_metric_result(dataset_name, first_metric_key, self.metric_results[first_metric_key], current_iter)
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            else:
                logger = get_root_logger()
                logger.warning(f"No images processed during validation for {dataset_name}. Metrics not calculated.")


    def _log_validation_metric_values(self, current_iter: int, dataset_name: str,
                                      tb_logger: Optional[TensorboardLoggerType]) -> None:
        if not hasattr(self, 'metric_results') or not self.metric_results: return
        
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results') and \
               dataset_name in self.best_metric_results and \
               metric in self.best_metric_results[dataset_name]:
                best_val = self.best_metric_results[dataset_name][metric]["val"]
                best_iter = self.best_metric_results[dataset_name][metric]["iter"]
                log_str += f'\tBest: {best_val:.4f} @ {best_iter} iter'
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def save(self, epoch: int, current_iter: int) -> None:
        if self.net_g_ema is not None:
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter, param_key='params')

        if self.net_d: self.save_network(self.net_d, 'net_d', current_iter)
        
        if self.use_facial_disc:
            if self.net_d_left_eye: self.save_network(self.net_d_left_eye, 'net_d_left_eye', current_iter)
            if self.net_d_right_eye: self.save_network(self.net_d_right_eye, 'net_d_right_eye', current_iter)
            if self.net_d_mouth: self.save_network(self.net_d_mouth, 'net_d_mouth', current_iter)
        
        self.save_training_state(epoch, current_iter)