import cv2
import math
import numpy as np
import os.path as osp
import torch
import torch.utils.data as data
from torch import Tensor
from typing import Dict, Any, List, Tuple, Optional, Sequence # Added Sequence

from basicsr.data import degradations as degradations_module # Renamed to avoid conflict
from basicsr.data.data_util import paths_from_folder
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY # For dataset registration
from torchvision.transforms.functional import ( # PyTorch's color jitter
    adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation, normalize
)


@DATASET_REGISTRY.register()
class FFHQDegradationDataset(data.Dataset):
    """
    FFHQ dataset for ScaleX (formerly GFPGAN) training.
    Reads high-resolution images and generates low-quality (LQ) images on-the-fly.

    Args:
        opt (Dict[str, Any]): Configuration dictionary. Expected keys include:
            - dataroot_gt (str): Root path for GT images or LMDB file.
            - io_backend (Dict[str, Any]): IO backend configuration.
            - mean (Sequence[float]): Image normalization mean.
            - std (Sequence[float]): Image normalization standard deviation.
            - out_size (int): Output size of the images (used for component coordinates).
            - use_hflip (bool): Whether to apply horizontal flipping.
            - crop_components (bool, optional): Whether to crop facial components.
            - component_path (str, optional): Path to pre-processed facial component data.
            - eye_enlarge_ratio (float, optional): Ratio to enlarge eye regions.
            - blur_kernel_size (int): Kernel size for blur.
            - kernel_list (List[str]): List of kernel types for blur.
            - kernel_prob (List[float]): Probabilities for kernel types.
            - blur_sigma (Tuple[float, float]): Sigma range for Gaussian blur.
            - downsample_range (Tuple[float, float]): Range for downsampling scale.
            - noise_range (Optional[Tuple[float, float]]): Range for Gaussian noise.
            - jpeg_range (Optional[Tuple[int, int]]): Range for JPEG compression quality.
            - color_jitter_prob (Optional[float]): Probability for numpy-based color jitter.
            - color_jitter_pt_prob (Optional[float]): Probability for PyTorch-based color jitter.
            - color_jitter_shift (float, optional): Shift value for numpy color jitter.
            - gray_prob (Optional[float]): Probability for converting to grayscale.
            - gt_gray (bool, optional): Whether to also convert GT to grayscale if LQ is grayscaled.
            - brightness, contrast, saturation, hue (Tuple[float, float], optional): Ranges for PyTorch color jitter.
    """

    def __init__(self, opt: Dict[str, Any]):
        super().__init__() # Python 3 super()
        self.opt: Dict[str, Any] = opt
        self.file_client: Optional[FileClient] = None # Lazy initialization
        self.io_backend_opt: Dict[str, Any] = opt['io_backend']

        self.gt_folder: str = opt['dataroot_gt']
        self.mean: Sequence[float] = opt['mean']
        self.std: Sequence[float] = opt['std']
        self.out_size: int = opt['out_size'] # Used for component coordinate calculations

        # Facial component options
        self.crop_components: bool = opt.get('crop_components', False)
        self.eye_enlarge_ratio: float = opt.get('eye_enlarge_ratio', 1.0)
        self.components_list: Optional[Dict[str, Any]] = None # For pre-loaded component data
        if self.crop_components:
            component_path = opt.get('component_path')
            if component_path is None:
                raise ValueError("component_path must be provided if crop_components is True.")
            self.components_list = torch.load(component_path)

        # Initialize paths
        self.paths: List[str]
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder] # db_paths expects a list
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb' for lmdb backend, got {self.gt_folder}")
            meta_info_path = osp.join(self.gt_folder, 'meta_info.txt')
            if not osp.exists(meta_info_path):
                 raise FileNotFoundError(f"meta_info.txt not found in {self.gt_folder}. Required for LMDB.")
            with open(meta_info_path) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else: # Disk backend
            self.paths = paths_from_folder(self.gt_folder)

        # Degradation configurations
        self.blur_kernel_size: int = opt['blur_kernel_size']
        self.kernel_list: List[str] = opt['kernel_list']
        self.kernel_prob: List[float] = opt['kernel_prob']
        self.blur_sigma: Tuple[float, float] = opt['blur_sigma']
        self.downsample_range: Tuple[float, float] = opt['downsample_range']
        self.noise_range: Optional[Tuple[float, float]] = opt.get('noise_range') # Can be None
        self.jpeg_range: Optional[Tuple[int, int]] = opt.get('jpeg_range') # Can be None

        # Color jitter options
        self.color_jitter_prob: Optional[float] = opt.get('color_jitter_prob')
        self.color_jitter_pt_prob: Optional[float] = opt.get('color_jitter_pt_prob')
        self.color_jitter_shift: float = opt.get('color_jitter_shift', 20.0) / 255.0 # Normalize shift

        # Grayscale option
        self.gray_prob: Optional[float] = opt.get('gray_prob')

        # Logging (using basicsr logger)
        logger = get_root_logger()
        logger.info(f"Dataset: FFHQDegradationDataset, GT path: {self.gt_folder}, number of images: {len(self.paths)}")
        logger.info(f"Blur settings: kernel_size={self.blur_kernel_size}, sigma=[{', '.join(map(str, self.blur_sigma))}], "
                    f"kernels={self.kernel_list}, probs={self.kernel_prob}")
        logger.info(f"Downsample range: [{', '.join(map(str, self.downsample_range))}]")
        if self.noise_range:
            logger.info(f"Noise range: [{', '.join(map(str, self.noise_range))}]")
        if self.jpeg_range:
            logger.info(f"JPEG compression range: [{', '.join(map(str, self.jpeg_range))}]")
        if self.color_jitter_prob:
            logger.info(f"Numpy color jitter: prob={self.color_jitter_prob}, shift={self.color_jitter_shift * 255.0:.1f}/255")
        if self.color_jitter_pt_prob:
            logger.info(f"PyTorch color jitter: prob={self.color_jitter_pt_prob}")
        if self.gray_prob:
            logger.info(f"Grayscale conversion: prob={self.gray_prob}")

    @staticmethod
    def color_jitter_numpy(img: np.ndarray, shift_range: float) -> np.ndarray:
        """Jitter color using numpy: randomly jitter RGB values."""
        jitter_val = np.random.uniform(-shift_range, shift_range, 3).astype(np.float32)
        img_jittered = img + jitter_val
        img_jittered = np.clip(img_jittered, 0.0, 1.0)
        return img_jittered

    @staticmethod
    def color_jitter_pytorch(
        img: Tensor,
        brightness_range: Optional[Tuple[float, float]],
        contrast_range: Optional[Tuple[float, float]],
        saturation_range: Optional[Tuple[float, float]],
        hue_range: Optional[Tuple[float, float]],
    ) -> Tensor:
        """Jitter color using PyTorch transforms: brightness, contrast, saturation, hue."""
        # Create a list of transform functions to apply in random order
        transforms_to_apply: List[Any] = [] # Should be Callable, but PyTorch transforms are complex to type fully
        
        if brightness_range:
            brightness_factor = torch.empty(1).uniform_(brightness_range[0], brightness_range[1]).item()
            transforms_to_apply.append(lambda t: adjust_brightness(t, brightness_factor))
        if contrast_range:
            contrast_factor = torch.empty(1).uniform_(contrast_range[0], contrast_range[1]).item()
            transforms_to_apply.append(lambda t: adjust_contrast(t, contrast_factor))
        if saturation_range:
            saturation_factor = torch.empty(1).uniform_(saturation_range[0], saturation_range[1]).item()
            transforms_to_apply.append(lambda t: adjust_saturation(t, saturation_factor))
        if hue_range:
            hue_factor = torch.empty(1).uniform_(hue_range[0], hue_range[1]).item()
            transforms_to_apply.append(lambda t: adjust_hue(t, hue_factor))

        # Shuffle and apply transforms
        random.shuffle(transforms_to_apply) # random imported from Python stdlib
        img_transformed = img
        for transform_fn in transforms_to_apply:
            img_transformed = transform_fn(img_transformed)
        return img_transformed

    def get_component_coordinates(self, index: int, hflip_status: bool) -> List[Tensor]:
        """Get facial component coordinates (left_eye, right_eye, mouth)."""
        if self.components_list is None:
            raise RuntimeError("Components list is not loaded. Ensure 'component_path' is set and valid.")
            
        # Components are indexed by '00000000', '00000001', ...
        component_key = f'{index:08d}'
        if component_key not in self.components_list:
            raise KeyError(f"Component data for index {index} (key: {component_key}) not found.")
            
        components_bbox = self.components_list[component_key].copy() # Make a copy to modify

        if hflip_status: # If horizontally flipped
            # Exchange right and left eye data
            components_bbox['left_eye'], components_bbox['right_eye'] = \
                components_bbox['right_eye'], components_bbox['left_eye']
            
            # Adjust x-coordinates for flip (center_x' = out_size - center_x)
            for part_name in ['left_eye', 'right_eye', 'mouth']:
                components_bbox[part_name][0] = self.out_size - components_bbox[part_name][0]

        locations: List[Tensor] = []
        for part_name in ['left_eye', 'right_eye', 'mouth']:
            # Assuming format: [center_x, center_y, half_length_x, half_length_y] or similar
            # Original code implies: [center_x, center_y, half_diagonal_or_radius]
            # Let's assume components_bbox[part_name] = [cx, cy, half_len_square_bbox]
            center_x, center_y, half_len = components_bbox[part_name][0:3] # Ensure correct parsing
            
            if 'eye' in part_name:
                half_len *= self.eye_enlarge_ratio
            
            # Bbox: [x_min, y_min, x_max, y_max]
            # Original: loc = np.hstack((mean - half_len + 1, mean + half_len))
            # This seems to create [cx-hl+1, cy-hl+1, cx+hl, cy+hl] if mean=[cx,cy]
            # Corrected for clarity:
            x_min = center_x - half_len + 1 # +1 for inclusive? Original seems to use 1-based indexing logic
            y_min = center_y - half_len + 1
            x_max = center_x + half_len
            y_max = center_y + half_len
            
            loc_np = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
            locations.append(torch.from_numpy(loc_np))
            
        return locations

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.file_client is None:
            # Create a new FileClient instance for each worker process
            self.file_client = FileClient(self.io_backend_opt['type'], **self.io_backend_opt)

        gt_path: str = self.paths[index]
        img_bytes: bytes = self.file_client.get(gt_path)
        img_gt: np.ndarray = imfrombytes(img_bytes, float32=True) # (H, W, C), BGR, [0, 1]

        # Random horizontal flip
        img_gt, hflip_done = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False, return_status=True)
        h, w, _ = img_gt.shape

        # Get facial component coordinates if needed
        component_locations: Optional[List[Tensor]] = None
        if self.crop_components:
            component_locations = self.get_component_coordinates(index, hflip_done)

        # ----- Generate LQ image -----
        img_lq = img_gt.copy() # Start with a copy of GT

        # 1. Blur
        # degradation_module from basicsr.data.degradations
        kernel: np.ndarray = degradations_module.random_mixed_kernels(
            self.kernel_list, self.kernel_prob, self.blur_kernel_size,
            self.blur_sigma, self.blur_sigma, # sigma_x, sigma_y range
            (-math.pi, math.pi), # rotation_range
            noise_range=None # No noise in kernel itself
        )
        img_lq = cv2.filter2D(img_lq, -1, kernel)

        # 2. Downsample
        scale: float = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)

        # 3. Add noise
        if self.noise_range is not None:
            img_lq = degradations_module.random_add_gaussian_noise(img_lq, self.noise_range)

        # 4. JPEG compression
        if self.jpeg_range is not None:
            img_lq = degradations_module.random_add_jpg_compression(img_lq, self.jpeg_range)

        # 5. Resize back to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        # 6. Numpy-based color jitter (LQ only)
        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            img_lq = self.color_jitter_numpy(img_lq, self.color_jitter_shift)

        # 7. Grayscale conversion (LQ and optionally GT)
        if self.gray_prob is not None and (np.random.uniform() < self.gray_prob):
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
            img_lq = np.tile(img_lq[:, :, np.newaxis], (1, 1, 3)) # (H,W) -> (H,W,1) -> (H,W,3)
            if self.opt.get('gt_gray', False): # Check if GT should also be grayscaled
                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
                img_gt = np.tile(img_gt[:, :, np.newaxis], (1, 1, 3))
        
        # ----- Convert to Tensor and further processing -----
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt_tensor, img_lq_tensor = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # 8. PyTorch-based color jitter (LQ tensor only)
        if self.color_jitter_pt_prob is not None and (np.random.uniform() < self.color_jitter_pt_prob):
            brightness_range = self.opt.get('brightness', (0.5, 1.5))
            contrast_range = self.opt.get('contrast', (0.5, 1.5))
            saturation_range = self.opt.get('saturation', (0.0, 1.5)) # Saturation can go to 0 (grayscale)
            hue_range = self.opt.get('hue', (-0.1, 0.1))
            img_lq_tensor = self.color_jitter_pytorch(
                img_lq_tensor, brightness_range, contrast_range, saturation_range, hue_range
            )

        # Round and clip LQ image values (after all jittering)
        img_lq_tensor = torch.clamp((img_lq_tensor * 255.0).round(), 0, 255) / 255.0

        # Normalize both GT and LQ tensors
        normalize(img_gt_tensor, self.mean, self.std, inplace=True)
        normalize(img_lq_tensor, self.mean, self.std, inplace=True)

        # Prepare return dictionary
        return_dict: Dict[str, Any] = {
            'lq': img_lq_tensor,
            'gt': img_gt_tensor,
            'gt_path': gt_path,
        }
        if self.crop_components and component_locations:
            return_dict['loc_left_eye'] = component_locations[0]
            return_dict['loc_right_eye'] = component_locations[1]
            return_dict['loc_mouth'] = component_locations[2]
            
        return return_dict

    def __len__(self) -> int:
        return len(self.paths)