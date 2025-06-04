# scalex/archs/stylegan2_clean_arch.py
import math
import random
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Optional, Tuple, Literal, Dict, Any

from basicsr.archs.arch_util import default_init_weights
from basicsr.utils.registry import ARCH_REGISTRY

InterpolationModeClean = Literal[
    "nearest", "linear", "bilinear", "bicubic", "trilinear", "area"
]


class NormStyleCode(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


class ModulatedConv2dClean(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_style_feat: int,
        demodulate: bool = True,
        sample_mode: Optional[Literal["upsample", "downsample"]] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: int = kernel_size
        self.demodulate: bool = demodulate
        self.sample_mode: Optional[Literal["upsample", "downsample"]] = sample_mode
        self.eps: float = eps

        self.modulation: nn.Linear = nn.Linear(num_style_feat, in_channels, bias=True)
        nn.init.kaiming_normal_(
            self.modulation.weight, a=0, mode="fan_in", nonlinearity="linear"
        )
        if self.modulation.bias is not None:
            nn.init.constant_(self.modulation.bias, 1.0)

        self.weight: nn.Parameter = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
            / math.sqrt(in_channels * kernel_size**2)
        )
        self.padding: int = kernel_size // 2

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        batch_size, in_c, height, width = x.shape
        mod_style: Tensor = self.modulation(style).view(batch_size, 1, in_c, 1, 1)
        weight: Tensor = self.weight * mod_style

        if self.demodulate:
            demod_scale: Tensor = torch.rsqrt(
                weight.pow(2).sum(dim=(2, 3, 4), keepdim=True) + self.eps
            )
            weight = weight * demod_scale

        weight = weight.view(
            batch_size * self.out_channels, in_c, self.kernel_size, self.kernel_size
        )

        if self.sample_mode == "upsample":
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        elif self.sample_mode == "downsample":
            x = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)

        current_b, current_c, current_h, current_w = x.shape
        x = x.view(1, batch_size * current_c, current_h, current_w)

        out: Tensor = F.conv2d(x, weight, padding=self.padding, groups=batch_size)
        out = out.view(batch_size, self.out_channels, out.shape[2], out.shape[3])
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, kernel_size={self.kernel_size}, "
            f"num_style_feat={self.modulation.in_features}, demodulate={self.demodulate}, "
            f"sample_mode='{self.sample_mode}')"
        )


class StyleConvClean(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_style_feat: int,
        demodulate: bool = True,
        sample_mode: Optional[Literal["upsample", "downsample"]] = None,
        leaky_inplace: bool = False,
    ):
        super().__init__()
        self.modulated_conv: ModulatedConv2dClean = ModulatedConv2dClean(
            in_channels,
            out_channels,
            kernel_size,
            num_style_feat,
            demodulate=demodulate,
            sample_mode=sample_mode,
        )
        # --- REVERTED NAME ---
        self.weight: nn.Parameter = nn.Parameter(
            torch.zeros(1)
        )  # for noise injection (original name)
        # --- END REVERT ---
        self.bias: nn.Parameter = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.activate: nn.LeakyReLU = nn.LeakyReLU(
            negative_slope=0.2, inplace=leaky_inplace
        )

    def forward(
        self, x: Tensor, style: Tensor, noise: Optional[Tensor] = None
    ) -> Tensor:
        out: Tensor = self.modulated_conv(x, style) * math.sqrt(2.0)

        if noise is None:
            batch_size, _, height, width = out.shape
            noise = torch.randn(batch_size, 1, height, width, device=out.device)
        # --- REVERTED USAGE ---
        out = out + self.weight * noise
        # --- END REVERT ---

        out = out + self.bias
        out = self.activate(out)
        return out


class ToRGBClean(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_style_feat: int,
        upsample_skip: bool = True,
    ):
        super().__init__()
        self.upsample_skip: bool = upsample_skip
        self.modulated_conv: ModulatedConv2dClean = ModulatedConv2dClean(
            in_channels,
            3,
            kernel_size=1,
            num_style_feat=num_style_feat,
            demodulate=False,
        )
        self.bias: nn.Parameter = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(
        self, x: Tensor, style: Tensor, skip: Optional[Tensor] = None
    ) -> Tensor:
        out: Tensor = self.modulated_conv(x, style)
        out = out + self.bias

        if skip is not None:
            if self.upsample_skip:
                skip = F.interpolate(
                    skip, scale_factor=2.0, mode="bilinear", align_corners=False
                )
            out = out + skip
        return out


class ConstantInputClean(nn.Module):
    def __init__(self, num_channels: int, size: int):
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(
            torch.randn(1, num_channels, size, size)
        )

    def forward(self, batch_size: int) -> Tensor:
        return self.weight.repeat(batch_size, 1, 1, 1)


@ARCH_REGISTRY.register()
class StyleGAN2GeneratorClean(nn.Module):
    def __init__(
        self,
        out_size: int,
        num_style_feat: int = 512,
        num_mlp: int = 8,
        channel_multiplier: int = 2,
        narrow: float = 1.0,
        leaky_mlp_inplace: bool = False,
        leaky_conv_inplace: bool = False,
    ):
        super().__init__()
        self.num_style_feat: int = num_style_feat
        self.log_size: int = int(math.log2(out_size))
        if 2**self.log_size != out_size:
            raise ValueError(f"out_size must be a power of 2. Got {out_size}")

        style_mlp_layers: List[nn.Module] = [NormStyleCode()]
        for _ in range(num_mlp):
            linear_layer = nn.Linear(num_style_feat, num_style_feat, bias=True)
            nn.init.kaiming_normal_(
                linear_layer.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
            )
            if linear_layer.bias is not None:
                nn.init.constant_(linear_layer.bias, 0.0)
            style_mlp_layers.append(linear_layer)
            style_mlp_layers.append(
                nn.LeakyReLU(negative_slope=0.2, inplace=leaky_mlp_inplace)
            )
        self.style_mlp: nn.Sequential = nn.Sequential(*style_mlp_layers)

        calc_channels: Dict[str, int] = {
            "4": int(512 * narrow),
            "8": int(512 * narrow),
            "16": int(512 * narrow),
            "32": int(512 * narrow),
            "64": int(256 * channel_multiplier * narrow),
            "128": int(128 * channel_multiplier * narrow),
            "256": int(64 * channel_multiplier * narrow),
            "512": int(32 * channel_multiplier * narrow),
            "1024": int(16 * channel_multiplier * narrow),
        }
        self.channels: Dict[str, int] = {
            k: v for k, v in calc_channels.items() if int(k) <= out_size
        }
        for i_res_log in range(2, self.log_size + 1):
            res_str = str(2**i_res_log)
            if res_str not in self.channels:
                if res_str == "4":
                    self.channels[res_str] = int(512 * narrow)
                else:
                    raise KeyError(f"Channel count for resolution {res_str} not found.")

        self.constant_input: ConstantInputClean = ConstantInputClean(
            self.channels["4"], size=4
        )
        self.style_conv1: StyleConvClean = StyleConvClean(
            self.channels["4"],
            self.channels["4"],
            kernel_size=3,
            num_style_feat=num_style_feat,
            demodulate=True,
            sample_mode=None,
            leaky_inplace=leaky_conv_inplace,
        )
        self.to_rgb1: ToRGBClean = ToRGBClean(
            self.channels["4"], num_style_feat, upsample_skip=False
        )

        self.num_layers: int = (self.log_size - 2) * 2 + 1
        self.num_latent: int = self.log_size * 2 - 2

        self.style_convs: nn.ModuleList = nn.ModuleList()
        self.to_rgbs: nn.ModuleList = nn.ModuleList()
        self.noises: nn.Module = nn.Module()

        current_in_channels: int = self.channels["4"]
        for i_res_log in range(3, self.log_size + 1):
            res_str: str = str(2**i_res_log)
            current_out_channels: int = self.channels[res_str]

            self.style_convs.append(
                StyleConvClean(
                    current_in_channels,
                    current_out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode="upsample",
                    leaky_inplace=leaky_conv_inplace,
                )
            )
            self.style_convs.append(
                StyleConvClean(
                    current_out_channels,
                    current_out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode=None,
                    leaky_inplace=leaky_conv_inplace,
                )
            )
            self.to_rgbs.append(
                ToRGBClean(current_out_channels, num_style_feat, upsample_skip=True)
            )
            current_in_channels = current_out_channels

        # --- REVERTED NOISE BUFFER REGISTRATION ---
        for layer_idx in range(self.num_layers):
            resolution = 2 ** ((layer_idx + 5) // 2)  # (4, 4, 8, 8, 16, 16, ...)
            shape = [1, 1, resolution, resolution]
            self.noises.register_buffer(f"noise{layer_idx}", torch.randn(*shape))
        # --- END REVERT ---

    def _get_noise_list(
        self, randomize_noise: bool, device: torch.device
    ) -> List[Optional[Tensor]]:
        # --- REVERTED NOISE LIST RETRIEVAL ---
        if randomize_noise:
            return [None] * self.num_layers  # StyleConv will generate noise
        else:
            # Use the stored noise buffers, named noise0, noise1, ...
            return [getattr(self.noises, f"noise{i}") for i in range(self.num_layers)]
        # --- END REVERT ---

    def make_noise(self) -> List[Tensor]:  # This method generates new noise on the fly
        device = self.constant_input.weight.device
        noises: List[Tensor] = []  # List for newly generated noise
        for layer_idx in range(self.num_layers):
            resolution = 2 ** ((layer_idx + 5) // 2)
            noises.append(torch.randn(1, 1, resolution, resolution, device=device))
        return noises

    def get_latent(self, x: Tensor) -> Tensor:
        return self.style_mlp(x)

    def mean_latent(self, num_samples_for_mean: int) -> Tensor:
        latent_z_in: Tensor = torch.randn(
            num_samples_for_mean,
            self.num_style_feat,
            device=self.constant_input.weight.device,
        )
        mean_w_latent: Tensor = self.style_mlp(latent_z_in).mean(0, keepdim=True)
        return mean_w_latent

    def forward(
        self,
        styles: List[Tensor],
        input_is_latent: bool = False,
        noise: Optional[
            List[Optional[Tensor]]
        ] = None,  # Can be a list of noise tensors
        randomize_noise: bool = True,
        truncation: float = 1.0,
        truncation_latent: Optional[Tensor] = None,
        inject_index: Optional[int] = None,
        return_latents: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        processed_styles: List[Tensor]
        if not input_is_latent:
            processed_styles = [self.style_mlp(s) for s in styles]
        else:
            processed_styles = styles

        if truncation < 1.0 and truncation_latent is not None:
            processed_styles = [
                truncation_latent + truncation * (s - truncation_latent)
                for s in processed_styles
            ]

        latent: Tensor
        if len(processed_styles) == 1:
            if processed_styles[0].ndim < 3:
                latent = processed_styles[0].unsqueeze(1).repeat(1, self.num_latent, 1)
            else:
                latent = processed_styles[0]
        elif len(processed_styles) == 2:
            idx_to_inject = (
                inject_index
                if inject_index is not None
                else random.randint(1, self.num_latent - 1)
            )
            latent1 = processed_styles[0].unsqueeze(1).repeat(1, idx_to_inject, 1)
            latent2 = (
                processed_styles[1]
                .unsqueeze(1)
                .repeat(1, self.num_latent - idx_to_inject, 1)
            )
            latent = torch.cat([latent1, latent2], 1)
        else:
            raise ValueError(
                f"Unsupported number of styles: {len(processed_styles)}. Expected 1 or 2."
            )

        # Get the noise list based on randomize_noise and provided noise argument
        actual_noise_list: List[Optional[Tensor]]
        if noise is None:  # If no explicit noise list is passed to forward()
            actual_noise_list = self._get_noise_list(randomize_noise, latent.device)
        else:  # If an explicit noise list is passed, use it
            actual_noise_list = noise

        if len(actual_noise_list) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} noise tensors, got {len(actual_noise_list)}"
            )

        batch_size = latent.shape[0]
        out: Tensor = self.constant_input(batch_size)

        # The first noise (noise[0]) is for style_conv1
        out = self.style_conv1(out, latent[:, 0], noise=actual_noise_list[0])
        skip: Tensor = self.to_rgb1(out, latent[:, 1])  # latent[:, 1] for to_rgb1

        latent_idx: int = 1  # Start from latent index 1 (0 used by style_conv1)
        # Noise indices for the loop start from actual_noise_list[1]

        # Loop for subsequent layers
        # Original loop: for conv1, conv2, noise1, noise2, to_rgb in zip(self.style_convs[::2], self.style_convs[1::2], noise[1::2], noise[2::2], self.to_rgbs):
        # This implies noise indices map directly.
        # noise[1] for convs[0], noise[2] for convs[1], noise[3] for convs[2], etc.

        current_noise_idx_offset = 1  # Start from noise1 for the loop
        for conv_idx in range(
            0, len(self.style_convs), 2
        ):  # Iterate through pairs of style_convs
            conv1 = self.style_convs[conv_idx]
            conv2 = self.style_convs[conv_idx + 1]
            to_rgb_layer = self.to_rgbs[conv_idx // 2]

            noise1 = actual_noise_list[current_noise_idx_offset]
            noise2 = actual_noise_list[current_noise_idx_offset + 1]

            latent_idx += 1
            out = conv1(out, latent[:, latent_idx], noise=noise1)

            latent_idx += 1
            out = conv2(out, latent[:, latent_idx], noise=noise2)

            latent_idx += 1
            skip = to_rgb_layer(out, latent[:, latent_idx], skip=skip)

            current_noise_idx_offset += 2

        image: Tensor = skip
        return (image, latent) if return_latents else (image, None)
