import math
import random
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Optional, Tuple, Literal, Dict, Any  # Added Literal, Dict, Any

from basicsr.ops.fused_act import (
    FusedLeakyReLU,
    fused_leaky_relu,
)  # Custom ops from basicsr
from basicsr.utils.registry import ARCH_REGISTRY  # For registering the main generator


InterpolationMode = Literal[
    "nearest", "linear", "bilinear", "bicubic", "trilinear", "area"
]


class NormStyleCode(nn.Module):
    """Normalizes style codes."""

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Style codes with shape (Batch, Channels).
        Returns:
            Tensor: Normalized style codes.
        """
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


class EqualLinear(nn.Module):
    """
    Equalized Linear layer as in StyleGAN2.

    Args:
        in_features (int): Size of each input sample. Renamed from in_channels.
        out_features (int): Size of each output sample. Renamed from out_channels.
        bias (bool): If True, learns an additive bias. Default: True.
        bias_init_val (float): Initial value for bias. Default: 0.
        lr_mul (float): Learning rate multiplier for weights and bias. Default: 1.
        activation (Optional[Literal['fused_lrelu']]): Activation after linear. Default: None.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bias_init_val: float = 0.0,
        lr_mul: float = 1.0,
        activation: Optional[Literal["fused_lrelu"]] = None,
    ):
        super().__init__()  # Python 3 super()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.lr_mul: float = lr_mul
        self.activation: Optional[Literal["fused_lrelu"]] = activation
        if self.activation not in ["fused_lrelu", None]:
            raise ValueError(
                f"Unsupported activation: {activation}. Supported: ['fused_lrelu', None]."
            )

        # He's constant for weight scaling
        self.scale: float = (1.0 / math.sqrt(in_features)) * lr_mul

        self.weight: nn.Parameter = nn.Parameter(
            torch.randn(out_features, in_features) / lr_mul
        )
        if bias:
            self.bias: Optional[nn.Parameter] = nn.Parameter(
                torch.full((out_features,), bias_init_val)
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        # Prepare bias with lr_mul
        actual_bias: Optional[Tensor] = None
        if self.bias is not None:
            actual_bias = self.bias * self.lr_mul

        # Linear transformation
        weight_scaled: Tensor = self.weight * self.scale

        if self.activation == "fused_lrelu":
            # fused_leaky_relu expects bias to be passed to it, not to F.linear
            out: Tensor = F.linear(x, weight_scaled)  # Bias handled by fused_leaky_relu
            out = fused_leaky_relu(
                out, actual_bias if actual_bias is not None else self.bias
            )  # Pass bias to fused op
        else:
            out = F.linear(x, weight_scaled, actual_bias)
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None}, "
            f"lr_mul={self.lr_mul}, activation='{self.activation}')"
        )


class ModulatedConv2d(nn.Module):
    """
    Modulated Convolutional Layer from StyleGAN2.
    Uses bilinear interpolation for up/downsampling if sample_mode is set.
    No bias term in the convolution itself.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_style_feat: int,  # Dimension of the style vector 'w'
        demodulate: bool = True,
        sample_mode: Optional[Literal["upsample", "downsample"]] = None,
        eps: float = 1e-8,
        interpolation_mode: InterpolationMode = "bilinear",
    ):
        super().__init__()  # Python 3 super()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: int = kernel_size
        self.demodulate: bool = demodulate
        self.sample_mode: Optional[Literal["upsample", "downsample"]] = sample_mode
        self.eps: float = eps
        self.interpolation_mode: InterpolationMode = interpolation_mode

        self.align_corners: Optional[bool] = (
            None if interpolation_mode == "nearest" else False
        )

        self.scale: float = 1.0 / math.sqrt(in_channels * kernel_size**2)
        self.modulation: EqualLinear = EqualLinear(
            num_style_feat, in_channels, bias=True, bias_init_val=1.0, lr_mul=1.0
        )

        # Weights: (1, out_c, in_c, k, k) - shared across batch, modulated per instance
        self.weight: nn.Parameter = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )
        self.padding: int = kernel_size // 2

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        batch_size, in_c, height, width = x.shape

        # Modulate weights with style vector
        # style: (B, num_style_feat) -> mod_weights_style: (B, 1, in_c, 1, 1)
        mod_weights_style: Tensor = self.modulation(style).view(
            batch_size, 1, in_c, 1, 1
        )
        # self.weight: (1, out_c, in_c, k, k)
        # weight: (B, out_c, in_c, k, k) after broadcasting style
        weight: Tensor = (
            self.scale * self.weight * mod_weights_style
        )  # Apply learnable scale and instance-specific style modulation

        # Demodulation
        if self.demodulate:
            demod_scale: Tensor = torch.rsqrt(
                weight.pow(2).sum(dim=(2, 3, 4), keepdim=True) + self.eps
            )
            weight = weight * demod_scale  # (B, out_c, 1, 1, 1)

        # Reshape weight for grouped convolution: (B * out_c, in_c, k, k)
        weight = weight.view(
            batch_size * self.out_channels, in_c, self.kernel_size, self.kernel_size
        )

        # Optional up/downsampling via interpolation
        if self.sample_mode == "upsample":
            x = F.interpolate(
                x,
                scale_factor=2.0,
                mode=self.interpolation_mode,
                align_corners=self.align_corners,
            )
        elif self.sample_mode == "downsample":
            x = F.interpolate(
                x,
                scale_factor=0.5,
                mode=self.interpolation_mode,
                align_corners=self.align_corners,
            )

        # Reshape input for grouped convolution
        # (B, in_c, H, W) -> (1, B * in_c, H, W)
        current_b, current_c, current_h, current_w = (
            x.shape
        )  # Get current shape after potential interpolation
        x = x.view(1, batch_size * current_c, current_h, current_w)

        # Grouped convolution: each item in batch is a separate group
        out: Tensor = F.conv2d(x, weight, padding=self.padding, groups=batch_size)

        # Reshape output back: (1, B * out_c, H_out, W_out) -> (B, out_c, H_out, W_out)
        out = out.view(batch_size, self.out_channels, out.shape[2], out.shape[3])
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, kernel_size={self.kernel_size}, "
            f"num_style_feat={self.modulation.in_features}, demodulate={self.demodulate}, "
            f"sample_mode='{self.sample_mode}', interpolation_mode='{self.interpolation_mode}')"
        )


class StyleConv(nn.Module):
    """Style-modulated convolutional layer with noise injection and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_style_feat: int,
        demodulate: bool = True,
        sample_mode: Optional[Literal["upsample", "downsample"]] = None,
        interpolation_mode: InterpolationMode = "bilinear",
    ):
        super().__init__()  # Python 3 super()
        self.modulated_conv: ModulatedConv2d = ModulatedConv2d(
            in_channels,
            out_channels,
            kernel_size,
            num_style_feat,
            demodulate=demodulate,
            sample_mode=sample_mode,
            interpolation_mode=interpolation_mode,
        )
        self.noise_strength: nn.Parameter = nn.Parameter(
            torch.zeros(1)
        )  # Learnable strength for noise
        self.activate: FusedLeakyReLU = FusedLeakyReLU(
            out_channels
        )  # Bias is handled by FusedLeakyReLU

    def forward(
        self, x: Tensor, style: Tensor, noise: Optional[Tensor] = None
    ) -> Tensor:
        out: Tensor = self.modulated_conv(x, style)

        # Noise injection
        if noise is None:  # If no pre-generated noise, create it
            batch_size, _, height, width = out.shape
            noise = torch.randn(batch_size, 1, height, width, device=out.device)
        out = out + self.noise_strength * noise

        # Activation (FusedLeakyReLU includes bias)
        out = self.activate(out)
        return out


class ToRGB(nn.Module):
    """Converts features to RGB image, optionally adding a skip connection."""

    def __init__(
        self,
        in_channels: int,
        num_style_feat: int,
        upsample_skip: bool = True,  # Whether to upsample the skip connection
        interpolation_mode: InterpolationMode = "bilinear",
    ):
        super().__init__()  # Python 3 super()
        self.upsample_skip: bool = upsample_skip
        self.interpolation_mode: InterpolationMode = interpolation_mode
        self.align_corners: Optional[bool] = (
            None if interpolation_mode == "nearest" else False
        )

        # Modulated 1x1 conv to map features to 3 RGB channels
        self.modulated_conv: ModulatedConv2d = ModulatedConv2d(
            in_channels,
            3,
            kernel_size=1,
            num_style_feat=num_style_feat,
            demodulate=False,  # No demodulation for ToRGB usually
            interpolation_mode=interpolation_mode,  # Passed for consistency, though sample_mode=None here
        )
        self.bias: nn.Parameter = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(
        self, x: Tensor, style: Tensor, skip: Optional[Tensor] = None
    ) -> Tensor:
        out: Tensor = self.modulated_conv(x, style)
        out = out + self.bias  # Apply learnable bias

        if skip is not None:
            if self.upsample_skip:
                skip = F.interpolate(
                    skip,
                    scale_factor=2.0,
                    mode=self.interpolation_mode,
                    align_corners=self.align_corners,
                )
            out = out + skip
        return out


class ConstantInput(nn.Module):
    """Generates a learnable constant tensor as input."""

    def __init__(self, num_channels: int, size: int):  # Renamed num_channel
        super().__init__()  # Python 3 super()
        self.weight: nn.Parameter = nn.Parameter(
            torch.randn(1, num_channels, size, size)
        )

    def forward(self, batch_size: int) -> Tensor:  # Renamed batch to batch_size
        return self.weight.repeat(batch_size, 1, 1, 1)


@ARCH_REGISTRY.register()
class StyleGAN2GeneratorBilinear(nn.Module):
    """
    StyleGAN2 Generator using bilinear interpolation for upsampling.

    Args:
        out_size (int): Output spatial size (must be power of 2).
        num_style_feat (int): Dimension of style vector 'w'. Default: 512.
        num_mlp (int): Number of layers in Style MLP. Default: 8.
        channel_multiplier (int): Multiplier for network channels. Default: 2.
        lr_mlp (float): Learning rate multiplier for Style MLP. Default: 0.01.
        narrow (float): Factor to narrow channel counts. Default: 1.0.
        interpolation_mode (InterpolationMode): Interpolation mode for up/downsampling. Default: 'bilinear'.
    """

    def __init__(
        self,
        out_size: int,
        num_style_feat: int = 512,
        num_mlp: int = 8,
        channel_multiplier: int = 2,
        lr_mlp: float = 0.01,
        narrow: float = 1.0,
        interpolation_mode: InterpolationMode = "bilinear",
    ):
        super().__init__()  # Python 3 super()
        self.num_style_feat: int = num_style_feat
        self.log_size: int = int(math.log2(out_size))
        if 2**self.log_size != out_size:
            raise ValueError(f"out_size must be a power of 2. Got {out_size}")

        # Style MLP
        style_mlp_layers: List[nn.Module] = [NormStyleCode()]
        for _ in range(num_mlp):
            style_mlp_layers.append(
                EqualLinear(
                    num_style_feat,
                    num_style_feat,
                    bias=True,
                    bias_init_val=0,
                    lr_mul=lr_mlp,
                    activation="fused_lrelu",
                )
            )
        self.style_mlp: nn.Sequential = nn.Sequential(*style_mlp_layers)

        # Channel configuration based on StyleGAN2 paper
        self.channels: Dict[str, int] = {
            # Format: 'resolution_str': num_channels
            str(2**i): min(
                int(num_style_feat * narrow),
                int(base_ch * channel_multiplier * narrow / (2 ** (j - 1))),
            )
            for j, i in enumerate(
                range(2, self.log_size + 1)
            )  # i from 2 (4x4) to log_size (out_size)
            # Approximation of StyleGAN2 channel scheme. Original uses specific values.
            # For simplicity, let's use the provided `channels` dict structure if it's standard for this repo.
        }
        # Using the originally provided channel calculation structure for consistency:
        calc_channels: Dict[str, int] = {
            "4": int(512 * narrow),
            "8": int(512 * narrow),
            "16": int(512 * narrow),
            "32": int(
                512 * narrow
            ),  # Up to 32x32, channels are capped or based on 512*narrow
            "64": int(256 * channel_multiplier * narrow),
            "128": int(128 * channel_multiplier * narrow),
            "256": int(64 * channel_multiplier * narrow),
            "512": int(32 * channel_multiplier * narrow),
            "1024": int(16 * channel_multiplier * narrow),
        }
        # Filter to only include relevant resolutions up to out_size
        self.channels = {k: v for k, v in calc_channels.items() if int(k) <= out_size}
        # Ensure all necessary resolutions are present from 4x4 up to out_size
        for i_res_log in range(2, self.log_size + 1):
            res_str = str(2**i_res_log)
            if res_str not in self.channels:
                # Fallback or error if a resolution's channel count is missing
                # This might happen if out_size is e.g. 4 or 8 and calc_channels doesn't cover it well.
                # For StyleGAN2, minimum is usually 4x4.
                if res_str == "4":
                    self.channels[res_str] = int(512 * narrow)  # Ensure 4x4 is present
                else:
                    raise KeyError(
                        f"Channel count for resolution {res_str} not found in channel configuration."
                    )

        # Initial layers (4x4 resolution)
        self.constant_input: ConstantInput = ConstantInput(self.channels["4"], size=4)
        self.style_conv1: StyleConv = StyleConv(
            self.channels["4"],
            self.channels["4"],
            kernel_size=3,
            num_style_feat=num_style_feat,
            demodulate=True,
            sample_mode=None,
            interpolation_mode=interpolation_mode,
        )
        self.to_rgb1: ToRGB = ToRGB(
            self.channels["4"],
            num_style_feat,
            upsample_skip=False,  # No upsampling for the first ToRGB
            interpolation_mode=interpolation_mode,
        )

        # Number of layers and latent codes needed
        self.num_layers: int = (
            self.log_size - 2
        ) * 2 + 1  # Total StyleConv layers (conv1 + pairs in loop)
        self.num_latent: int = (
            self.log_size * 2 - 2
        )  # Number of w latents for conv layers and ToRGBs

        # Main convolutional blocks (from 8x8 up to out_size)
        self.style_convs: nn.ModuleList = nn.ModuleList()
        self.to_rgbs: nn.ModuleList = nn.ModuleList()
        self.noises: nn.Module = nn.Module()  # Container for noise buffers

        current_in_channels: int = self.channels["4"]
        for i_res_log in range(
            3, self.log_size + 1
        ):  # Loop from 8x8 (2^3) up to out_size (2^log_size)
            res_str: str = str(2**i_res_log)
            current_out_channels: int = self.channels[res_str]

            # First StyleConv in the pair (upsamples)
            self.style_convs.append(
                StyleConv(
                    current_in_channels,
                    current_out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode="upsample",
                    interpolation_mode=interpolation_mode,
                )
            )
            # Second StyleConv in the pair (no spatial change)
            self.style_convs.append(
                StyleConv(
                    current_out_channels,
                    current_out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode=None,
                    interpolation_mode=interpolation_mode,
                )
            )
            self.to_rgbs.append(
                ToRGB(
                    current_out_channels,
                    num_style_feat,
                    upsample_skip=True,
                    interpolation_mode=interpolation_mode,
                )
            )
            current_in_channels = current_out_channels

        # Register noise buffers
        # Noise for style_conv1 (operates on 4x4)
        # The original code's layer_idx loop for noise was a bit complex to map directly.
        # Let's be explicit:
        # Noise for style_conv1 (4x4)
        self.noises.register_buffer(f"noise_s1", torch.randn(1, 1, 4, 4))
        # Noises for the pairs of style_convs
        noise_counter = 0
        for i_res_log in range(3, self.log_size + 1):  # 8x8 up to out_size
            resolution = 2**i_res_log
            # Noise for the upsampling conv in the pair
            self.noises.register_buffer(
                f"noise_p{noise_counter}_0", torch.randn(1, 1, resolution, resolution)
            )
            # Noise for the non-upsampling conv in the pair
            self.noises.register_buffer(
                f"noise_p{noise_counter}_1", torch.randn(1, 1, resolution, resolution)
            )
            noise_counter += 1

    def _get_noise_list(
        self, randomize: bool, device: torch.device
    ) -> List[Optional[Tensor]]:
        """Helper to get the list of noise tensors."""
        if randomize:
            return [
                None
            ] * self.num_layers  # Will trigger noise generation in StyleConv

        noise_list: List[Optional[Tensor]] = []
        noise_list.append(getattr(self.noises, "noise_s1"))  # For style_conv1

        noise_counter = 0
        for _ in range(3, self.log_size + 1):  # 8x8 up to out_size
            noise_list.append(getattr(self.noises, f"noise_p{noise_counter}_0"))
            noise_list.append(getattr(self.noises, f"noise_p{noise_counter}_1"))
            noise_counter += 1
        return noise_list

    def make_noise(self) -> List[Tensor]:
        """Generates a list of noise tensors on the fly (not used if randomize_noise=True in forward)."""
        device = self.constant_input.weight.device
        noises: List[Tensor] = [
            torch.randn(1, 1, 4, 4, device=device)
        ]  # For style_conv1
        for i_res_log in range(3, self.log_size + 1):  # 8x8 up to out_size
            resolution = 2**i_res_log
            noises.append(
                torch.randn(1, 1, resolution, resolution, device=device)
            )  # For upsampling conv
            noises.append(
                torch.randn(1, 1, resolution, resolution, device=device)
            )  # For regular conv
        return noises

    def get_latent(self, x: Tensor) -> Tensor:
        """Passes input z through Style MLP to get w."""
        return self.style_mlp(x)

    def mean_latent(
        self, num_samples_for_mean: int
    ) -> Tensor:  # Renamed num_latent to num_samples_for_mean
        """Calculates the mean w latent."""
        # Generate random z vectors
        latent_z_in: Tensor = torch.randn(
            num_samples_for_mean,
            self.num_style_feat,
            device=self.constant_input.weight.device,
        )
        # Map z to w and average
        mean_w_latent: Tensor = self.style_mlp(latent_z_in).mean(0, keepdim=True)
        return mean_w_latent

    def forward(
        self,
        styles: List[Tensor],
        input_is_latent: bool = False,  # If True, styles are already 'w' latents
        noise: Optional[
            List[Optional[Tensor]]
        ] = None,  # Can provide a list of noise tensors
        randomize_noise: bool = True,  # If True, StyleConv generates noise or uses its buffer if noise is None
        truncation: float = 1.0,  # Truncation trick factor
        truncation_latent: Optional[
            Tensor
        ] = None,  # The mean 'w' latent for truncation
        inject_index: Optional[int] = None,  # Index for style mixing
        return_latents: bool = False,  # If True, returns the 'w' latent used
    ) -> Tuple[Tensor, Optional[Tensor]]:

        # 1. Process styles: map z to w if needed, apply truncation
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

        # 2. Prepare latent tensor for layers (handles style mixing)
        latent: Tensor
        if len(processed_styles) == 1:
            # If single style, repeat it for all num_latent injection points
            # num_latent = total w vectors needed for all convs and ToRGBs
            if processed_styles[0].ndim < 3:  # (B, num_style_feat)
                latent = processed_styles[0].unsqueeze(1).repeat(1, self.num_latent, 1)
            else:  # (B, num_latent, num_style_feat) - already prepared per-layer latents
                latent = processed_styles[0]
        elif len(processed_styles) == 2:  # Style mixing
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

        # 3. Prepare noise
        # If `noise` is None, `_get_noise_list` handles `randomize_noise` to either
        # return existing buffers or a list of Nones (StyleConv will make its own noise).
        # If `noise` is provided as a list, it's used directly.
        actual_noise_list: List[Optional[Tensor]]
        if noise is None:
            actual_noise_list = self._get_noise_list(randomize_noise, latent.device)
        else:
            actual_noise_list = noise

        if len(actual_noise_list) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} noise tensors, got {len(actual_noise_list)}"
            )

        # 4. Main generation path
        batch_size = latent.shape[0]
        out: Tensor = self.constant_input(batch_size)

        # Initial 4x4 block
        # latent[:, 0] for style_conv1's modulation
        # latent[:, 1] for to_rgb1's modulation
        out = self.style_conv1(out, latent[:, 0], noise=actual_noise_list[0])
        skip: Tensor = self.to_rgb1(
            out, latent[:, 1]
        )  # No skip connection for the first ToRGB

        latent_idx: int = 1  # Start from latent index 1 (0 used by style_conv1)
        noise_list_idx: int = 1  # Start from noise index 1

        # Loop through resolution blocks (8x8, 16x16, ..., out_size x out_size)
        for conv_pair_idx in range(len(self.style_convs) // 2):
            conv1 = self.style_convs[conv_pair_idx * 2]
            conv2 = self.style_convs[conv_pair_idx * 2 + 1]
            to_rgb_layer = self.to_rgbs[conv_pair_idx]

            # Each StyleConv uses one latent, so a pair uses two
            # Each ToRGB also uses one latent
            latent_idx += 1  # For conv1 modulation
            out = conv1(
                out, latent[:, latent_idx], noise=actual_noise_list[noise_list_idx]
            )
            noise_list_idx += 1

            latent_idx += 1  # For conv2 modulation
            out = conv2(
                out, latent[:, latent_idx], noise=actual_noise_list[noise_list_idx]
            )
            noise_list_idx += 1

            latent_idx += 1  # For to_rgb_layer modulation
            skip = to_rgb_layer(
                out, latent[:, latent_idx], skip=skip
            )  # Add previous skip

        image: Tensor = skip

        return (image, latent) if return_latents else (image, None)


# These are helper classes, often part of discriminator or used by other GFPGAN components.
# Included here as they were in the original stylegan2_bilinear_arch.py


class ScaledLeakyReLU(nn.Module):
    """Scaled LeakyReLU: F.leaky_relu(x, negative_slope) * sqrt(2)."""

    def __init__(self, negative_slope: float = 0.2):
        super().__init__()  # Python 3 super()
        self.negative_slope: float = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = F.leaky_relu(x, negative_slope=self.negative_slope)
        return out * math.sqrt(2.0)  # Ensure float for sqrt


class EqualConv2d(nn.Module):
    """Equalized Convolutional Layer (no modulation by style)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        bias_init_val: float = 0.0,
    ):
        super().__init__()  # Python 3 super()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: int = kernel_size
        self.stride: int = stride
        self.padding: int = padding

        # He's constant for weight scaling
        self.scale: float = 1.0 / math.sqrt(in_channels * kernel_size**2)

        self.weight: nn.Parameter = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias: Optional[nn.Parameter] = nn.Parameter(
                torch.full((out_channels,), bias_init_val)
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(
            x,
            self.weight * self.scale,  # Apply scaling to weights
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, bias={self.bias is not None})"
        )


class ConvLayer(nn.Sequential):  # Inherits from nn.Sequential
    """
    Convolutional Layer used in StyleGAN2 Discriminator (and by GFPGAN U-Net).
    Optionally downsamples using interpolation before convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        downsample: bool = False,
        bias: bool = True,
        activate: bool = True,  # If True, applies FusedLeakyReLU or ScaledLeakyReLU
        interpolation_mode: InterpolationMode = "bilinear",
    ):
        layers: List[nn.Module] = []
        self.interpolation_mode: InterpolationMode = interpolation_mode

        align_corners: Optional[bool] = (
            None if interpolation_mode == "nearest" else False
        )

        if downsample:
            # Using nn.Upsample for downsampling as a Module
            layers.append(
                nn.Upsample(
                    scale_factor=0.5,
                    mode=interpolation_mode,
                    align_corners=align_corners,
                )
            )

        # Padding for the convolution
        padding: int = kernel_size // 2

        # EqualizedConv2d or standard Conv2d could be used. Original uses EqualConv2d.
        # Bias is handled by activation if FusedLeakyReLU, so pass bias=False to Conv if activate and bias are True.
        conv_bias: bool = bias and not (
            activate and bias
        )  # Only add conv bias if not handled by FusedLeakyReLU
        layers.append(
            EqualConv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=padding,
                bias=conv_bias,  # Bias handled by FusedLeakyReLU if activate=True, bias=True
            )
        )

        # Activation
        if activate:
            if bias:  # If bias is conceptually True for the layer
                layers.append(
                    FusedLeakyReLU(out_channels)
                )  # FusedLeakyReLU has its own bias
            else:
                layers.append(
                    ScaledLeakyReLU(0.2)
                )  # ScaledLeakyReLU does not manage bias

        super().__init__(*layers)  # Python 3 super() for nn.Sequential


class ResBlock(nn.Module):
    """Residual Block, typically for Discriminator, using ConvLayer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,  # Output channels after downsampling
        interpolation_mode: InterpolationMode = "bilinear",
    ):
        super().__init__()  # Python 3 super()
        # First conv: no change in channels or resolution
        self.conv1: ConvLayer = ConvLayer(
            in_channels,
            in_channels,
            kernel_size=3,
            bias=True,
            activate=True,
            interpolation_mode=interpolation_mode,  # Not used if downsample=False
        )
        # Second conv: maps to out_channels and downsamples
        self.conv2: ConvLayer = ConvLayer(
            in_channels,
            out_channels,
            kernel_size=3,
            downsample=True,
            bias=True,
            activate=True,
            interpolation_mode=interpolation_mode,
        )
        # Skip connection: maps to out_channels and downsamples
        self.skip: ConvLayer = ConvLayer(
            in_channels,
            out_channels,
            kernel_size=1,
            downsample=True,
            bias=False,
            activate=False,  # No activation on skip, bias=False for 1x1 conv
            interpolation_mode=interpolation_mode,
        )

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.conv1(x)
        out = self.conv2(out)

        skip_connection: Tensor = self.skip(x)
        # Normalize sum as in StyleGAN2
        return (out + skip_connection) / math.sqrt(2.0)
