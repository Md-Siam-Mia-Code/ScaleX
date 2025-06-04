import math
import random
import torch
from torch import nn, Tensor  # Added Tensor
from torch.nn import functional as F
from typing import List, Tuple, Optional, Dict, Any, Sequence  # Added for type hinting

from basicsr.archs.stylegan2_arch import (  # These are from basicsr, not local
    ConvLayer,
    EqualConv2d,
    EqualLinear,
    ResBlock,  # Used in GFPGANv1 U-Net encoder
    ScaledLeakyReLU,
    StyleGAN2Generator,  # Parent class for StyleGAN2GeneratorSFT
)
from basicsr.ops.fused_act import FusedLeakyReLU  # Used in ConvUpLayer
from basicsr.utils.registry import ARCH_REGISTRY


class StyleGAN2GeneratorSFT(StyleGAN2Generator):
    """
    StyleGAN2 Generator with SFT modulation (Spatial Feature Transform).
    Uses UpFirDn operations from basicsr.archs.stylegan2_arch.

    Args:
        out_size (int): Spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Number of layers in MLP style network. Default: 8.
        channel_multiplier (int): Channel multiplier for StyleGAN2 network. Default: 2.
        resample_kernel (Sequence[int]): 1D resample kernel. Default: (1, 3, 3, 1).
        lr_mlp (float): Learning rate multiplier for MLP layers. Default: 0.01.
        narrow (float): Channel narrowing ratio. Default: 1.0.
        sft_half (bool): Apply SFT to half of the input channels. Default: False.
    """

    def __init__(
        self,
        out_size: int,
        num_style_feat: int = 512,
        num_mlp: int = 8,
        channel_multiplier: int = 2,
        resample_kernel: Sequence[int] = (1, 3, 3, 1),
        lr_mlp: float = 0.01,
        narrow: float = 1.0,
        sft_half: bool = False,
    ):
        super().__init__(  # Python 3 super()
            out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            resample_kernel=resample_kernel,
            lr_mlp=lr_mlp,
            narrow=narrow,
        )
        self.sft_half: bool = sft_half

    def forward(
        self,
        styles: List[Tensor],
        conditions: List[Tensor],
        input_is_latent: bool = False,
        noise: List[Tensor | None] | None = None,
        randomize_noise: bool = True,
        truncation: float = 1.0,
        truncation_latent: Tensor | None = None,
        inject_index: int | None = None,
        return_latents: bool = False,
    ) -> Tuple[Tensor, Tensor | None]:
        """Forward function for StyleGAN2GeneratorSFT.

        Args:
            styles (List[Tensor]): Sample codes of styles.
            conditions (List[Tensor]): SFT conditions for the generator.
            input_is_latent (bool): Whether input is latent style. Default: False.
            noise (List[Tensor | None] | None): Input noise list or None. Default: None.
            randomize_noise (bool): Randomize noise if 'noise' is None. Default: True.
            truncation (float): Truncation ratio. Default: 1.0.
            truncation_latent (Tensor | None): Truncation latent tensor. Default: None.
            inject_index (int | None): Injection index for style mixing. Default: None.
            return_latents (bool): Whether to return style latents. Default: False.

        Returns:
            Tuple[Tensor, Tensor | None]: Generated image and latents (if requested).
        """
        # Style codes -> latents with Style MLP layer
        if not input_is_latent:
            processed_styles: List[Tensor] = [self.style_mlp(s) for s in styles]
        else:
            processed_styles = styles

        # Noises
        actual_noise: List[Tensor | None]
        if noise is None:
            if randomize_noise:
                actual_noise = [None] * self.num_layers
            else:
                actual_noise = [
                    getattr(self.noises, f"noise{i}") for i in range(self.num_layers)
                ]
        else:
            actual_noise = noise

        # Style truncation
        if truncation < 1 and truncation_latent is not None:
            truncated_styles: List[Tensor] = []
            for style in processed_styles:
                truncated_styles.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )
            processed_styles = truncated_styles

        # Get style latents with injection
        latent: Tensor
        if len(processed_styles) == 1:
            current_inject_index: int = self.num_latent
            if processed_styles[0].ndim < 3:
                latent = (
                    processed_styles[0].unsqueeze(1).repeat(1, current_inject_index, 1)
                )
            else:
                latent = processed_styles[0]
        elif len(processed_styles) == 2:
            if inject_index is None:
                current_inject_index = random.randint(1, self.num_latent - 1)
            else:
                current_inject_index = inject_index

            latent1 = (
                processed_styles[0].unsqueeze(1).repeat(1, current_inject_index, 1)
            )
            latent2 = (
                processed_styles[1]
                .unsqueeze(1)
                .repeat(1, self.num_latent - current_inject_index, 1)
            )
            latent = torch.cat([latent1, latent2], 1)
        else:
            if not processed_styles:
                raise ValueError("Styles list cannot be empty.")
            raise ValueError(
                f"Unsupported number of styles for mixing: {len(processed_styles)}. Expected 1 or 2."
            )

        # Main generation (attributes like constant_input, style_conv1, etc., are from StyleGAN2Generator)
        out: Tensor = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=actual_noise[0])
        skip: Tensor = self.to_rgb1(out, latent[:, 1])

        layer_idx: int = 1
        noise_idx: int = 1
        condition_idx: int = 0  # SFT conditions are pairs of (scale, shift)

        for conv1, conv2, to_rgb in zip(
            self.style_convs[::2], self.style_convs[1::2], self.to_rgbs
        ):
            current_noise1 = actual_noise[noise_idx]
            current_noise2 = actual_noise[noise_idx + 1]

            out = conv1(out, latent[:, layer_idx], noise=current_noise1)

            if condition_idx < len(
                conditions
            ):  # Check if enough conditions are provided
                sft_scale = conditions[condition_idx]
                sft_shift = conditions[
                    condition_idx + 1
                ]  # Assuming conditions are [scale0, shift0, scale1, shift1, ...]

                if self.sft_half:
                    out_same, out_sft = torch.split(out, out.size(1) // 2, dim=1)
                    # Ensure sft_scale and sft_shift match out_sft's channel dimension
                    out_sft = out_sft * sft_scale + sft_shift
                    out = torch.cat([out_same, out_sft], dim=1)
                else:
                    # Ensure sft_scale and sft_shift match out's channel dimension
                    out = out * sft_scale + sft_shift

            out = conv2(out, latent[:, layer_idx + 1], noise=current_noise2)
            skip = to_rgb(out, latent[:, layer_idx + 2], skip)

            layer_idx += 2
            noise_idx += 2
            condition_idx += 2  # Consumed two conditions (scale, shift)

        image: Tensor = skip

        if return_latents:
            return image, latent
        else:
            return image, None


class ConvUpLayer(nn.Module):
    """
    Convolutional upsampling layer (bilinear upsampler + Conv).
    This is a custom layer for GFPGAN's U-Net decoder, distinct from StyleGAN2's UpFirDn.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        bias_init_val: float = 0.0,
        activate: bool = True,
    ):
        super().__init__()  # Python 3 super()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: int = kernel_size
        self.stride: int = stride
        self.padding: int = padding

        self.scale: float = 1.0 / math.sqrt(in_channels * kernel_size**2)
        self.weight: nn.Parameter = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )

        if bias:  # Bias is handled by FusedLeakyReLU if activate=True and bias=True
            self.bias: nn.Parameter | None = nn.Parameter(
                torch.full((out_channels,), bias_init_val)
            )
        else:
            self.register_parameter("bias", None)  # Explicitly register as None

        # Activation
        self.activation: nn.Module | None
        if activate:
            # FusedLeakyReLU handles bias internally if provided
            self.activation = FusedLeakyReLU(
                out_channels, bias=bias, negative_slope=0.2
            )  # Pass bias to FusedLeakyReLU
            if (
                bias
            ):  # If FusedLeakyReLU handles bias, set self.bias to None for F.conv2d
                self.bias = None
        else:
            self.activation = None

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        out = F.conv2d(
            input=out,
            weight=self.weight * self.scale,  # Apply weight scaling
            bias=self.bias,  # self.bias is None if FusedLeakyReLU handles it
            stride=self.stride,
            padding=self.padding,
        )
        if self.activation is not None:
            out = self.activation(out)
        return out


class ResUpBlock(nn.Module):
    """Residual block with upsampling, using ConvUpLayer."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()  # Python 3 super()
        # Standard conv layer for the first convolution
        self.conv1: ConvLayer = ConvLayer(
            in_channels, in_channels, kernel_size=3, bias=True, activate=True
        )
        # Upsampling conv layer for the main path
        self.conv2: ConvUpLayer = ConvUpLayer(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            activate=True,
        )
        # Upsampling conv layer for the skip connection (1x1 kernel for channel matching if needed, or same as conv2)
        self.skip: ConvUpLayer = ConvUpLayer(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            activate=False,  # No activation on skip
        )

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.conv1(x)
        out = self.conv2(out)

        skip_connection: Tensor = self.skip(x)
        out = (out + skip_connection) / math.sqrt(2)  # Normalize combined output
        return out


@ARCH_REGISTRY.register()
class GFPGANv1(nn.Module):
    """The GFPGANv1 architecture: U-Net + StyleGAN2 decoder (with UpFirDn) with SFT."""

    def __init__(
        self,
        out_size: int,
        num_style_feat: int = 512,
        channel_multiplier: int = 1,  # GFPGAN typically uses 1 here
        resample_kernel: Sequence[int] = (1, 3, 3, 1),
        decoder_load_path: str | None = None,
        fix_decoder: bool = True,
        # For StyleGAN decoder
        num_mlp: int = 8,
        lr_mlp: float = 0.01,
        input_is_latent: bool = False,
        different_w: bool = False,
        narrow: float = 1.0,
        sft_half: bool = False,
    ):
        super().__init__()  # Python 3 super()
        self.input_is_latent: bool = input_is_latent
        self.different_w: bool = different_w
        self.num_style_feat: int = num_style_feat
        self.sft_half: bool = sft_half  # Store for SFT condition layers

        unet_narrow: float = narrow * 0.5
        channels: Dict[str, int] = {
            "4": int(512 * unet_narrow),
            "8": int(512 * unet_narrow),
            "16": int(512 * unet_narrow),
            "32": int(512 * unet_narrow),
            "64": int(256 * channel_multiplier * unet_narrow),
            "128": int(128 * channel_multiplier * unet_narrow),
            "256": int(64 * channel_multiplier * unet_narrow),
            "512": int(32 * channel_multiplier * unet_narrow),
            "1024": int(16 * channel_multiplier * unet_narrow),
        }

        self.log_size: int = int(math.log2(out_size))
        if 2**self.log_size != out_size:
            raise ValueError(f"out_size must be a power of 2. Got {out_size}")

        first_conv_out_channels_key: str = str(
            out_size
        )  # U-Net starts at full resolution features
        self.conv_body_first: ConvLayer = ConvLayer(
            3,
            channels[first_conv_out_channels_key],
            kernel_size=1,
            bias=True,
            activate=True,
        )

        # U-Net Encoder (Downsample)
        in_c: int = channels[first_conv_out_channels_key]
        self.conv_body_down: nn.ModuleList = nn.ModuleList()
        # ResBlock from basicsr.archs.stylegan2_arch uses resample_kernel for downsampling
        for i in range(
            self.log_size, 2, -1
        ):  # From log_size down to res 4x4 (i.e., loop until i=3)
            out_c: int = channels[str(2 ** (i - 1))]
            self.conv_body_down.append(ResBlock(in_c, out_c, resample_kernel))
            in_c = out_c

        self.final_conv: ConvLayer = ConvLayer(
            in_c, channels["4"], kernel_size=3, bias=True, activate=True
        )

        # U-Net Decoder (Upsample using ResUpBlock)
        in_c = channels["4"]
        self.conv_body_up: nn.ModuleList = nn.ModuleList()
        for i in range(
            3, self.log_size + 1
        ):  # From res 8x8 (2^3) up to out_size (2^log_size)
            out_c = channels[str(2**i)]
            self.conv_body_up.append(ResUpBlock(in_c, out_c))
            in_c = out_c

        # ToRGB layers for intermediate outputs from U-Net decoder
        self.toRGB: nn.ModuleList = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            current_channels: int = channels[str(2**i)]
            self.toRGB.append(
                EqualConv2d(
                    current_channels,
                    3,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                    bias_init_val=0,
                )
            )

        # Linear layer for style code
        if self.different_w:
            linear_out_channel: int = (self.log_size * 2 - 2) * num_style_feat
        else:
            linear_out_channel = num_style_feat

        self.final_linear: EqualLinear = EqualLinear(
            channels["4"] * 4 * 4,
            linear_out_channel,
            bias=True,
            bias_init_val=0,
            lr_mul=1,
            activation=None,
        )

        # StyleGAN2 decoder with SFT (using UpFirDn)
        self.stylegan_decoder: StyleGAN2GeneratorSFT = StyleGAN2GeneratorSFT(
            out_size=out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,  # Pass GFPGAN's CM
            resample_kernel=resample_kernel,
            lr_mlp=lr_mlp,
            narrow=narrow,
            sft_half=sft_half,
        )

        if decoder_load_path:
            try:
                state_dict = torch.load(
                    decoder_load_path, map_location=lambda storage, loc: storage
                )
                key_to_load = (
                    "params_ema"
                    if "params_ema" in state_dict
                    else "g_ema" if "g_ema" in state_dict else None
                )
                if key_to_load:
                    self.stylegan_decoder.load_state_dict(
                        state_dict[key_to_load], strict=False
                    )  # strict=False often safer for g_ema
                    print(
                        f"Loaded pre-trained StyleGAN2 decoder from: {decoder_load_path} (key: {key_to_load})"
                    )
                else:  # Try loading the whole dict
                    self.stylegan_decoder.load_state_dict(state_dict)
                    print(
                        f"Loaded pre-trained StyleGAN2 decoder from: {decoder_load_path} (root)"
                    )
            except Exception as e:
                print(
                    f"Error loading pre-trained StyleGAN2 decoder: {e}. Check path and checkpoint structure."
                )

        if fix_decoder:
            for param in self.stylegan_decoder.parameters():
                param.requires_grad = False

        # SFT condition layers (scale and shift)
        self.condition_scale: nn.ModuleList = nn.ModuleList()
        self.condition_shift: nn.ModuleList = nn.ModuleList()
        for i in range(
            3, self.log_size + 1
        ):  # For U-Net output features from 8x8 up to out_size
            unet_out_channels: int = channels[str(2**i)]

            # Determine target channels for SFT based on StyleGAN decoder's internal channels
            # Assuming self.stylegan_decoder.channels dictionary exists and is populated correctly
            stylegan_res_channels: int = self.stylegan_decoder.channels[str(2**i)]

            sft_target_channels: int
            if self.sft_half:
                sft_target_channels = stylegan_res_channels // 2
            else:
                sft_target_channels = stylegan_res_channels

            self.condition_scale.append(
                nn.Sequential(
                    EqualConv2d(
                        unet_out_channels,
                        unet_out_channels,
                        3,
                        stride=1,
                        padding=1,
                        bias=True,
                        bias_init_val=0,
                    ),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(
                        unet_out_channels,
                        sft_target_channels,
                        3,
                        stride=1,
                        padding=1,
                        bias=True,
                        bias_init_val=1,
                    ),
                )
            )
            self.condition_shift.append(
                nn.Sequential(
                    EqualConv2d(
                        unet_out_channels,
                        unet_out_channels,
                        3,
                        stride=1,
                        padding=1,
                        bias=True,
                        bias_init_val=0,
                    ),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(
                        unet_out_channels,
                        sft_target_channels,
                        3,
                        stride=1,
                        padding=1,
                        bias=True,
                        bias_init_val=0,
                    ),
                )
            )

    def forward(
        self,
        x: Tensor,
        return_latents: bool = False,
        return_rgb: bool = True,
        randomize_noise: bool = True,
        **kwargs: Any,  # Allow additional kwargs for StyleGAN2GeneratorSFT if any
    ) -> Tuple[Tensor, List[Tensor] | None]:

        conditions: List[Tensor] = []
        unet_skips: List[Tensor] = []
        out_rgbs_list: List[Tensor] = []

        # U-Net Encoder
        feat: Tensor = self.conv_body_first(x)
        for i in range(self.log_size - 2):
            feat = self.conv_body_down[i](feat)
            unet_skips.insert(0, feat)
        feat = self.final_conv(feat)

        # Style code
        style_code: Tensor = self.final_linear(feat.view(feat.size(0), -1))
        if self.different_w:
            style_code = style_code.view(style_code.size(0), -1, self.num_style_feat)

        # U-Net Decoder & SFT condition generation
        for i in range(self.log_size - 2):
            feat = feat + unet_skips[i]
            feat = self.conv_body_up[i](feat)

            scale = self.condition_scale[i](feat)
            conditions.append(
                scale
            )  # .clone() if modification is a concern, but usually not needed here
            shift = self.condition_shift[i](feat)
            conditions.append(shift)

            if return_rgb:
                out_rgbs_list.append(self.toRGB[i](feat))

        # StyleGAN Decoder
        styles_for_decoder = [style_code]
        image, _ = self.stylegan_decoder(
            styles_for_decoder,
            conditions,
            return_latents=return_latents,
            input_is_latent=self.input_is_latent or self.different_w,
            randomize_noise=randomize_noise,
            **kwargs,  # Pass any other relevant kwargs to StyleGAN2GeneratorSFT
        )

        returned_rgbs = out_rgbs_list if return_rgb else None
        return image, returned_rgbs


@ARCH_REGISTRY.register()
class FacialComponentDiscriminator(nn.Module):
    """Facial component (eyes, mouth, nose) discriminator used in GFPGAN."""

    def __init__(
        self, resample_kernel: Sequence[int] = (1, 3, 3, 1)
    ):  # Added resample_kernel for ConvLayer
        super().__init__()  # Python 3 super()
        # VGG-style architecture
        self.conv1: ConvLayer = ConvLayer(
            3,
            64,
            3,
            downsample=False,
            resample_kernel=resample_kernel,
            bias=True,
            activate=True,
        )
        self.conv2: ConvLayer = ConvLayer(
            64,
            128,
            3,
            downsample=True,
            resample_kernel=resample_kernel,
            bias=True,
            activate=True,
        )
        self.conv3: ConvLayer = ConvLayer(
            128,
            128,
            3,
            downsample=False,
            resample_kernel=resample_kernel,
            bias=True,
            activate=True,
        )
        self.conv4: ConvLayer = ConvLayer(
            128,
            256,
            3,
            downsample=True,
            resample_kernel=resample_kernel,
            bias=True,
            activate=True,
        )
        self.conv5: ConvLayer = ConvLayer(
            256,
            256,
            3,
            downsample=False,
            resample_kernel=resample_kernel,
            bias=True,
            activate=True,
        )
        self.final_conv: ConvLayer = ConvLayer(
            256,
            1,
            3,
            downsample=False,
            resample_kernel=resample_kernel,
            bias=True,
            activate=False,
        )  # No activation on final layer

    def forward(
        self, x: Tensor, return_feats: bool = False, **kwargs: Any
    ) -> Tuple[Tensor, List[Tensor] | None]:
        feat: Tensor = self.conv1(x)
        feat = self.conv3(self.conv2(feat))

        rlt_feats: List[Tensor] = []
        if return_feats:
            rlt_feats.append(
                feat.clone()
            )  # Clone if features are modified later or to avoid graph issues

        feat = self.conv5(self.conv4(feat))
        if return_feats:
            rlt_feats.append(feat.clone())

        out: Tensor = self.final_conv(feat)

        if return_feats:
            return out, rlt_feats
        else:
            return out, None  # Maintain tuple structure for consistency
