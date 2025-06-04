import math
import random
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Tuple, Optional, Dict, Any, Literal  # Added Literal

from basicsr.utils.registry import ARCH_REGISTRY

# Relative import for the clean StyleGAN2 generator
from .stylegan2_clean_arch import StyleGAN2GeneratorClean


class StyleGAN2GeneratorCSFT(StyleGAN2GeneratorClean):
    """
    StyleGAN2 Generator with SFT modulation (Spatial Feature Transform).
    Clean version: uses PyTorch native ops, no custom CUDA extensions.

    Args:
        out_size (int): Spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Number of layers in MLP style network. Default: 8.
        channel_multiplier (int): Channel multiplier for StyleGAN2 network. Default: 2.
        narrow (float): Channel narrowing ratio. Default: 1.0.
        sft_half (bool): Apply SFT to half of the input channels. Default: False.
    """

    def __init__(
        self,
        out_size: int,
        num_style_feat: int = 512,
        num_mlp: int = 8,
        channel_multiplier: int = 2,
        narrow: float = 1.0,
        sft_half: bool = False,
    ):
        super().__init__(  # Python 3 super()
            out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
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
        """Forward function for StyleGAN2GeneratorCSFT. (Content is similar to other SFT generators)"""
        # Style codes -> latents
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

        # Latent injection
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

        # Main generation (attributes from StyleGAN2GeneratorClean)
        out: Tensor = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=actual_noise[0])
        skip: Tensor = self.to_rgb1(out, latent[:, 1])

        layer_idx: int = 1
        noise_idx: int = 1
        condition_idx: int = 0

        for conv1, conv2, to_rgb in zip(
            self.style_convs[::2], self.style_convs[1::2], self.to_rgbs
        ):
            current_noise1 = actual_noise[noise_idx]
            current_noise2 = actual_noise[noise_idx + 1]
            out = conv1(out, latent[:, layer_idx], noise=current_noise1)

            if condition_idx < len(
                conditions
            ):  # Check ensures we don't go out of bounds
                sft_scale = conditions[condition_idx]
                sft_shift = conditions[condition_idx + 1]

                if self.sft_half:
                    out_same, out_sft = torch.split(out, out.size(1) // 2, dim=1)
                    out_sft = out_sft * sft_scale + sft_shift
                    out = torch.cat([out_same, out_sft], dim=1)
                else:
                    out = out * sft_scale + sft_shift

            out = conv2(out, latent[:, layer_idx + 1], noise=current_noise2)
            skip = to_rgb(out, latent[:, layer_idx + 2], skip)

            layer_idx += 2
            noise_idx += 2
            condition_idx += 2

        image: Tensor = skip
        return (image, latent) if return_latents else (image, None)


class ResBlockClean(nn.Module):  # Renamed to avoid conflict if imported elsewhere
    """
    Residual block with bilinear upsampling/downsampling (Clean version).
    Uses standard nn.Conv2d and F.interpolate.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        mode (Literal['down', 'up']): Upsampling/downsampling mode. Default: 'down'.
        negative_slope (float): LeakyReLU negative slope. Default: 0.2.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["down", "up"] = "down",
        negative_slope: float = 0.2,
    ):
        super().__init__()  # Python 3 super()
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.skip: nn.Conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )  # 1x1 conv for skip

        self.negative_slope: float = negative_slope

        if mode == "down":
            self.scale_factor: float = 0.5
        elif mode == "up":
            self.scale_factor: float = 2.0
        else:
            raise ValueError(
                f"Unknown mode for ResBlockClean: {mode}. Choose 'up' or 'down'."
            )

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = F.leaky_relu(self.conv1(x), negative_slope=self.negative_slope)

        # Upsample/downsample main path
        out = F.interpolate(
            out, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )
        out = F.leaky_relu(self.conv2(out), negative_slope=self.negative_slope)

        # Upsample/downsample skip connection
        x_skip: Tensor = F.interpolate(
            x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )
        skip_connection: Tensor = self.skip(x_skip)

        return out + skip_connection  # Simple sum, original did not normalize


@ARCH_REGISTRY.register()
class GFPGANv1Clean(nn.Module):
    """
    The GFPGANv1Clean architecture: U-Net + StyleGAN2 Clean decoder with SFT.
    No custom CUDA extensions.
    """

    def __init__(
        self,
        out_size: int,
        num_style_feat: int = 512,
        channel_multiplier: int = 1,  # GFPGAN typically uses 1
        decoder_load_path: str | None = None,
        fix_decoder: bool = True,
        # For StyleGAN decoder
        num_mlp: int = 8,
        input_is_latent: bool = False,
        different_w: bool = False,
        narrow: float = 1.0,
        sft_half: bool = False,
        unet_leaky_slope: float = 0.2,  # Added slope for U-Net activations
    ):
        super().__init__()  # Python 3 super()
        self.input_is_latent: bool = input_is_latent
        self.different_w: bool = different_w
        self.num_style_feat: int = num_style_feat
        self.sft_half: bool = sft_half
        self.unet_leaky_slope: float = unet_leaky_slope

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

        first_conv_out_channels_key: str = str(out_size)
        self.conv_body_first: nn.Conv2d = nn.Conv2d(
            3, channels[first_conv_out_channels_key], kernel_size=1
        )

        # U-Net Encoder (Downsample)
        in_c: int = channels[first_conv_out_channels_key]
        self.conv_body_down: nn.ModuleList = nn.ModuleList()
        for i in range(self.log_size, 2, -1):
            out_c: int = channels[str(2 ** (i - 1))]
            self.conv_body_down.append(
                ResBlockClean(
                    in_c, out_c, mode="down", negative_slope=self.unet_leaky_slope
                )
            )
            in_c = out_c

        self.final_conv: nn.Conv2d = nn.Conv2d(
            in_c, channels["4"], kernel_size=3, stride=1, padding=1
        )

        # U-Net Decoder (Upsample)
        in_c = channels["4"]
        self.conv_body_up: nn.ModuleList = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_c = channels[str(2**i)]
            self.conv_body_up.append(
                ResBlockClean(
                    in_c, out_c, mode="up", negative_slope=self.unet_leaky_slope
                )
            )
            in_c = out_c

        # ToRGB layers
        self.toRGB: nn.ModuleList = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            self.toRGB.append(nn.Conv2d(channels[str(2**i)], 3, kernel_size=1))

        # Linear layer for style code
        if self.different_w:
            linear_out_channel: int = (self.log_size * 2 - 2) * num_style_feat
        else:
            linear_out_channel = num_style_feat
        self.final_linear: nn.Linear = nn.Linear(
            channels["4"] * 4 * 4, linear_out_channel
        )

        # StyleGAN2 Clean decoder with SFT
        self.stylegan_decoder: StyleGAN2GeneratorCSFT = StyleGAN2GeneratorCSFT(
            out_size=out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
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
                    )
                    print(
                        f"Loaded pre-trained StyleGAN2 (Clean) decoder from: {decoder_load_path} (key: {key_to_load})"
                    )
                else:
                    self.stylegan_decoder.load_state_dict(state_dict)
                    print(
                        f"Loaded pre-trained StyleGAN2 (Clean) decoder from: {decoder_load_path} (root)"
                    )
            except Exception as e:
                print(f"Error loading pre-trained StyleGAN2 (Clean) decoder: {e}.")

        if fix_decoder:
            for param in self.stylegan_decoder.parameters():
                param.requires_grad = False

        # SFT condition layers (using standard nn.Conv2d and nn.LeakyReLU)
        self.condition_scale: nn.ModuleList = nn.ModuleList()
        self.condition_shift: nn.ModuleList = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            unet_out_channels: int = channels[str(2**i)]
            # Get target channels from the decoder instance
            stylegan_res_channels: int = self.stylegan_decoder.channels[str(2**i)]

            sft_target_channels: int
            if self.sft_half:
                sft_target_channels = stylegan_res_channels // 2
            else:
                sft_target_channels = stylegan_res_channels

            self.condition_scale.append(
                nn.Sequential(
                    nn.Conv2d(unet_out_channels, unet_out_channels, 3, 1, 1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(unet_out_channels, sft_target_channels, 3, 1, 1),
                )
            )
            # Initialize last conv of scale to output something close to 1
            nn.init.constant_(self.condition_scale[-1][-1].bias, 1)  # type: ignore[arg-type] # last element of Sequential

            self.condition_shift.append(
                nn.Sequential(
                    nn.Conv2d(unet_out_channels, unet_out_channels, 3, 1, 1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(unet_out_channels, sft_target_channels, 3, 1, 1),
                )
            )
            # Initialize last conv of shift to output something close to 0 (default for bias is 0)

    def forward(
        self,
        x: Tensor,
        return_latents: bool = False,
        return_rgb: bool = True,
        randomize_noise: bool = True,
        **kwargs: Any,  # For StyleGAN2GeneratorCSFT if it has other args
    ) -> Tuple[Tensor, List[Tensor] | None]:

        conditions: List[Tensor] = []
        unet_skips: List[Tensor] = []
        out_rgbs_list: List[Tensor] = []

        # U-Net Encoder
        feat: Tensor = F.leaky_relu(
            self.conv_body_first(x), negative_slope=self.unet_leaky_slope
        )
        for i in range(self.log_size - 2):
            feat = self.conv_body_down[i](
                feat
            )  # ResBlockClean includes its own activations
            unet_skips.insert(0, feat)
        feat = F.leaky_relu(self.final_conv(feat), negative_slope=self.unet_leaky_slope)

        # Style code
        style_code: Tensor = self.final_linear(feat.view(feat.size(0), -1))
        if self.different_w:
            style_code = style_code.view(style_code.size(0), -1, self.num_style_feat)

        # U-Net Decoder & SFT condition generation
        for i in range(self.log_size - 2):
            feat = feat + unet_skips[i]
            feat = self.conv_body_up[i](
                feat
            )  # ResBlockClean includes its own activations

            scale = self.condition_scale[i](feat)
            conditions.append(scale)
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
            **kwargs,
        )

        returned_rgbs = out_rgbs_list if return_rgb else None
        return image, returned_rgbs
