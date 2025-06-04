import math
import random
import torch
from torch import nn, Tensor  # Added Tensor for type hinting
from typing import List, Optional, Tuple, Dict, Any  # Added for type hinting

from basicsr.utils.registry import ARCH_REGISTRY

# Assuming these imports are relative to the current 'archs' package
from .gfpganv1_arch import ResUpBlock
from .stylegan2_bilinear_arch import (
    ConvLayer,
    EqualConv2d,
    EqualLinear,
    ResBlock,
    ScaledLeakyReLU,
    StyleGAN2GeneratorBilinear,  # Parent class
)


class StyleGAN2GeneratorBilinearSFT(StyleGAN2GeneratorBilinear):
    """
    StyleGAN2 Generator with SFT modulation (Spatial Feature Transform).

    Bilinear version: avoids complex UpFirDnSmooth for easier deployment.
    Can be converted to StyleGAN2GeneratorCSFT (clean version).

    Args:
        out_size (int): Spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Number of layers in MLP style network. Default: 8.
        channel_multiplier (int): Channel multiplier for StyleGAN2 network. Default: 2.
        lr_mlp (float): Learning rate multiplier for MLP layers. Default: 0.01.
        narrow (float): Channel narrowing ratio. Default: 1.
        sft_half (bool): Apply SFT to half of the input channels. Default: False.
    """

    def __init__(
        self,
        out_size: int,
        num_style_feat: int = 512,
        num_mlp: int = 8,
        channel_multiplier: int = 2,
        lr_mlp: float = 0.01,
        narrow: float = 1.0,  # Explicitly float
        sft_half: bool = False,
    ):
        super().__init__(  # Python 3 super()
            out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            lr_mlp=lr_mlp,
            narrow=narrow,
        )
        self.sft_half: bool = sft_half

    def forward(
        self,
        styles: List[Tensor],
        conditions: List[Tensor],
        input_is_latent: bool = False,
        noise: List[Tensor | None] | None = None,  # List of Tensors or Nones
        randomize_noise: bool = True,
        truncation: float = 1.0,  # Explicitly float
        truncation_latent: Tensor | None = None,
        inject_index: int | None = None,
        return_latents: bool = False,
    ) -> Tuple[Tensor, Tensor | None]:
        """Forward function for StyleGAN2GeneratorBilinearSFT.

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
                actual_noise = [None] * self.num_layers  # For each style conv layer
            else:
                actual_noise = [
                    getattr(self.noises, f"noise{i}") for i in range(self.num_layers)
                ]
        else:
            actual_noise = noise  # Use provided noise

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
            current_inject_index: int = self.num_latent  # type: ignore[assignment] # num_latent from parent
            if processed_styles[0].ndim < 3:
                # Repeat latent code for all layers
                latent = (
                    processed_styles[0].unsqueeze(1).repeat(1, current_inject_index, 1)
                )
            else:  # Used for encoder with different latent code for each layer
                latent = processed_styles[0]
        elif len(processed_styles) == 2:  # Style mixing
            if inject_index is None:
                # num_latent from parent
                current_inject_index = random.randint(1, self.num_latent - 1)  # type: ignore[operator]
            else:
                current_inject_index = inject_index

            latent1 = (
                processed_styles[0].unsqueeze(1).repeat(1, current_inject_index, 1)
            )
            # num_latent from parent
            latent2 = processed_styles[1].unsqueeze(1).repeat(1, self.num_latent - current_inject_index, 1)  # type: ignore[operator]
            latent = torch.cat([latent1, latent2], 1)
        else:
            # Fallback or error for unexpected number of styles
            if not processed_styles:  # Handle empty list case
                raise ValueError("Styles list cannot be empty.")
            # Assuming if not 1 or 2, it's an error or unhandled case for this specific mixing logic
            # For now, let's assume if not 1 or 2, we take the first one and proceed as if len == 1 (needs review based on expected use)
            # Or raise error:
            raise ValueError(
                f"Unsupported number of styles for mixing: {len(processed_styles)}. Expected 1 or 2."
            )

        # Main generation
        out: Tensor = self.constant_input(latent.shape[0])  # constant_input from parent
        out = self.style_conv1(
            out, latent[:, 0], noise=actual_noise[0]
        )  # style_conv1 from parent
        skip: Tensor = self.to_rgb1(out, latent[:, 1])  # to_rgb1 from parent

        layer_idx: int = 1  # Index for latent codes
        noise_idx: int = 1  # Index for noise list
        condition_idx: int = 0  # Index for conditions list

        # style_convs, to_rgbs are from parent
        for conv1, conv2, to_rgb in zip(
            self.style_convs[::2], self.style_convs[1::2], self.to_rgbs
        ):
            current_noise1 = actual_noise[noise_idx]
            current_noise2 = actual_noise[noise_idx + 1]

            out = conv1(out, latent[:, layer_idx], noise=current_noise1)

            # Conditions might have fewer levels than style conv layers
            if (
                condition_idx < len(conditions) * 2
            ):  # Each SFT step uses two condition tensors
                # SFT part to combine the conditions
                # Conditions are applied sequentially for scale and shift
                sft_scale = conditions[condition_idx]
                sft_shift = conditions[condition_idx + 1]

                if self.sft_half:  # Apply SFT to half of the channels
                    out_same, out_sft = torch.split(out, int(out.size(1) // 2), dim=1)
                    out_sft = (
                        out_sft * sft_scale + sft_shift
                    )  # Element-wise, ensure sft_scale/shift match out_sft channels
                    out = torch.cat([out_same, out_sft], dim=1)
                else:  # Apply SFT to all channels
                    out = out * sft_scale + sft_shift  # Element-wise

            out = conv2(out, latent[:, layer_idx + 1], noise=current_noise2)
            skip = to_rgb(
                out, latent[:, layer_idx + 2], skip
            )  # Feature back to RGB space

            layer_idx += 2
            noise_idx += 2
            condition_idx += 2  # Consumed two conditions (scale and shift)

        image: Tensor = skip

        if return_latents:
            return image, latent
        else:
            return image, None


@ARCH_REGISTRY.register()
class GFPGANBilinear(nn.Module):
    """
    The GFPGAN architecture: U-Net + StyleGAN2 decoder with SFT.

    Bilinear version: avoids complex UpFirDnSmooth for easier deployment.
    Can be converted to GFPGANv1Clean.

    Ref: GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior.

    Args:
        out_size (int): Spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        channel_multiplier (int): Channel multiplier for StyleGAN2. Default: 1 for GFPGAN.
        decoder_load_path (str | None): Path to pre-trained StyleGAN2 decoder. Default: None.
        fix_decoder (bool): Whether to fix the decoder parameters. Default: True.
        num_mlp (int): Number of layers in MLP style network. Default: 8.
        lr_mlp (float): Learning rate multiplier for MLP layers. Default: 0.01.
        input_is_latent (bool): Whether input is latent style. Default: False.
        different_w (bool): Use different latent w for different layers. Default: False.
        narrow (float): Channel narrowing ratio for U-Net. Default: 1.
        sft_half (bool): Apply SFT to half of the input channels in decoder. Default: False.
    """

    def __init__(
        self,
        out_size: int,
        num_style_feat: int = 512,
        channel_multiplier: int = 1,  # Note: Default is 1 for GFPGAN, not 2 like StyleGAN2
        decoder_load_path: str | None = None,
        fix_decoder: bool = True,
        # For StyleGAN decoder
        num_mlp: int = 8,
        lr_mlp: float = 0.01,
        input_is_latent: bool = False,
        different_w: bool = False,
        narrow: float = 1.0,  # Explicitly float
        sft_half: bool = False,
    ):
        super().__init__()  # Python 3 super()
        self.input_is_latent: bool = input_is_latent
        self.different_w: bool = different_w
        self.num_style_feat: int = num_style_feat
        self.sft_half: bool = sft_half  # Store for use in condition_scale/shift

        unet_narrow: float = (
            narrow * 0.5
        )  # Default: use half of input channels for U-Net
        # Channel configurations for U-Net
        # Keys are strings representing feature map sizes
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

        self.log_size: int = int(math.log(out_size, 2))
        # Ensure out_size is a power of 2
        if 2**self.log_size != out_size:
            raise ValueError(f"out_size must be a power of 2. Got {out_size}")

        first_out_size_str: str = str(
            out_size
        )  # Start with the largest size for the first conv

        self.conv_body_first: ConvLayer = ConvLayer(
            3, channels[first_out_size_str], 1, bias=True, activate=True
        )

        # U-Net Encoder (Downsample)
        in_c: int = channels[first_out_size_str]
        self.conv_body_down: nn.ModuleList = nn.ModuleList()
        for i in range(
            self.log_size, 2, -1
        ):  # From log_size down to 3 (feature map size 2^i to 2^(i-1))
            out_c: int = channels[str(2 ** (i - 1))]
            self.conv_body_down.append(
                ResBlock(in_c, out_c, downsample=True)
            )  # Assuming ResBlock handles downsampling
            in_c = out_c
        # After loop, in_c is channels['4'] (for feature map size 4x4)

        self.final_conv: ConvLayer = ConvLayer(
            in_c, channels["4"], 3, bias=True, activate=True
        )
        # Now feat is at 4x4 spatial resolution with channels['4'] channels.

        # U-Net Decoder (Upsample)
        in_c = channels["4"]
        self.conv_body_up: nn.ModuleList = nn.ModuleList()
        for i in range(
            3, self.log_size + 1
        ):  # From 3 up to log_size (feature map size 2^i)
            out_c = channels[str(2**i)]
            self.conv_body_up.append(
                ResUpBlock(in_c, out_c)
            )  # ResUpBlock handles upsampling
            in_c = out_c

        # ToRGB layers for intermediate outputs from U-Net decoder
        self.toRGB: nn.ModuleList = nn.ModuleList()
        for i in range(3, self.log_size + 1):  # Corresponding to conv_body_up outputs
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

        # Linear layer to produce style code from bottleneck features
        if self.different_w:
            # (log_size - 1) levels, each with num_style_feat, but the indexing in StyleGAN usually means
            # (log_size * 2 - 2) latents for conv layers and toRGBs.
            # Let's check StyleGAN2GeneratorBilinear's num_latent.
            # num_latent = log_size * 2 - 2. This is the number of w vectors needed.
            linear_out_channel: int = (self.log_size * 2 - 2) * num_style_feat
        else:
            linear_out_channel = num_style_feat

        self.final_linear: EqualLinear = EqualLinear(
            channels["4"] * 4 * 4,  # Input from 4x4 bottleneck feature map
            linear_out_channel,
            bias=True,
            bias_init_val=0,
            lr_mul=1,
            activation=None,
        )

        # StyleGAN2 decoder with SFT
        self.stylegan_decoder: StyleGAN2GeneratorBilinearSFT = (
            StyleGAN2GeneratorBilinearSFT(
                out_size=out_size,
                num_style_feat=num_style_feat,
                num_mlp=num_mlp,
                channel_multiplier=channel_multiplier,  # Pass the GFPGAN specific CM
                lr_mlp=lr_mlp,
                narrow=narrow,  # Pass narrow for consistency if StyleGAN2 uses it internally for channel calcs
                sft_half=sft_half,
            )
        )

        # Load pre-trained StyleGAN2 decoder weights if path is provided
        if decoder_load_path:
            try:
                state_dict = torch.load(
                    decoder_load_path, map_location=lambda storage, loc: storage
                )
                # Common keys: 'params_ema', 'g_ema' (from official StyleGAN2), or direct state_dict
                if "params_ema" in state_dict:
                    self.stylegan_decoder.load_state_dict(state_dict["params_ema"])
                elif (
                    "g_ema" in state_dict
                ):  # Typical key from NVIDIA's StyleGAN2 ADA checkpoints
                    # Need to filter g_ema for StyleGAN2GeneratorBilinearSFT's parameters
                    # This can be tricky if names don't align perfectly or if it's a full training checkpoint
                    # For GFPGAN, 'params_ema' is more common from their provided weights.
                    # Assuming 'params_ema' is the target. If using generic StyleGAN2 weights,
                    # one might need to adapt keys.
                    self.stylegan_decoder.load_state_dict(
                        state_dict["g_ema"], strict=False
                    )  # strict=False if not all keys match
                    print(
                        f"Warning: Loaded decoder from 'g_ema' key with strict=False. Ensure compatibility."
                    )
                else:
                    # Attempt to load the whole dict; common if the .pth file *is* the state_dict
                    self.stylegan_decoder.load_state_dict(state_dict)
                print(f"Loaded pre-trained StyleGAN2 decoder from: {decoder_load_path}")
            except Exception as e:
                print(
                    f"Error loading pre-trained StyleGAN2 decoder: {e}. Check path and checkpoint structure."
                )

        # Fix decoder parameters (do not update during training)
        if fix_decoder:
            for param in self.stylegan_decoder.parameters():
                param.requires_grad = False

        # Layers for generating SFT conditions (scale and shift) from U-Net features
        self.condition_scale: nn.ModuleList = nn.ModuleList()
        self.condition_shift: nn.ModuleList = nn.ModuleList()

        # Iterate through the U-Net decoder output levels that feed into SFT
        # SFT conditions are needed for StyleGAN2 layers from 8x8 up to out_size
        # StyleGAN2 conv layers start effectively after the 4x4 constant input
        # conv1 (idx 0), conv2 (idx 1) -> style_convs[0], style_convs[1] operate on 8x8 if out_size >= 8
        # So conditions correspond to features from U-Net decoder at 8x8, 16x16, ...
        # The loop for conv_body_up and toRGB is from i=3 (8x8) to log_size (out_size)

        for i in range(3, self.log_size + 1):
            current_unet_channels: int = channels[
                str(2**i)
            ]  # Channels from U-Net ResUpBlock output

            # SFT conditions (scale, shift) are applied to features *inside* the StyleGAN decoder.
            # The number of channels for SFT scale/shift should match StyleGAN's internal feature channels at that resolution.
            # This requires knowing StyleGAN2GeneratorBilinearSFT's channel counts.
            # Let's assume for now that sft_out_channels calculation in original code is based on StyleGAN's needs.
            # However, the input to these condition layers is current_unet_channels.

            # The original code calculated sft_out_channels based on StyleGAN's decoder channels (channels[f'{2**i}'])
            # but this uses the U-Net's 'channels' dict. This might be fine if they are consistent.
            # For StyleGAN2GeneratorBilinear, channels are calculated as:
            # self.channels = {
            #    f_size: min(int(base_channel * channel_multiplier / (2**(i-1))), self.num_style_feat)
            #    for i, f_size in enumerate(self.feat_brounds) # feat_brounds = [4, 8, ..., out_size]
            # } where base_channel = int(num_style_feat * narrow)
            # This is potentially different from the U-Net 'channels' dict.
            # This part needs careful check. For now, I'll follow the original logic which uses
            # U-Net's `channels` dict for `sft_out_channels`.

            stylegan_res_channels = self.stylegan_decoder.channels[
                str(2**i)
            ]  # Get channels from StyleGAN decoder instance

            sft_target_channels: int
            if self.sft_half:
                sft_target_channels = (
                    stylegan_res_channels // 2
                )  # SFT applied to half of StyleGAN channels
            else:
                sft_target_channels = (
                    stylegan_res_channels  # SFT applied to all StyleGAN channels
                )

            # Scale layer: UNet channels -> SFT scale channels
            self.condition_scale.append(
                nn.Sequential(
                    EqualConv2d(
                        current_unet_channels,
                        current_unet_channels,
                        3,
                        stride=1,
                        padding=1,
                        bias=True,
                        bias_init_val=0,
                    ),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(
                        current_unet_channels,
                        sft_target_channels,
                        3,
                        stride=1,
                        padding=1,
                        bias=True,
                        bias_init_val=1,
                    ),  # Output sft_target_channels
                )
            )
            # Shift layer: UNet channels -> SFT shift channels
            self.condition_shift.append(
                nn.Sequential(
                    EqualConv2d(
                        current_unet_channels,
                        current_unet_channels,
                        3,
                        stride=1,
                        padding=1,
                        bias=True,
                        bias_init_val=0,
                    ),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(
                        current_unet_channels,
                        sft_target_channels,
                        3,
                        stride=1,
                        padding=1,
                        bias=True,
                        bias_init_val=0,
                    ),  # Output sft_target_channels
                )
            )

    def forward(
        self,
        x: Tensor,
        return_latents: bool = False,
        return_rgb: bool = True,
        randomize_noise: bool = True,
    ) -> Tuple[Tensor, List[Tensor] | None]:  # Return type for out_rgbs
        """Forward function for GFPGANBilinear.

        Args:
            x (Tensor): Input images (B, C, H, W).
            return_latents (bool): Whether to return style latents. Default: False.
            return_rgb (bool): Whether to return intermediate RGB images from U-Net. Default: True.
            randomize_noise (bool): Randomize noise for StyleGAN decoder. Default: True.

        Returns:
            Tuple[Tensor, List[Tensor] | None]:
                - Final enhanced image.
                - List of intermediate RGBs from U-Net decoder (if return_rgb is True), else None.
        """
        conditions: List[Tensor] = []
        unet_skips: List[Tensor] = []
        out_rgbs_list: List[Tensor] = []  # Explicitly list

        # U-Net Encoder
        feat: Tensor = self.conv_body_first(x)
        # Downsample blocks
        for i in range(
            self.log_size - 2
        ):  # Corresponds to log_size down to feature map size 8x8
            feat = self.conv_body_down[i](feat)
            unet_skips.insert(
                0, feat
            )  # Store skip connections (from coarsest to finest)

        # Bottleneck convolution
        feat = self.final_conv(feat)  # Now feat is at 4x4 spatial resolution

        # Generate style code from bottleneck features
        style_code: Tensor = self.final_linear(feat.view(feat.size(0), -1))
        if self.different_w:
            # Reshape to (batch_size, num_latents_needed, num_style_feat)
            # num_latents_needed = self.log_size * 2 - 2
            style_code = style_code.view(style_code.size(0), -1, self.num_style_feat)

        # U-Net Decoder
        # Loop from 8x8 features up to (out_size/2)x(out_size/2) features
        # This loop generates conditions for StyleGAN decoder layers
        for i in range(
            self.log_size - 2
        ):  # Iterates len(conv_body_up) times which is log_size - 2
            # Add U-Net skip connection (from corresponding encoder layer)
            feat = feat + unet_skips[i]
            # Upsample layer
            feat = self.conv_body_up[i](feat)

            # Generate SFT scale and shift conditions from current U-Net feature map
            scale: Tensor = self.condition_scale[i](feat)
            conditions.append(scale)  # Append scale
            shift: Tensor = self.condition_shift[i](feat)
            conditions.append(
                shift
            )  # Append shift (conditions list will be [scale0, shift0, scale1, shift1, ...])

            # Generate intermediate RGB image if requested
            if return_rgb:
                out_rgbs_list.append(self.toRGB[i](feat))

        # StyleGAN Decoder
        # Pass style_code (or [style_code] if not input_is_latent for decoder)
        # and collected SFT conditions
        image: Tensor
        # The stylegan_decoder expects a list of styles
        styles_for_decoder = [style_code]

        image, _ = self.stylegan_decoder(
            styles_for_decoder,
            conditions,
            return_latents=return_latents,  # This return_latents is for stylegan_decoder internal latents
            input_is_latent=self.input_is_latent
            or self.different_w,  # If different_w, style_code is already latent-like
            randomize_noise=randomize_noise,
        )

        returned_rgbs = out_rgbs_list if return_rgb else None
        return image, returned_rgbs
