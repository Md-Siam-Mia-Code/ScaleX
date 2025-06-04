import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, List, Dict, Any, Sequence


class VectorQuantizer(nn.Module):

    def __init__(self, n_e: int, e_dim: int, beta: float):
        super().__init__() # Python 3 super()
        self.n_e: int = n_e
        self.e_dim: int = e_dim
        self.beta: float = beta

        self.embedding: nn.Embedding = nn.Embedding(self.n_e, self.e_dim)
        # Initialize embedding weights uniformly
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:

        # Reshape z: (B, C, H, W) -> (B, H, W, C) and flatten
        z_permuted: Tensor = z.permute(0, 2, 3, 1).contiguous()
        z_flattened: Tensor = z_permuted.view(-1, self.e_dim)

        # Calculate distances: (z - e)^2 = z^2 + e^2 - 2*z*e
        d: Tensor = (
            torch.sum(z_flattened**2, dim=1, keepdim=True) +
            torch.sum(self.embedding.weight**2, dim=1) - # dim=1 sums over e_dim for each embedding
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
        ) # d shape: (B*H*W, n_e)

        # Find closest encodings
        min_encoding_indices: Tensor = torch.argmin(d, dim=1)
        
        # Simpler way to get z_q using the indices from argmin:
        z_q_flat: Tensor = self.embedding(min_encoding_indices) # Shape: (B*H*W, e_dim)
        z_q: Tensor = z_q_flat.view(z_permuted.shape) # Reshape to (B, H, W, C)

        # One-hot encodings (needed for perplexity calculation)
        min_encodings_one_hot: Tensor = F.one_hot(min_encoding_indices, num_classes=self.n_e).float()


        # Compute loss for embedding (VQ objective)
        # detach() makes the operation non-differentiable w.r.t. the detached tensor in that part of the loss
        codebook_loss: Tensor = torch.mean((z_q.detach() - z_permuted)**2) # Moves z_q towards z
        commitment_loss: Tensor = torch.mean((z_q - z_permuted.detach())**2) # Moves z towards z_q
        loss: Tensor = codebook_loss + self.beta * commitment_loss

        # Preserve gradients (straight-through estimator)
        z_q = z_permuted + (z_q - z_permuted).detach()

        # Perplexity calculation
        e_mean: Tensor = torch.mean(min_encodings_one_hot, dim=0) # Average usage of each codebook vector
        perplexity: Tensor = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # Reshape z_q back to original input shape (B, C, H, W)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        # For consistency with original return, use unsqueezed indices
        min_encoding_indices_ret = min_encoding_indices.unsqueeze(1)

        return z_q, loss, (perplexity, min_encodings_one_hot, min_encoding_indices_ret, d)

    def get_codebook_entry(self, indices: Tensor, shape: Optional[Tuple[int, int, int, int]] = None) -> Tensor:
        z_q_flat: Tensor = self.embedding(indices) # Shape: (N, e_dim)

        if shape is not None:
            # Reshape to (Batch, Height, Width, Channel)
            z_q: Tensor = z_q_flat.view(shape[0], shape[1], shape[2], self.e_dim) # Assuming shape is B,H,W,C
            # Permute back to (Batch, Channel, Height, Width)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
            return z_q
        else:
            return z_q_flat


# SiLU (Swish) activation function
def silu_activation(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x) # or F.silu(x) for PyTorch 1.7+


class Normalize(nn.Module):
    """Group Normalization layer."""
    def __init__(self, in_channels: int, num_groups: int = 32):
        super().__init__()
        self.norm: nn.GroupNorm = nn.GroupNorm(
            num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)


class Upsample(nn.Module):
    """Upsampling layer with optional convolution."""
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv: bool = with_conv
        if self.with_conv:
            self.conv: nn.Conv2d = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Downsampling layer with optional convolution."""
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv: bool = with_conv
        if self.with_conv:
            
            self.conv: nn.Conv2d = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=1 # Standard downsampling conv
            )
        else:
            # For avg_pool2d, kernel_size=2, stride=2 directly downsamples by 2
            pass


    def forward(self, x: Tensor) -> Tensor:
        if self.with_conv:
            # Original padding logic:
            # pad = (0, 1, 0, 1) # Pad last dim (width) by (0,1), then 2nd to last (height) by (0,1)
            # x = F.pad(x, pad, mode='constant', value=0)
            # x = self.conv(x) # self.conv was padding=0
            # Simplified if using padding=1 in self.conv:
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    """Residual block with optional time/conditional embedding projection."""
    def __init__(
        self,
        *, # Make in_channels and out_channels keyword-only for clarity
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float,
        temb_channels: int = 512, # Time embedding channels
    ):
        super().__init__()
        self.in_channels: int = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels: int = out_channels
        self.use_conv_shortcut: bool = conv_shortcut
        self.temb_channels: int = temb_channels

        self.norm1: Normalize = Normalize(in_channels)
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.temb_channels > 0:
            self.temb_proj: nn.Linear = nn.Linear(temb_channels, out_channels)
        
        self.norm2: Normalize = Normalize(out_channels)
        self.dropout_layer: nn.Dropout = nn.Dropout(dropout) # Renamed from self.dropout
        self.conv2: nn.Conv2d = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        
        self.conv_shortcut_layer: Optional[nn.Conv2d] = None
        self.nin_shortcut_layer: Optional[nn.Conv2d] = None
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut_layer = nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut_layer = nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x: Tensor, temb: Optional[Tensor]) -> Tensor:
        h_res: Tensor = x # Store original x for residual connection

        h: Tensor = self.norm1(x)
        h = silu_activation(h)
        h = self.conv1(h)

        if temb is not None and self.temb_channels > 0:
            # Project time embedding and add to feature map
            # Original: h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
            # Assuming temb is (B, temb_channels), nonlinearity(temb) is (B, temb_channels)
            # self.temb_proj maps (B, temb_channels) to (B, out_channels)
            # Then unsqueeze to (B, out_channels, 1, 1) for broadcasting
            h = h + self.temb_proj(silu_activation(temb)).unsqueeze(2).unsqueeze(3)

        h = self.norm2(h)
        h = silu_activation(h)
        h = self.dropout_layer(h)
        h = self.conv2(h)

        # Apply shortcut connection
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut and self.conv_shortcut_layer:
                h_res = self.conv_shortcut_layer(h_res)
            elif self.nin_shortcut_layer: # Implies not use_conv_shortcut
                h_res = self.nin_shortcut_layer(h_res)
        
        return h_res + h


class MultiHeadAttnBlock(nn.Module):
    """Multi-head attention block."""
    def __init__(self, in_channels: int, num_heads: int = 1): # Renamed head_size to num_heads
        super().__init__()
        self.in_channels: int = in_channels
        self.num_heads: int = num_heads
        if in_channels % num_heads != 0:
             raise ValueError(f"in_channels ({in_channels}) must be divisible by num_heads ({num_heads}).")
        self.head_dim: int = in_channels // num_heads # Dimension per head

        self.norm_query: Normalize = Normalize(in_channels) # For query (y)
        self.norm_key_value: Normalize = Normalize(in_channels) # For key/value (x)

        self.q_conv: nn.Conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k_conv: nn.Conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v_conv: nn.Conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out: nn.Conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x_kv: Tensor, y_q: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x_kv (Tensor): Input tensor for Key and Value.
            y_q (Tensor, optional): Input tensor for Query. If None, x_kv is used for Query (self-attention).
                                     Defaults to None.
        """
        b, c, h, w = x_kv.shape
        
        h_kv_norm: Tensor = self.norm_key_value(x_kv)
        k: Tensor = self.k_conv(h_kv_norm) # (B, C, H, W)
        v: Tensor = self.v_conv(h_kv_norm) # (B, C, H, W)

        if y_q is None:
            y_q_norm: Tensor = h_kv_norm # Self-attention, reuse normalized x_kv
        else:
            y_q_norm = self.norm_query(y_q)
        q: Tensor = self.q_conv(y_q_norm) # (B, C, H, W) (or shape of y_q)

        # Reshape for multi-head attention
        # (B, C, H, W) -> (B, num_heads, head_dim, H*W) -> (B, num_heads, H*W, head_dim) for q,v
        #                                               -> (B, num_heads, head_dim, H*W) for k
        q = q.view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2) # (B, num_heads, H*W, head_dim)
        k = k.view(b, self.num_heads, self.head_dim, h * w)                     # (B, num_heads, head_dim, H*W)
        v = v.view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2) # (B, num_heads, H*W, head_dim)

        # Attention: Q * K^T
        # (B, num_heads, H*W, head_dim) @ (B, num_heads, head_dim, H*W) -> (B, num_heads, H*W, H*W)
        attention_scores: Tensor = torch.matmul(q, k) * (self.head_dim**-0.5) # Scale
        attention_weights: Tensor = F.softmax(attention_scores, dim=-1)

        # Weighted sum: Attention_Weights * V
        # (B, num_heads, H*W, H*W) @ (B, num_heads, H*W, head_dim) -> (B, num_heads, H*W, head_dim)
        out: Tensor = torch.matmul(attention_weights, v)

        # Reshape back: (B, num_heads, H*W, head_dim) -> (B, H*W, num_heads, head_dim) -> (B, C, H, W)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, c, h, w)
        out = self.proj_out(out)

        return x_kv + out # Residual connection (original was x + w_)


# Helper nn.Module for cleaner MultiHeadEncoder/Decoder
class ResAttnBlockDown(nn.Module):
    def __init__(self, in_channels_block: int, out_channels_block: int, temb_channels: int, dropout: float,
                 curr_res: int, attn_resolutions: Sequence[int], num_heads: int,
                 is_last_level: bool, resamp_with_conv: bool):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList() # Changed from attn, matching nn.ModuleList convention

        current_block_in_channels = in_channels_block
        for _ in range(num_res_blocks): # num_res_blocks from outer scope
            self.res_blocks.append(
                ResnetBlock(
                    in_channels=current_block_in_channels,
                    out_channels=out_channels_block,
                    temb_channels=temb_channels,
                    dropout=dropout
                )
            )
            current_block_in_channels = out_channels_block # Update for next res block
            if curr_res in attn_resolutions:
                self.attn_blocks.append(MultiHeadAttnBlock(current_block_in_channels, num_heads))
            else: # Keep lists aligned if no attn block
                self.attn_blocks.append(nn.Identity())


        self.downsample_layer: Optional[Downsample] = None
        if not is_last_level:
            self.downsample_layer = Downsample(current_block_in_channels, resamp_with_conv)

    def forward(self, h: Tensor, temb: Optional[Tensor]) -> Tensor:
        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            h = res_block(h, temb)
            h = attn_block(h) # Attn block is self-attention (y_q=None)
        if self.downsample_layer:
            h = self.downsample_layer(h)
        return h

# Helper nn.Module for cleaner MultiHeadEncoder/Decoder
class ResAttnBlockUp(nn.Module):
    def __init__(self, in_channels_block: int, out_channels_block: int, temb_channels: int, dropout: float,
                 curr_res: int, attn_resolutions: Sequence[int], num_heads: int,
                 is_not_first_level: bool, resamp_with_conv: bool,
                 use_cross_attention: bool = False # For Transformer variant
                 ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        self.use_cross_attention = use_cross_attention

        current_block_in_channels = in_channels_block
        # num_res_blocks + 1 as in original
        for _ in range(num_res_blocks + 1): # num_res_blocks from outer scope
            self.res_blocks.append(
                ResnetBlock(
                    in_channels=current_block_in_channels,
                    out_channels=out_channels_block,
                    temb_channels=temb_channels,
                    dropout=dropout
                )
            )
            current_block_in_channels = out_channels_block
            if curr_res in attn_resolutions:
                self.attn_blocks.append(MultiHeadAttnBlock(current_block_in_channels, num_heads))
            else:
                self.attn_blocks.append(nn.Identity())

        self.upsample_layer: Optional[Upsample] = None
        if is_not_first_level: # Corresponds to i_level != 0 in original
            self.upsample_layer = Upsample(current_block_in_channels, resamp_with_conv)

    def forward(self, h: Tensor, temb: Optional[Tensor], cross_attn_hs: Optional[Tensor] = None) -> Tensor:
        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            h = res_block(h, temb)
            if self.use_cross_attention and isinstance(attn_block, MultiHeadAttnBlock):
                # If cross_attn_hs is provided, use it for query (y_q), h is key/value (x_kv)
                h = attn_block(x_kv=h, y_q=cross_attn_hs)
            elif isinstance(attn_block, MultiHeadAttnBlock): # Self-attention
                 h = attn_block(x_kv=h)
            # If attn_block is nn.Identity, it does nothing
        if self.upsample_layer:
            h = self.upsample_layer(h)
        return h


class MultiHeadEncoder(nn.Module):
    """Multi-head U-Net Encoder."""
    def __init__(
        self,
        *,
        ch: int, # Base channel count
        out_ch: int, # Output channels of the final layer (not used by encoder directly)
        ch_mult: Sequence[int] = (1, 2, 4, 8), # Channel multipliers per resolution
        num_res_blocks: int = 2,
        attn_resolutions: Sequence[int] = (16,), # Resolutions at which to apply attention
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int = 3, # Input image channels
        resolution: int = 512, # Input image resolution
        z_channels: int = 256, # Channels of the bottleneck z
        double_z: bool = True, # If True, output 2*z_channels (for mean and logvar in VAE)
        enable_mid_block: bool = True, # enable_mid in original
        num_heads: int = 1, # head_size in original
        **ignore_kwargs: Any
    ):
        super().__init__()
        self.ch: int = ch
        self.temb_ch: int = 0 # No time embedding in this version
        self.num_resolutions: int = len(ch_mult)
        self.num_res_blocks: int = num_res_blocks
        self.resolution: int = resolution
        self.in_channels: int = in_channels
        self.enable_mid_block: bool = enable_mid_block

        self.conv_in: nn.Conv2d = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res: int = resolution
        in_ch_multiplier: Sequence[int] = (1,) + tuple(ch_mult) # Prepend 1 for initial ch
        self.down_blocks: nn.ModuleList = nn.ModuleList() # Renamed from self.down
        
        current_block_input_ch = self.ch # After conv_in
        for i_level in range(self.num_resolutions):
            block_output_ch = ch * ch_mult[i_level]
            is_last = (i_level == self.num_resolutions - 1)
            
            level_block = ResAttnBlockDown(
                in_channels_block=current_block_input_ch,
                out_channels_block=block_output_ch,
                temb_channels=self.temb_ch,
                dropout=dropout,
                curr_res=curr_res,
                attn_resolutions=attn_resolutions,
                num_heads=num_heads,
                is_last_level=is_last,
                resamp_with_conv=resamp_with_conv
            )
            self.down_blocks.append(level_block)
            current_block_input_ch = block_output_ch # Input for next level or mid block
            if not is_last:
                curr_res //= 2
        
        # Middle block
        self.mid_block: Optional[nn.ModuleDict] = None # Renamed from self.mid
        if self.enable_mid_block:
            self.mid_block = nn.ModuleDict({
                'block_1': ResnetBlock(in_channels=current_block_input_ch, out_channels=current_block_input_ch,
                                       temb_channels=self.temb_ch, dropout=dropout),
                'attn_1': MultiHeadAttnBlock(current_block_input_ch, num_heads),
                'block_2': ResnetBlock(in_channels=current_block_input_ch, out_channels=current_block_input_ch,
                                       temb_channels=self.temb_ch, dropout=dropout)
            })

        # Output layers
        self.norm_out: Normalize = Normalize(current_block_input_ch)
        self.conv_out: nn.Conv2d = nn.Conv2d(
            current_block_input_ch,
            2 * z_channels if double_z else z_channels,
            kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # hs (hidden states) dictionary to store intermediate features for skip connections
        hs: Dict[str, Tensor] = {}
        temb: Optional[Tensor] = None # No time embedding

        # Initial convolution
        h: Tensor = self.conv_in(x)
        hs['in'] = h.clone() # Store after initial conv

        # Downsampling blocks
        for i_level, down_block_level in enumerate(self.down_blocks):
            h_prev_level = h # Feature before this level's main processing but after previous downsample
            h = down_block_level(h, temb)
            hs[f'down_{i_level}'] = h # Output of the i_level down_block processing

        # Middle block
        if self.enable_mid_block and self.mid_block:
            h = self.mid_block['block_1'](h, temb)
            h = self.mid_block['attn_1'](h) # Self-attention
            hs['mid_attn'] = h.clone() # Feature after mid-block attention
            h = self.mid_block['block_2'](h, temb)
            hs['mid_out'] = h.clone() # Feature after full mid-block

        # Final output processing
        h = self.norm_out(h)
        h = silu_activation(h)
        h_out: Tensor = self.conv_out(h)
        hs['out'] = h_out # Bottleneck features

        return hs


class MultiHeadDecoderBase(nn.Module): # Base class for Decoders
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int, # Final output channels (e.g., 3 for RGB)
        ch_mult: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_resolutions: Sequence[int] = (16,),
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        # in_channels: int = 3, # Not directly used by decoder for input image
        resolution: int = 512,
        z_channels: int = 256, # Input z channels from bottleneck
        enable_mid_block: bool = True,
        num_heads: int = 1,
        use_cross_attention: bool = False, # Specific to Transformer variant
        **ignore_kwargs: Any
    ):
        super().__init__()
        self.ch: int = ch
        self.temb_ch: int = 0 # No time embedding
        self.num_resolutions: int = len(ch_mult)
        self.num_res_blocks: int = num_res_blocks
        self.resolution: int = resolution
        self.enable_mid_block: bool = enable_mid_block
        self.use_cross_attention = use_cross_attention

        # Initial channel count at the bottleneck (lowest resolution)
        current_block_input_ch: int = ch * ch_mult[-1] # ch_mult is reversed in decoder logic
        
        # Resolution at bottleneck
        curr_res: int = resolution // (2**(self.num_resolutions - 1))
        self.z_shape: Tuple[int, int, int, int] = (1, z_channels, curr_res, curr_res)
        print(f"Decoder working with z of shape {self.z_shape} = {np.prod(self.z_shape)} dimensions.")

        # Input convolution: maps z_channels to current_block_input_ch
        self.conv_in: nn.Conv2d = nn.Conv2d(z_channels, current_block_input_ch, kernel_size=3, stride=1, padding=1)

        # Middle block (operates at bottleneck resolution)
        self.mid_block: Optional[nn.ModuleDict] = None
        if self.enable_mid_block:
            self.mid_block = nn.ModuleDict({
                'block_1': ResnetBlock(in_channels=current_block_input_ch, out_channels=current_block_input_ch,
                                       temb_channels=self.temb_ch, dropout=dropout),
                'attn_1': MultiHeadAttnBlock(current_block_input_ch, num_heads),
                'block_2': ResnetBlock(in_channels=current_block_input_ch, out_channels=current_block_input_ch,
                                       temb_channels=self.temb_ch, dropout=dropout)
            })

        # Upsampling blocks
        self.up_blocks: nn.ModuleList = nn.ModuleList() # Renamed from self.up
        # Iterate from bottleneck up to full resolution
        for i_level in reversed(range(self.num_resolutions)):
            block_output_ch = ch * ch_mult[i_level]
            is_not_first = (i_level != 0) # Not the highest resolution level yet

            level_block = ResAttnBlockUp(
                in_channels_block=current_block_input_ch, # Input to this level's ResBlocks
                out_channels_block=block_output_ch,      # Output of this level's ResBlocks
                temb_channels=self.temb_ch,
                dropout=dropout,
                curr_res=curr_res,
                attn_resolutions=attn_resolutions,
                num_heads=num_heads,
                is_not_first_level=is_not_first,
                resamp_with_conv=resamp_with_conv,
                use_cross_attention=self.use_cross_attention
            )
            self.up_blocks.insert(0, level_block) # Prepend to maintain order from low to high res
            current_block_input_ch = block_output_ch # Output of this level becomes input for next if no upsample
                                                     # Or, input to the upsampler then next level.
                                                     # ResAttnBlockUp handles this internally.
            if is_not_first:
                curr_res *= 2
        
        # Output layers
        self.norm_out: Normalize = Normalize(current_block_input_ch) # current_block_input_ch is now ch (mult=1 at highest res)
        self.conv_out: nn.Conv2d = nn.Conv2d(current_block_input_ch, out_ch, kernel_size=3, stride=1, padding=1)


class MultiHeadDecoder(MultiHeadDecoderBase):
    def __init__(self, give_pre_end: bool = False, **kwargs: Any):
        super().__init__(use_cross_attention=False, **kwargs)
        self.give_pre_end: bool = give_pre_end

    def forward(self, z: Tensor) -> Tensor:
        temb: Optional[Tensor] = None
        h: Tensor = self.conv_in(z)

        if self.enable_mid_block and self.mid_block:
            h = self.mid_block['block_1'](h, temb)
            h = self.mid_block['attn_1'](h) # Self-attention
            h = self.mid_block['block_2'](h, temb)

        for up_block_level in self.up_blocks: # up_blocks are already ordered low to high res
            h = up_block_level(h, temb) # Self-attention inside ResAttnBlockUp

        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = silu_activation(h)
        h = self.conv_out(h)
        return h


class MultiHeadDecoderTransformer(MultiHeadDecoderBase):
    def __init__(self, give_pre_end: bool = False, **kwargs: Any):
        super().__init__(use_cross_attention=True, **kwargs)
        self.give_pre_end: bool = give_pre_end

    def forward(self, z: Tensor, hs_encoder: Dict[str, Tensor]) -> Tensor:

        temb: Optional[Tensor] = None
        h: Tensor = self.conv_in(z)

        if self.enable_mid_block and self.mid_block:
            h = self.mid_block['block_1'](h, temb)
            # Cross-attention with a mid-level feature from encoder
            cross_attn_mid_key = 'mid_attn' # Or other appropriate key from MultiHeadEncoder's hs
            if cross_attn_mid_key in hs_encoder:
                 h = self.mid_block['attn_1'](x_kv=h, y_q=hs_encoder[cross_attn_mid_key])
            else: # Fallback to self-attention if key not found
                 h = self.mid_block['attn_1'](x_kv=h)
            h = self.mid_block['block_2'](h, temb)

        for i, up_block_level in enumerate(self.up_blocks):

            encoder_hs_key = f'down_{self.num_resolutions - 1 - i}' # Example key structure

            
            cross_attn_hs: Optional[Tensor] = hs_encoder.get(encoder_hs_key)
            h = up_block_level(h, temb, cross_attn_hs=cross_attn_hs)

        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = silu_activation(h)
        h = self.conv_out(h)
        return h


class RestoreFormer(nn.Module):
    """RestoreFormer main model: Encoder + VectorQuantizer + DecoderTransformer."""
    def __init__(
        self,
        n_embed: int = 1024,       # Number of codebook embeddings
        embed_dim: int = 256,      # Dimension of each embedding
        ch: int = 64,              # Base channels for Encoder/Decoder
        out_ch: int = 3,           # Output image channels (RGB)
        ch_mult: Sequence[int] = (1, 2, 2, 4, 4, 8), # Channel multipliers
        num_res_blocks: int = 2,
        attn_resolutions: Sequence[int] = (16,), # Resolutions for attention
        dropout: float = 0.0,
        in_channels: int = 3,      # Input image channels
        resolution: int = 512,     # Input image resolution
        z_channels: int = 256,     # Bottleneck z channels (output of encoder conv_out before quant_conv)
        double_z: bool = False,    # For VAE-like mean/logvar, not used by VQVAE's encoder here
        enable_mid_block: bool = True,
        fix_decoder: bool = False,
        fix_codebook: bool = True, # Usually True after VQVAE pretraining
        fix_encoder: bool = False,
        num_heads: int = 8,        # num_heads for MultiHeadAttnBlock
    ):
        super().__init__() # Python 3 super()

        self.encoder: MultiHeadEncoder = MultiHeadEncoder(
            ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
            resolution=resolution, z_channels=z_channels, double_z=double_z, # double_z for encoder's conv_out
            enable_mid_block=enable_mid_block, num_heads=num_heads
        )
        # Note: z_channels for decoder's conv_in should match embed_dim after quantization
        self.decoder: MultiHeadDecoderTransformer = MultiHeadDecoderTransformer(
            ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions, dropout=dropout, # in_channels not needed for decoder
            resolution=resolution, z_channels=embed_dim, # Decoder input z is from post_quant_conv (embed_dim)
            enable_mid_block=enable_mid_block, num_heads=num_heads
        )

        self.quantize: VectorQuantizer = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        # Maps encoder's z_channels to VQ's embed_dim
        self.quant_conv: nn.Conv2d = nn.Conv2d(z_channels, embed_dim, kernel_size=1)

        self.post_quant_conv: nn.Conv2d = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)


        # Parameter freezing logic
        if fix_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.post_quant_conv.parameters(): # Also part of "decoding" path
                param.requires_grad = False
            # If decoder is fixed, VQ part used by decoder (codebook for lookup) should also be fixed
            if fix_codebook: # Redundant if fix_decoder implies fixing its VQ path
                 for param in self.quantize.parameters():
                    param.requires_grad = False
        
        # fix_codebook is often independent of fix_decoder (e.g., train encoder/decoder with fixed codebook)
        if fix_codebook and not fix_decoder: # Apply if fix_codebook is true and not already handled by fix_decoder
            for param in self.quantize.parameters():
                param.requires_grad = False
        
        if fix_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.quant_conv.parameters(): # Part of "encoding" path
                param.requires_grad = False


    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor, Tensor, Tensor], Dict[str, Tensor]]:
        hs_encoder: Dict[str, Tensor] = self.encoder(x)
        # 'out' key from hs_encoder contains the bottleneck features from encoder's conv_out
        z_from_encoder: Tensor = hs_encoder['out'] 
        
        # Project encoder output to embedding dimension for VQ
        h_quant_conv: Tensor = self.quant_conv(z_from_encoder)
        quantized_z, emb_loss, vq_info = self.quantize(h_quant_conv)
        return quantized_z, emb_loss, vq_info, hs_encoder

    def decode(self, quantized_z: Tensor, hs_encoder: Dict[str, Tensor]) -> Tensor:
        # Project quantized_z (which is in embed_dim) back using post_quant_conv
        quant_after_proj: Tensor = self.post_quant_conv(quantized_z)
        # Decoder takes this projected tensor and encoder hidden states for cross-attention
        reconstruction: Tensor = self.decoder(quant_after_proj, hs_encoder)
        return reconstruction

    def forward(self, x_input: Tensor, **kwargs: Any) -> Tuple[Tensor, Tensor | None]:
        # kwargs is for compatibility with GFPGANer which passes 'weight' sometimes.
        # RestoreFormer doesn't use it directly.
        quantized_z, diff_loss, vq_info, hs_encoder = self.encode(x_input)
        reconstruction = self.decode(quantized_z, hs_encoder)
        
        return reconstruction, None