"""
OTT3R Student Model

Compact transformer for multi-view 3D reconstruction via knowledge distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from functools import partial
from copy import deepcopy
import warnings
import sys
from pathlib import Path

# Add pi3 to path for importing components
pi3_path = str(Path(__file__).parent.parent.parent / "external" / "pi3")
if pi3_path not in sys.path:
    sys.path.insert(0, pi3_path)

# Import components from teacher model
from pi3.models.dinov2.layers import Mlp
from pi3.utils.geometry import homogenize_points
from pi3.models.layers.pos_embed import RoPE2D, PositionGetter
from pi3.models.layers.block import BlockRope
from pi3.models.layers.attention import FlashAttentionRope
from pi3.models.layers.transformer_head import TransformerDecoder, LinearPts3d
from pi3.models.layers.camera_head import CameraHead


class OTT3R(nn.Module):
    """
    OTT3R student model for multi-view 3D reconstruction.

    Outputs:
    - local_points: [B, N, H, W, 3] - camera-space 3D points
    - conf: [B, N, H, W, 1] - confidence
    - camera_poses: [B, N, 4, 4] - SE(3) poses
    - points: [B, N, H, W, 3] - global 3D points
    """

    def __init__(
        self,
        # Encoder config (DUNE ViT-Small - fixed)
        enc_embed_dim: int = 384,
        patch_size: int = 14,

        # Main decoder config
        dec_embed_dim: int = 384,
        dec_depth: int = 8,
        dec_num_heads: int = 6,
        mlp_ratio: float = 4.0,

        # Task decoder config (points, confidence)
        task_dec_dim: int = 384,
        task_dec_depth: int = 3,
        task_dec_heads: int = 6,

        # Camera decoder config (separate for larger capacity)
        # Teacher uses: dec_dim=1024, heads=16, out_dim=512
        camera_dec_dim: int = 384,
        camera_dec_heads: int = 6,
        camera_dec_depth: int = 3,
        camera_out_dim: int = 256,

        # Positional encoding
        pos_type: str = 'rope100',

        # Register tokens
        num_register_tokens: int = 5,

        # Loading
        load_pretrained_encoder: bool = True,
        encoder_type: str = "dune",  # "dune" or "dinov2" (both patch_size=14)
        freeze_encoder: bool = False,  # Freeze encoder weights (useful for fine-tuning)
    ):
        super().__init__()

        # Store config
        self.patch_size = patch_size
        self.enc_embed_dim = enc_embed_dim
        self.encoder_type = encoder_type
        self.dec_embed_dim = dec_embed_dim
        self.pos_type = pos_type
        self.num_register_tokens = num_register_tokens
        self.camera_dec_dim = camera_dec_dim
        self.camera_dec_heads = camera_dec_heads
        self.camera_out_dim = camera_out_dim

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = self._build_encoder(load_pretrained_encoder, encoder_type)

        # Optionally freeze encoder (for fine-tuning on small datasets)
        if freeze_encoder:
            print("Freezing encoder weights")
            for param in self.encoder.parameters():
                param.requires_grad = False

        # ----------------------
        #  Positional Encoding
        # ----------------------
        self.rope = None
        if pos_type.startswith('rope'):
            if RoPE2D is None:
                raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError(f"Unknown pos_type: {pos_type}")

        # ----------------------
        #     Projection
        # ----------------------
        # Project encoder output to decoder dimension if different
        if enc_embed_dim != dec_embed_dim:
            self.enc_to_dec = nn.Linear(enc_embed_dim, dec_embed_dim)
        else:
            self.enc_to_dec = nn.Identity()

        # ----------------------
        #     Main Decoder
        # ----------------------
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)
        ])
        self.dec_depth = dec_depth

        # ----------------------
        #    Register Tokens
        # ----------------------
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(
            torch.randn(1, 1, num_register_tokens, dec_embed_dim)
        )
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2 * dec_embed_dim,  # Concat of last two decoder layers
            dec_embed_dim=task_dec_dim,
            dec_num_heads=task_dec_heads,
            out_dim=task_dec_dim,
            depth=task_dec_depth,
            rope=self.rope,
        )
        self.point_head = LinearPts3d(
            patch_size=self.patch_size,  # Use self.patch_size (updated by encoder)
            dec_embed_dim=task_dec_dim,
            output_dim=3
        )

        # ----------------------
        #     Conf Decoder
        # ----------------------
        self.conf_decoder = TransformerDecoder(
            in_dim=2 * dec_embed_dim,
            dec_embed_dim=task_dec_dim,
            dec_num_heads=task_dec_heads,
            out_dim=task_dec_dim,
            depth=task_dec_depth,
            rope=self.rope,
        )
        self.conf_head = LinearPts3d(
            patch_size=self.patch_size,  # Use self.patch_size (updated by encoder)
            dec_embed_dim=task_dec_dim,
            output_dim=1
        )

        # ----------------------
        #  Camera Pose Decoder (separate config for larger capacity)
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2 * dec_embed_dim,
            dec_embed_dim=camera_dec_dim,
            dec_num_heads=camera_dec_heads,
            out_dim=camera_out_dim,
            depth=camera_dec_depth,
            rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=camera_out_dim)

        # ----------------------
        #  ImageNet Normalize
        # ----------------------
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

    def _build_encoder(self, load_pretrained: bool, encoder_type: str = "dune") -> nn.Module:
        """Build encoder (DUNE, DINOv2, or DINOv3)."""
        if not load_pretrained:
            raise NotImplementedError("Non-pretrained encoder not implemented")

        expected_dim = self.enc_embed_dim

        if encoder_type == "dune":
            try:
                print("Loading DUNE ViT-Small encoder ...")

                # Add external/dune to Python path
                dune_path = str(Path(__file__).parent.parent.parent / "external" / "dune")
                if dune_path not in sys.path:
                    sys.path.insert(0, dune_path)

                # Load DUNE ViT-Small encoder via torch.hub
                encoder = torch.hub.load(
                    "naver/dune",
                    "dune_vitsmall_14_448_encoder",
                    trust_repo=True
                )
                print("Successfully loaded DUNE ViT-Small encoder")
                print("DUNE encoder: 384 embed_dim, 14 patch_size, 12 depth, 6 heads")

            except Exception as e:
                raise RuntimeError(f"Failed to load DUNE encoder: {e}")

        elif encoder_type == "dinov2":
            try:
                print("Loading DINOv2 ViT-Small encoder ...")

                # Load DINOv2 ViT-Small (patch_size=14) via torch.hub
                encoder = torch.hub.load(
                    "facebookresearch/dinov2",
                    "dinov2_vits14",
                    trust_repo=True
                )
                print("Successfully loaded DINOv2 ViT-Small encoder")
                print("DINOv2 encoder: 384 embed_dim, 14 patch_size, 12 depth, 6 heads")

            except Exception as e:
                raise RuntimeError(f"Failed to load DINOv2 encoder: {e}")

        elif encoder_type == "dinov2_base":
            try:
                print("Loading DINOv2 ViT-Base encoder ...")

                # Load DINOv2 ViT-Base (patch_size=14) via torch.hub
                encoder = torch.hub.load(
                    "facebookresearch/dinov2",
                    "dinov2_vitb14",
                    trust_repo=True
                )
                print("Successfully loaded DINOv2 ViT-Base encoder")
                print("DINOv2-Base encoder: 768 embed_dim, 14 patch_size, 12 depth, 12 heads")

            except Exception as e:
                raise RuntimeError(f"Failed to load DINOv2-Base encoder: {e}")

        elif encoder_type == "dinov3":
            try:
                print("Loading DINOv3 ViT-Small encoder ...")

                # Load DINOv3 directly from cached repo (avoids torch.hub PyTorch version issues)
                dinov3_path = Path.home() / ".cache/torch/hub/facebookresearch_dinov3_main"
                if not dinov3_path.exists():
                    # Download if not cached
                    torch.hub.load("facebookresearch/dinov3", "dinov3_vits16", trust_repo=True)

                if str(dinov3_path) not in sys.path:
                    sys.path.insert(0, str(dinov3_path))

                from dinov3.models.vision_transformer import vit_small
                encoder = vit_small(patch_size=16)

                # Load pretrained weights
                weights_path = Path(__file__).parent.parent.parent / "pretrained_models" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
                if weights_path.exists():
                    state_dict = torch.load(weights_path, map_location='cpu')
                    encoder.load_state_dict(state_dict, strict=False)
                    print(f"Loaded DINOv3 weights from {weights_path}")
                else:
                    print(f"WARNING: DINOv3 weights not found at {weights_path}, using random init")

                print("Successfully loaded DINOv3 ViT-Small encoder")
                print("DINOv3 encoder: 384 embed_dim, 16 patch_size, 12 depth, 6 heads")

            except Exception as e:
                raise RuntimeError(f"Failed to load DINOv3 encoder: {e}")

        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Options: 'dune', 'dinov2', 'dinov2_base', 'dinov3'")

        # Verify dimensions match config
        assert hasattr(encoder, 'embed_dim'), "Encoder missing embed_dim attribute"
        assert hasattr(encoder, 'patch_size') and encoder.patch_size in [14, 16], "Encoder must have patch_size=14 or 16"
        # Update patch_size from encoder (DINOv3 uses 16, others use 14)
        self.patch_size = encoder.patch_size
        if encoder.embed_dim != expected_dim:
            raise ValueError(f"Encoder embed_dim ({encoder.embed_dim}) doesn't match config enc_embed_dim ({expected_dim})")

        # Check for NaN/inf in weights
        for name, param in encoder.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"WARNING: Encoder parameter {name} contains NaN/inf!")
                break

        return encoder

    def decode(self, hidden, N, H, W):
        """
        Main decoder with alternating self/cross attention.

        Even layers: within-view self-attention [B*N, hw, D]
        Odd layers: cross-view attention [B, N*hw, D]

        Returns concatenation of last two layer outputs for task decoders.
        """
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []

        hidden = hidden.reshape(B * N, hw, -1)

        # Add register tokens
        register_token = self.register_token.repeat(B, N, 1, 1).reshape(
            B * N, *self.register_token.shape[-2:]
        )
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        # Setup RoPE positions
        if self.pos_type.startswith('rope'):
            pos = self.position_getter(
                B * N, H // self.patch_size, W // self.patch_size, hidden.device
            )

        # Add offset for register tokens (no position embedding for them)
        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(
                B * N, self.patch_start_idx, 2
            ).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # Alternating attention through decoder
        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                # Even layers: within-view self-attention
                pos = pos.reshape(B * N, hw, -1)
                hidden = hidden.reshape(B * N, hw, -1)
            else:
                # Odd layers: cross-view attention
                pos = pos.reshape(B, N * hw, -1)
                hidden = hidden.reshape(B, N * hw, -1)

            hidden = blk(hidden, xpos=pos)

            # Collect last two layers for task decoders
            if i + 1 in [len(self.decoder) - 1, len(self.decoder)]:
                final_output.append(hidden.reshape(B * N, hw, -1))

        # Concatenate last two layers
        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B * N, hw, -1)

    def forward(self, imgs):
        """
        Forward pass.

        Args:
            imgs: [B, N, 3, H, W] - batch of N images

        Returns:
            dict with:
            - points: [B, N, H, W, 3] - global 3D points
            - local_points: [B, N, H, W, 3] - camera-space 3D points
            - conf: [B, N, H, W, 1] - confidence
            - camera_poses: [B, N, 4, 4] - SE(3) poses
        """
        # Normalize images
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # Encode images
        imgs = imgs.reshape(B * N, -1, H, W)

        # Encoder forward (handle DUNE vs DINOv2/DINOv3 output formats)
        if self.encoder_type in ["dinov2", "dinov2_base", "dinov3"]:
            # DINOv2/DINOv3: use forward_features to get patch tokens
            encoder_output = self.encoder.forward_features(imgs)
            hidden = encoder_output['x_norm_patchtokens']
        else:
            # DUNE: returns dict with x_norm_patchtokens
            encoder_output = self.encoder(imgs)
            if isinstance(encoder_output, dict):
                hidden = encoder_output['x_norm_patchtokens']
            else:
                hidden = encoder_output

        # Project to decoder dimension
        hidden = self.enc_to_dec(hidden)

        # Main decoder with alternating attention
        hidden, pos = self.decode(hidden, N, H, W)

        # Task-specific decoders
        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        # Heads (disable autocast for numerical stability)
        # Compute actual output dimensions (may differ from H,W if patch_size doesn't divide evenly)
        H_out = (H // self.patch_size) * self.patch_size
        W_out = (W // self.patch_size) * self.patch_size
        need_resize = (H_out != H) or (W_out != W)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # Local points
            point_hidden = point_hidden.float()
            ret = self.point_head(
                [point_hidden[:, self.patch_start_idx:]], (H, W)
            ).reshape(B * N, H_out, W_out, -1)

            # Resize if needed (e.g., DINOv3 patch_size=16 doesn't divide 518 evenly)
            if need_resize:
                ret = ret.permute(0, 3, 1, 2)  # [BN, C, H_out, W_out]
                ret = F.interpolate(ret, size=(H, W), mode='bilinear', align_corners=False)
                ret = ret.permute(0, 2, 3, 1)  # [BN, H, W, C]

            ret = ret.reshape(B, N, H, W, -1)

            # Predict xy/z and log(z), then recover xyz
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # Confidence
            conf_hidden = conf_hidden.float()
            conf = self.conf_head(
                [conf_hidden[:, self.patch_start_idx:]], (H, W)
            ).reshape(B * N, H_out, W_out, -1)

            if need_resize:
                conf = conf.permute(0, 3, 1, 2)
                conf = F.interpolate(conf, size=(H, W), mode='bilinear', align_corners=False)
                conf = conf.permute(0, 2, 3, 1)

            conf = conf.reshape(B, N, H, W, -1)

            # Camera poses
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(
                camera_hidden[:, self.patch_start_idx:], patch_h, patch_w
            ).reshape(B, N, 4, 4)

            # Global points = camera_pose @ homogenize(local_points)
            points = torch.einsum(
                'bnij, bnhwj -> bnhwi',
                camera_poses,
                homogenize_points(local_points)
            )[..., :3]

        return dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
        )

    def get_model_stats(self) -> Dict[str, any]:
        """Get model statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Component breakdown
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        point_dec_params = sum(p.numel() for p in self.point_decoder.parameters())
        point_head_params = sum(p.numel() for p in self.point_head.parameters())
        conf_dec_params = sum(p.numel() for p in self.conf_decoder.parameters())
        conf_head_params = sum(p.numel() for p in self.conf_head.parameters())
        camera_dec_params = sum(p.numel() for p in self.camera_decoder.parameters())
        camera_head_params = sum(p.numel() for p in self.camera_head.parameters())

        return {
            'total_parameters': total_params,
            'total_parameters_M': total_params / 1e6,
            'trainable_parameters': trainable_params,
            'trainable_ratio': trainable_params / total_params,
            'component_breakdown': {
                'encoder': encoder_params / 1e6,
                'main_decoder': decoder_params / 1e6,
                'point_decoder': point_dec_params / 1e6,
                'point_head': point_head_params / 1e6,
                'conf_decoder': conf_dec_params / 1e6,
                'conf_head': conf_head_params / 1e6,
                'camera_decoder': camera_dec_params / 1e6,
                'camera_head': camera_head_params / 1e6,
            },
            'config': {
                'enc_embed_dim': self.enc_embed_dim,
                'dec_embed_dim': self.dec_embed_dim,
                'dec_depth': self.dec_depth,
                'patch_size': self.patch_size,
                'camera_dec_dim': self.camera_dec_dim,
                'camera_dec_heads': self.camera_dec_heads,
                'camera_out_dim': self.camera_out_dim,
            },
            'model_type': 'OTT3R'
        }




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-load', action='store_true', help='Skip loading pretrained encoder')
    args = parser.parse_args()

    print("=" * 60)
    print("OTT3R Model Test")
    print("=" * 60)

    model = OTT3R(load_pretrained_encoder=not args.no_load)

    # Get stats
    stats = model.get_model_stats()

    print(f"\nTotal Parameters: {stats['total_parameters_M']:.2f}M")
    print(f"Trainable Parameters: {stats['trainable_parameters'] / 1e6:.2f}M")
    print(f"\nComponent Breakdown (M params):")
    for name, params in stats['component_breakdown'].items():
        print(f"  {name}: {params:.2f}M")

    # Test forward pass with dummy input
    if torch.cuda.is_available():
        print("\nTesting forward pass on CUDA...")
        model = model.cuda()

        # Dummy input: batch=1, N=2 views, 224x518
        dummy_input = torch.randn(1, 2, 3, 224, 518).cuda()

        with torch.no_grad():
            output = model(dummy_input)

        print(f"\nOutput shapes:")
        for key, val in output.items():
            print(f"  {key}: {val.shape}")

        print("\nForward pass successful!")
    else:
        print("\nCUDA not available, skipping forward pass test")
