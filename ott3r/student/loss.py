"""
OTT3R Distillation Loss

Loss functions for knowledge distillation training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Union
import math
import sys
from pathlib import Path

# Add pi3 to path for geometry utils
pi3_path = str(Path(__file__).parent.parent.parent / "external" / "pi3")
if pi3_path not in sys.path:
    sys.path.insert(0, pi3_path)

from pi3.utils.geometry import se3_inverse, depth_edge
from pi3.utils.alignment import align_points_scale


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def weighted_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, tuple] = None,
                  keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    """Weighted mean, with optional mask as weights."""
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)


def _smooth(err: torch.Tensor, beta: float = 0.0) -> torch.Tensor:
    """Smooth L1-like loss function."""
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)


def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Angular difference between two 3D vectors using atan2 (numerically stable)."""
    return torch.atan2(torch.cross(v1, v2, dim=-1).norm(dim=-1) + eps, (v1 * v2).sum(dim=-1))


# ---------------------------------------------------------------------------
# Point Loss (adapted for distillation)
# ---------------------------------------------------------------------------

class PointLoss(nn.Module):
    """Scale-invariant local point loss using ROE solver for scale alignment."""

    def __init__(self, local_align_res: int = 4096, include_normal_loss: bool = True):
        super().__init__()
        self.local_align_res = local_align_res
        self.include_normal_loss = include_normal_loss
        self.criteria_local = nn.L1Loss(reduction='none')

    def prepare_ROE(self, pts: torch.Tensor, mask: torch.Tensor, target_size: int = 4096) -> torch.Tensor:
        """
        Prepare points for ROE (Robust Optimal Estimation) scale alignment.

        Subsamples valid points to target_size for efficient alignment computation.
        Uses nearest interpolation (important for stable results).
        """
        B, N, H, W, C = pts.shape
        output = []

        for i in range(B):
            valid_pts = pts[i][mask[i]]

            if valid_pts.shape[0] > 0:
                valid_pts = valid_pts.permute(1, 0).unsqueeze(0)  # (1, C, num_valid)
                # NOTE: Nearest interpolation is important for stable results
                valid_pts = F.interpolate(valid_pts, size=target_size, mode='nearest')
                valid_pts = valid_pts.squeeze(0).permute(1, 0)  # (target_size, C)
            else:
                valid_pts = torch.ones((target_size, C), device=pts.device, dtype=pts.dtype)

            output.append(valid_pts)

        return torch.stack(output, dim=0)

    def normal_loss(self, points: torch.Tensor, gt_points: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
        """
        4-way cross-product normal loss.

        Computes surface normals using 4 different cross-product combinations
        for robustness, with smoothed angular loss.
        """
        # Mask out depth edges
        not_edge = ~depth_edge(gt_points[..., 2], rtol=0.03)
        mask = torch.logical_and(mask, not_edge)

        # Get 4 corner points
        leftup = points[..., :-1, :-1, :]
        rightup = points[..., :-1, 1:, :]
        leftdown = points[..., 1:, :-1, :]
        rightdown = points[..., 1:, 1:, :]

        # Compute 4 cross products for student
        upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
        leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
        downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
        rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

        # Get GT corner points
        gt_leftup = gt_points[..., :-1, :-1, :]
        gt_rightup = gt_points[..., :-1, 1:, :]
        gt_leftdown = gt_points[..., 1:, :-1, :]
        gt_rightdown = gt_points[..., 1:, 1:, :]

        # Compute 4 cross products for GT
        gt_upxleft = torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)
        gt_leftxdown = torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)
        gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
        gt_rightxup = torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)

        # Compute validity masks for each cross product
        mask_leftup = mask[..., :-1, :-1]
        mask_rightup = mask[..., :-1, 1:]
        mask_leftdown = mask[..., 1:, :-1]
        mask_rightdown = mask[..., 1:, 1:]

        mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
        mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
        mask_downxright = mask_leftdown & mask_rightup & mask_leftup
        mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

        # Angular loss with smoothing (beta=3 degrees)
        MIN_ANGLE = math.radians(1)
        MAX_ANGLE = math.radians(90)
        BETA_RAD = math.radians(3)

        loss = (
            mask_upxleft * _smooth(angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) +
            mask_leftxdown * _smooth(angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) +
            mask_downxright * _smooth(angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) +
            mask_rightxup * _smooth(angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)
        )

        # Normalize by spatial dimensions
        loss = loss.mean() / (4 * max(points.shape[-3:-1]))

        return loss

    def forward(self, pred: Dict, target: Dict, mask: torch.Tensor) -> Tuple[torch.Tensor, Dict, torch.Tensor, torch.Tensor]:
        """
        Compute point loss with scale alignment.

        Args:
            pred: Dict with 'local_points' [B, N, H, W, 3]
            target: Dict with 'local_points' [B, N, H, W, 3]
            mask: Valid pixel mask [B, N, H, W]

        Returns:
            loss: Total point loss
            details: Dict of individual loss components
            scale: Computed scale alignment factor [B]
            aligned_local_pts: Scale-aligned student points [B, N, H, W, 3]
        """
        pred_local_pts = pred['local_points']
        gt_local_pts = target['local_points']

        details = {}
        device = pred_local_pts.device

        B, N, H, W, _ = pred_local_pts.shape

        # Check if we have any valid points
        num_valid = mask.sum()
        if num_valid == 0:
            # No valid points - return zero loss
            zero_loss = torch.tensor(0.0, device=device)
            details['local_pts_loss'] = zero_loss
            details['normal_loss'] = zero_loss
            return zero_loss, details, torch.ones(B, device=device), pred_local_pts

        # Replace NaN/inf in targets with 0 (masked regions have NaN in cache)
        # This prevents NaN * 0 = NaN in weighted computations
        gt_local_pts = torch.where(torch.isfinite(gt_local_pts), gt_local_pts, torch.zeros_like(gt_local_pts))

        # Compute depth-based weights (closer points weighted more)
        weights = gt_local_pts[..., 2].clone()
        weights = weights.clamp_min(0.1 * weighted_mean(weights, mask, dim=(-2, -1), keepdim=True))
        weights = 1 / (weights + 1e-6)

        # Scale alignment via ROE solver
        with torch.no_grad():
            xyz_pred = self.prepare_ROE(pred_local_pts, mask, self.local_align_res).contiguous()
            xyz_gt = self.prepare_ROE(gt_local_pts, mask, self.local_align_res).contiguous()
            xyz_weights = self.prepare_ROE(weights[..., None], mask, self.local_align_res).contiguous()[:, :, 0]

            scale = align_points_scale(xyz_pred, xyz_gt, xyz_weights)
            # Ensure positive scale, clamp to avoid zero
            scale = scale.abs().clamp(min=1e-6)

        # Apply scale alignment
        aligned_local_pts = scale.view(B, 1, 1, 1, 1) * pred_local_pts

        # Local point loss (depth-weighted L1)
        local_pts_loss = self.criteria_local(
            aligned_local_pts[mask].float(),
            gt_local_pts[mask].float()
        ) * weights[mask].float()[..., None]

        final_loss = local_pts_loss.mean()
        details['local_pts_loss'] = final_loss.clone()

        # Normal loss (optional)
        if self.include_normal_loss:
            normal_loss = self.normal_loss(aligned_local_pts, gt_local_pts, mask)
            final_loss = final_loss + normal_loss
            details['normal_loss'] = normal_loss
        else:
            details['normal_loss'] = torch.tensor(0.0, device=device)

        return final_loss, details, scale, aligned_local_pts


# ---------------------------------------------------------------------------
# Confidence Loss (adapted for distillation)
# ---------------------------------------------------------------------------

class ConfidenceLoss(nn.Module):
    """
    Confidence loss supporting three modes:
    - bce: Binary cross-entropy with labels derived from reconstruction error.
           Pixels with error < threshold receive label 1.
    - l1: Direct L1 regression against teacher confidence.
    - hybrid: Soft blend of BCE and L1 weighted by teacher confidence.
    """

    def __init__(self, loss_type: str = "bce", expected_dist_thresh: float = 0.02, use_logits: bool = True):
        """
        Args:
            loss_type: Loss computation mode - 'bce', 'l1', or 'hybrid'.
            expected_dist_thresh: Error threshold for BCE binary labels (default: 0.02).
            use_logits: If True, apply sigmoid to model outputs before comparison.
        """
        super().__init__()
        self.loss_type = loss_type
        self.expected_dist_thresh = expected_dist_thresh
        self.use_logits = use_logits

        if loss_type == "bce":
            if use_logits:
                self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
            else:
                self.bce_loss = nn.BCELoss(reduction='none')
        elif loss_type == "l1":
            pass  # Direct computation, no stored loss function needed
        elif loss_type == "hybrid":
            # Hybrid uses both BCE and L1
            if use_logits:
                self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
            else:
                self.bce_loss = nn.BCELoss(reduction='none')
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Options: 'bce', 'l1', 'hybrid'")

    def forward(self, pred: Dict, target: Dict, mask: torch.Tensor,
                scale: torch.Tensor, aligned_local_pts: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute confidence loss.

        Args:
            pred: Dict with 'conf' [B, N, H, W, 1]
            target: Dict with 'conf' [B, N, H, W, 1] and 'local_points' [B, N, H, W, 3]
            mask: Valid pixel mask [B, N, H, W]
            scale: Scale factor from point alignment [B]
            aligned_local_pts: Scale-aligned student points [B, N, H, W, 3]
        """
        pred_conf = pred['conf'].squeeze(-1)  # [B, N, H, W]

        if self.loss_type == "l1":
            return self._forward_l1(pred_conf, target, mask)
        elif self.loss_type == "bce":
            return self._forward_bce(pred_conf, target, mask, aligned_local_pts)
        elif self.loss_type == "hybrid":
            return self._forward_hybrid(pred_conf, target, mask, aligned_local_pts)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def _forward_l1(self, pred_conf: torch.Tensor, target: Dict, mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """L1 regression against teacher confidence in probability space."""
        gt_conf = target['conf'].squeeze(-1)  # [B, N, H, W], already probabilities

        # Convert student logits to probabilities for proper comparison
        if self.use_logits:
            pred_conf_prob = pred_conf.sigmoid()
        else:
            pred_conf_prob = pred_conf

        # L1 in probability space
        loss = (pred_conf_prob - gt_conf).abs()
        loss = (loss * mask.float()).sum() / (mask.sum() + 1e-8)

        return loss, {'conf_loss': loss}

    def _forward_bce(self, pred_conf: torch.Tensor, target: Dict, mask: torch.Tensor,
                     aligned_local_pts: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """BCE with binary labels from reconstruction error."""
        gt_local = target['local_points']

        # Depth-weighted error computation
        depth = gt_local[..., 2].clamp(min=1e-6)
        weights = 1.0 / (depth + 1e-6)
        point_error = (aligned_local_pts - gt_local).abs() * weights.unsqueeze(-1)
        point_error_mean = point_error.mean(dim=-1)

        # Binary labels based on error threshold
        with torch.no_grad():
            binary_labels = (point_error_mean < self.expected_dist_thresh).float()

        # Clamp if not using logits
        if not self.use_logits:
            pred_conf = pred_conf.clamp(1e-7, 1 - 1e-7)

        # BCE loss
        bce = self.bce_loss(pred_conf, binary_labels)
        loss = (bce * mask.float()).sum() / (mask.sum() + 1e-8)

        with torch.no_grad():
            pos_ratio = (binary_labels * mask.float()).sum() / (mask.sum() + 1e-8)
            pred_mean = (pred_conf.sigmoid() if self.use_logits else pred_conf)
            pred_mean = (pred_mean * mask.float()).sum() / (mask.sum() + 1e-8)

        return loss, {
            'conf_loss': loss,
            'conf_pos_ratio': pos_ratio,
            'conf_pred_mean': pred_mean,
        }

    def _forward_hybrid(self, pred_conf: torch.Tensor, target: Dict, mask: torch.Tensor,
                        aligned_local_pts: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Hybrid BCE/L1 loss with soft blending based on teacher confidence.

        High teacher confidence regions: mostly BCE (learn self-assessment)
        Low teacher confidence regions: mostly L1 (copy teacher's estimate)
        """
        teacher_conf = target['conf'].squeeze(-1)  # [B, N, H, W]
        gt_local = target['local_points']

        # BCE component: binary labels from reconstruction error
        depth = gt_local[..., 2].clamp(min=1e-6)
        weights = 1.0 / (depth + 1e-6)
        point_error = (aligned_local_pts - gt_local).abs() * weights.unsqueeze(-1)
        point_error_mean = point_error.mean(dim=-1)

        with torch.no_grad():
            binary_labels = (point_error_mean < self.expected_dist_thresh).float()

        # BCE loss
        if self.use_logits:
            bce = self.bce_loss(pred_conf, binary_labels)
            pred_conf_prob = pred_conf.sigmoid()
        else:
            pred_conf_clamped = pred_conf.clamp(1e-7, 1 - 1e-7)
            bce = self.bce_loss(pred_conf_clamped, binary_labels)
            pred_conf_prob = pred_conf

        # L1 component: direct regression to teacher confidence
        l1 = (pred_conf_prob - teacher_conf).abs()

        # Soft blend: alpha = teacher_conf (detached to avoid backprop through weights)
        alpha = teacher_conf.detach()
        combined = alpha * bce + (1.0 - alpha) * l1

        # Masked mean
        loss = (combined * mask.float()).sum() / (mask.sum() + 1e-8)

        # Statistics
        with torch.no_grad():
            pos_ratio = (binary_labels * mask.float()).sum() / (mask.sum() + 1e-8)
            pred_mean = (pred_conf_prob * mask.float()).sum() / (mask.sum() + 1e-8)
            bce_weight = (alpha * mask.float()).sum() / (mask.sum() + 1e-8)

        return loss, {
            'conf_loss': loss,
            'conf_pos_ratio': pos_ratio,
            'conf_pred_mean': pred_mean,
            'conf_bce_weight': bce_weight,
        }


# ---------------------------------------------------------------------------
# Camera Loss
# ---------------------------------------------------------------------------

class CameraLoss(nn.Module):
    """
    Affine-invariant camera pose loss.

    Uses relative poses for scale-invariant supervision:
    - Translation: Huber loss with delta=0.1
    - Rotation: Geodesic angular loss

    Optional baseline weighting: Weight pairs by their baseline distance (translation
    magnitude between views). Larger baselines get higher weight, helping the model
    focus on harder pairs. Weights are normalized per-sample to ensure each sample
    contributes equally to the total loss.
    """

    def __init__(self, alpha: float = 100.0, huber_delta: float = 0.1, baseline_weighting: bool = False):
        super().__init__()
        self.alpha = alpha  # Translation weight
        self.huber_delta = huber_delta
        self.baseline_weighting = baseline_weighting

    def rot_ang_loss(self, R: torch.Tensor, R_gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Geodesic angular loss between rotation matrices.

        Args:
            R: Predicted rotation [*, 3, 3]
            R_gt: Ground truth rotation [*, 3, 3]
        """
        residual = torch.matmul(R.transpose(-2, -1), R_gt)
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        R_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))
        return R_err.mean()

    def forward(self, pred: Dict, target: Dict, scale: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute camera pose loss using relative poses.

        Args:
            pred: Dict with 'camera_poses' [B, N, 4, 4]
            target: Dict with 'camera_poses' [B, N, 4, 4]
            scale: Scale factor from point alignment [B]
        """
        pred_pose = pred['camera_poses']
        gt_pose = target['camera_poses']

        B, N, _, _ = pred_pose.shape

        # Scale-align predicted translations
        pred_pose_aligned = pred_pose.clone()
        pred_pose_aligned[..., :3, 3] *= scale.view(B, 1, 1)

        # Compute world-to-camera transforms
        pred_w2c = se3_inverse(pred_pose_aligned)
        gt_w2c = se3_inverse(gt_pose)

        # Compute all pairwise relative poses
        pred_w2c_exp = pred_w2c.unsqueeze(2)  # [B, N, 1, 4, 4]
        pred_pose_exp = pred_pose_aligned.unsqueeze(1)  # [B, 1, N, 4, 4]

        gt_w2c_exp = gt_w2c.unsqueeze(2)
        gt_pose_exp = gt_pose.unsqueeze(1)

        pred_rel_all = torch.matmul(pred_w2c_exp, pred_pose_exp)  # [B, N, N, 4, 4]
        gt_rel_all = torch.matmul(gt_w2c_exp, gt_pose_exp)

        # Mask diagonal (i != j)
        diag_mask = ~torch.eye(N, dtype=torch.bool, device=pred_pose.device)

        if self.baseline_weighting:
            # Compute baseline weights per sample, then weighted average
            total_trans_loss = 0.0
            total_rot_loss = 0.0

            for b in range(B):
                # Extract off-diagonal relative poses for this sample
                pred_rel_b = pred_rel_all[b]  # [N, N, 4, 4]
                gt_rel_b = gt_rel_all[b]  # [N, N, 4, 4]

                t_pred_b = pred_rel_b[..., :3, 3][diag_mask]  # [N*(N-1), 3]
                R_pred_b = pred_rel_b[..., :3, :3][diag_mask]  # [N*(N-1), 3, 3]
                t_gt_b = gt_rel_b[..., :3, 3][diag_mask]
                R_gt_b = gt_rel_b[..., :3, :3][diag_mask]

                # Compute pairwise baselines from GT translations
                gt_trans = gt_pose[b, :, :3, 3]  # [N, 3]
                baselines = torch.cdist(gt_trans.unsqueeze(0), gt_trans.unsqueeze(0)).squeeze(0)  # [N, N]
                baseline_weights = baselines[diag_mask]  # [N*(N-1)]

                # Normalize weights to sum to 1 (per-sample normalization)
                baseline_weights = baseline_weights / (baseline_weights.sum() + 1e-8)

                # Per-pair translation loss (Huber, no reduction)
                t_diff = t_pred_b - t_gt_b
                t_abs = t_diff.abs()
                # Huber loss per element
                t_huber = torch.where(
                    t_abs < self.huber_delta,
                    0.5 * t_diff.pow(2) / self.huber_delta,
                    t_abs - 0.5 * self.huber_delta
                )
                t_loss_per_pair = t_huber.mean(dim=-1)  # [N*(N-1)]

                # Per-pair rotation loss (geodesic)
                residual = torch.matmul(R_pred_b.transpose(-2, -1), R_gt_b)
                trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
                cosine = (trace - 1) / 2
                r_loss_per_pair = torch.acos(torch.clamp(cosine, -1.0 + 1e-6, 1.0 - 1e-6))  # [N*(N-1)]

                # Weighted sum for this sample
                trans_loss_b = (t_loss_per_pair * baseline_weights).sum()
                rot_loss_b = (r_loss_per_pair * baseline_weights).sum()

                total_trans_loss = total_trans_loss + trans_loss_b
                total_rot_loss = total_rot_loss + rot_loss_b

            # Average across batch
            trans_loss = total_trans_loss / B
            rot_loss = total_rot_loss / B

        else:
            # Original behavior: uniform weighting across all pairs
            # Extract off-diagonal relative poses
            t_pred = pred_rel_all[..., :3, 3][:, diag_mask, ...]  # [B, N*(N-1), 3]
            R_pred = pred_rel_all[..., :3, :3][:, diag_mask, ...]  # [B, N*(N-1), 3, 3]

            t_gt = gt_rel_all[..., :3, 3][:, diag_mask, ...]
            R_gt = gt_rel_all[..., :3, :3][:, diag_mask, ...]

            # Translation loss (Huber)
            trans_loss = F.huber_loss(t_pred, t_gt, reduction='mean', delta=self.huber_delta)

            # Rotation loss (geodesic)
            rot_loss = self.rot_ang_loss(R_pred.reshape(-1, 3, 3), R_gt.reshape(-1, 3, 3))

        # Combined loss
        total_loss = self.alpha * trans_loss + rot_loss

        return total_loss, {'trans_loss': trans_loss, 'rot_loss': rot_loss}


# ---------------------------------------------------------------------------
# Combined Distillation Loss
# ---------------------------------------------------------------------------

class OTT3RLoss(nn.Module):
    """Combined distillation loss for OTT3R training."""

    def __init__(
        self,
        include_normal_loss: bool = True,
        include_camera_loss: bool = True,
        include_conf_loss: bool = True,
        camera_loss_weight: float = 0.1,
        camera_baseline_weighting: bool = False,
        conf_loss_weight: float = 0.05,
        conf_loss_type: str = "bce",
        conf_expected_dist_thresh: float = 0.02,
        conf_use_logits: bool = True,
    ):
        super().__init__()

        self.include_normal_loss = include_normal_loss
        self.include_camera_loss = include_camera_loss
        self.include_conf_loss = include_conf_loss

        self.camera_loss_weight = camera_loss_weight
        self.conf_loss_weight = conf_loss_weight

        # Loss components
        self.point_loss = PointLoss(include_normal_loss=include_normal_loss)
        self.camera_loss = CameraLoss(baseline_weighting=camera_baseline_weighting) if include_camera_loss else None
        self.conf_loss = ConfidenceLoss(
            loss_type=conf_loss_type,
            expected_dist_thresh=conf_expected_dist_thresh,
            use_logits=conf_use_logits
        ) if include_conf_loss else None

    def normalize_predictions(self, pred: Dict, mask: torch.Tensor) -> Dict:
        """Normalize predictions by mean distance to origin for scale invariance."""
        local_points = pred['local_points'].clone()
        camera_poses = pred['camera_poses'].clone()

        B, N, H, W, _ = local_points.shape

        # Check which batches have valid points
        valid_batch = mask.sum(dim=[-1, -2, -3]) > 0

        if valid_batch.sum() > 0:
            # Compute mean distance to origin for valid batches only
            all_pts = local_points[valid_batch].clone()
            all_pts[~mask[valid_batch]] = 0
            all_pts = all_pts.reshape(valid_batch.sum(), N, -1, 3)
            all_dis = all_pts.norm(dim=-1)
            norm_factor = all_dis.sum(dim=[-1, -2]) / (mask[valid_batch].float().sum(dim=[-1, -2, -3]) + 1e-8)

            # Clamp norm_factor to avoid division by very small values
            norm_factor = norm_factor.clamp(min=1e-6)

            # Normalize only valid batches
            local_points[valid_batch] = local_points[valid_batch] / norm_factor[..., None, None, None, None]
            camera_poses[valid_batch, :, :3, 3] = camera_poses[valid_batch, :, :3, 3] / norm_factor[..., None, None]

        return {
            'local_points': local_points,
            'camera_poses': camera_poses,
            'conf': pred['conf'],
            'points': pred.get('points'),
        }

    def normalize_targets(self, target: Dict, mask: torch.Tensor) -> Dict:
        """Normalize targets the same way as predictions."""
        local_points = target['local_points'].clone()
        camera_poses = target['camera_poses'].clone()

        B, N, H, W, _ = local_points.shape

        # Replace NaN/inf with 0 in masked regions (teacher cache has NaN there)
        local_points = torch.where(torch.isfinite(local_points), local_points, torch.zeros_like(local_points))

        # Check which batches have valid points
        valid_batch = mask.sum(dim=[-1, -2, -3]) > 0

        if valid_batch.sum() > 0:
            # Compute mean distance for valid batches only
            all_pts = local_points[valid_batch].clone()
            all_pts[~mask[valid_batch]] = 0
            all_pts = all_pts.reshape(valid_batch.sum(), N, -1, 3)
            all_dis = all_pts.norm(dim=-1)
            norm_factor = all_dis.sum(dim=[-1, -2]) / (mask[valid_batch].float().sum(dim=[-1, -2, -3]) + 1e-8)

            # Clamp norm_factor to avoid division by very small values
            norm_factor = norm_factor.clamp(min=1e-6)

            # Normalize only valid batches
            local_points[valid_batch] = local_points[valid_batch] / norm_factor[..., None, None, None, None]
            camera_poses[valid_batch, :, :3, 3] = camera_poses[valid_batch, :, :3, 3] / norm_factor[..., None, None]

        return {
            'local_points': local_points,
            'camera_poses': camera_poses,
            'conf': target['conf'],
        }

    def forward(self, pred: Dict, target: Dict, mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined distillation loss.

        Args:
            pred: Student predictions with keys:
                - local_points: [B, N, H, W, 3]
                - conf: [B, N, H, W, 1]
                - camera_poses: [B, N, 4, 4]
            target: Teacher outputs with same keys
            mask: Valid pixel mask [B, N, H, W]

        Returns:
            total_loss: Combined loss scalar
            details: Dict of individual loss components
        """
        details = {}

        # Normalize predictions and targets
        pred_norm = self.normalize_predictions(pred, mask)
        target_norm = self.normalize_targets(target, mask)

        # Point loss (includes optional normal loss)
        # Also returns scale and aligned points needed for confidence loss
        point_loss, point_details, scale, aligned_local_pts = self.point_loss(pred_norm, target_norm, mask)
        details.update(point_details)

        total_loss = point_loss

        # Camera loss
        if self.include_camera_loss and self.camera_loss is not None:
            cam_loss, cam_details = self.camera_loss(pred_norm, target_norm, scale)
            total_loss = total_loss + self.camera_loss_weight * cam_loss
            details.update(cam_details)
            details['camera_loss'] = cam_loss

        # Confidence loss (BCE based on student's reconstruction error against teacher)
        if self.include_conf_loss and self.conf_loss is not None:
            conf_loss, conf_details = self.conf_loss(
                pred_norm, target_norm, mask,
                scale=scale,
                aligned_local_pts=aligned_local_pts
            )
            total_loss = total_loss + self.conf_loss_weight * conf_loss
            details.update(conf_details)

        details['total_loss'] = total_loss

        return total_loss, details
