# hand_tracking/pipeline.py
# RTMPose-based hand tracking (Pure PyTorch, mmpose-compatible)

import math
import time
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "assets" / "models" / "rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth"

# COCO-WholeBody keypoint indices
# Body: 0-16, Foot: 17-22, Face: 23-90, Left hand: 91-111, Right hand: 112-132
LEFT_HAND_INDICES = list(range(91, 112))   # 21 keypoints
RIGHT_HAND_INDICES = list(range(112, 133)) # 21 keypoints

# Face landmark indices (68-point format, offset by 23)
# Mouth outer: 48-59, inner: 60-67
MOUTH_TOP = 23 + 51      # Upper lip top
MOUTH_BOTTOM = 23 + 57   # Lower lip bottom
MOUTH_LEFT = 23 + 48     # Left corner
MOUTH_RIGHT = 23 + 54    # Right corner
MOUTH_CENTER_INDICES = [23 + 51, 23 + 57, 23 + 62, 23 + 66]  # For center calculation

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def draw_landmarks(frame: np.ndarray, landmarks: np.ndarray, size: int = 2):
    """Draw hand landmarks."""
    if landmarks is None or len(landmarks) == 0:
        return
    for conn in HAND_CONNECTIONS:
        if conn[0] < len(landmarks) and conn[1] < len(landmarks):
            pt1 = tuple(landmarks[conn[0], :2].astype(int))
            pt2 = tuple(landmarks[conn[1], :2].astype(int))
            cv2.line(frame, pt1, pt2, (0, 255, 0), size)
    for pt in landmarks:
        cv2.circle(frame, tuple(pt[:2].astype(int)), size + 1, (0, 200, 255), -1)


def draw_detections(frame: np.ndarray, detections: np.ndarray, scale: float = 1.0, pad: Tuple[int, int] = (0, 0)):
    """Draw detection boxes."""
    for det in detections:
        if len(det) >= 4:
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)


# ============================================================
# RTMPose Model Components (mmpose-compatible)
# ============================================================

class ScaleNorm(nn.Module):
    """Scale Norm from mmpose."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x / norm * self.scale * self.g


class Scale(nn.Module):
    """Scale vector by element multiplications."""
    def __init__(self, dim, init_value=1., trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class RTMCCBlock(nn.Module):
    """Gated Attention Unit (GAU) in RTMBlock - mmpose compatible."""
    def __init__(self, num_token, in_token_dims, out_token_dims,
                 expansion_factor=2, s=128, eps=1e-5, dropout_rate=0.,
                 drop_path=0., attn_type='self-attn', act_fn='SiLU',
                 bias=False, use_rel_bias=True, pos_enc=False):
        super().__init__()
        self.s = s
        self.num_token = num_token
        self.use_rel_bias = use_rel_bias
        self.attn_type = attn_type
        self.pos_enc = pos_enc

        self.e = int(in_token_dims * expansion_factor)
        if use_rel_bias:
            if attn_type == 'self-attn':
                self.w = nn.Parameter(torch.rand([2 * num_token - 1], dtype=torch.float))
            else:
                self.a = nn.Parameter(torch.rand([1, s], dtype=torch.float))
                self.b = nn.Parameter(torch.rand([1, s], dtype=torch.float))

        self.o = nn.Linear(self.e, out_token_dims, bias=bias)

        if attn_type == 'self-attn':
            self.uv = nn.Linear(in_token_dims, 2 * self.e + self.s, bias=bias)
            self.gamma = nn.Parameter(torch.rand((2, self.s)))
            self.beta = nn.Parameter(torch.rand((2, self.s)))
        else:
            self.uv = nn.Linear(in_token_dims, self.e + self.s, bias=bias)
            self.k_fc = nn.Linear(in_token_dims, self.s, bias=bias)
            self.v_fc = nn.Linear(in_token_dims, self.e, bias=bias)
            nn.init.xavier_uniform_(self.k_fc.weight)
            nn.init.xavier_uniform_(self.v_fc.weight)

        self.ln = ScaleNorm(in_token_dims, eps=eps)
        nn.init.xavier_uniform_(self.uv.weight)

        if act_fn == 'SiLU' or act_fn == nn.SiLU:
            self.act_fn = nn.SiLU(True)
        elif act_fn == 'ReLU' or act_fn == nn.ReLU:
            self.act_fn = nn.ReLU(True)
        else:
            self.act_fn = nn.SiLU(True)

        if in_token_dims == out_token_dims:
            self.shortcut = True
            self.res_scale = Scale(in_token_dims)
        else:
            self.shortcut = False

        self.sqrt_s = math.sqrt(s)
        self.dropout_rate = dropout_rate
        if dropout_rate > 0.:
            self.dropout = nn.Dropout(dropout_rate)

    def rel_pos_bias(self, seq_len, k_len=None):
        """Add relative position bias."""
        if self.attn_type == 'self-attn':
            t = F.pad(self.w[:2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
            t = t[..., :-seq_len].reshape(-1, seq_len, 3 * seq_len - 2)
            r = (2 * seq_len - 1) // 2
            t = t[..., r:-r]
        else:
            a = self._rope(self.a.repeat(seq_len, 1), dim=0)
            b = self._rope(self.b.repeat(k_len, 1), dim=0)
            t = torch.bmm(a, b.permute(0, 2, 1))
        return t

    def _rope(self, x, dim):
        """Rotary position embedding."""
        shape = x.shape
        if isinstance(dim, int):
            dim = [dim]
        spatial_shape = [shape[i] for i in dim]
        total_len = 1
        for i in spatial_shape:
            total_len *= i
        position = torch.reshape(
            torch.arange(total_len, dtype=torch.int, device=x.device), spatial_shape)
        for i in range(dim[-1] + 1, len(shape) - 1, 1):
            position = torch.unsqueeze(position, dim=-1)
        half_size = shape[-1] // 2
        freq_seq = -torch.arange(half_size, dtype=torch.int, device=x.device) / float(half_size)
        inv_freq = 10000 ** -freq_seq
        sinusoid = position[..., None] * inv_freq[None, None, :]
        sin = torch.sin(sinusoid)
        cos = torch.cos(sinusoid)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def _forward(self, inputs):
        """GAU Forward function."""
        if self.attn_type == 'self-attn':
            x = inputs
        else:
            x, k, v = inputs

        x = self.ln(x)
        uv = self.uv(x)
        uv = self.act_fn(uv)

        if self.attn_type == 'self-attn':
            u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=2)
            base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta
            if self.pos_enc:
                base = self._rope(base, dim=1)
            q, k = torch.unbind(base, dim=2)
        else:
            u, q = torch.split(uv, [self.e, self.s], dim=2)
            k = self.k_fc(k)
            v = self.v_fc(v)
            if self.pos_enc:
                q = self._rope(q, 1)
                k = self._rope(k, 1)

        qk = torch.bmm(q, k.permute(0, 2, 1))

        if self.use_rel_bias:
            if self.attn_type == 'self-attn':
                bias = self.rel_pos_bias(q.size(1))
            else:
                bias = self.rel_pos_bias(q.size(1), k.size(1))
            qk += bias[:, :q.size(1), :k.size(1)]

        kernel = torch.square(F.relu(qk / self.sqrt_s))

        if self.dropout_rate > 0.:
            kernel = self.dropout(kernel)

        x = u * torch.bmm(kernel, v)
        x = self.o(x)
        return x

    def forward(self, x):
        """Forward function."""
        if self.shortcut:
            if self.attn_type == 'cross-attn':
                res_shortcut = x[0]
            else:
                res_shortcut = x
            main_branch = self._forward(x)
            return self.res_scale(res_shortcut) + main_branch
        else:
            return self._forward(x)


class ConvModule(nn.Module):
    """Conv + BN + Act module."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch, momentum=0.03, eps=0.001)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise separable conv: depthwise + pointwise."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise_conv = ConvModule(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch)
        self.pointwise_conv = ConvModule(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class ChannelAttention(nn.Module):
    """Channel attention module."""
    def __init__(self, channels):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            out = self.global_avgpool(x.float())
        out = self.fc(out)
        out = self.act(out)
        return x * out


class CSPNeXtBlock(nn.Module):
    """CSPNeXt block: conv1 (3x3) + DepthwiseSeparable (5x5)."""
    def __init__(self, in_ch, out_ch, expansion=0.5, add_identity=True, kernel_size=5):
        super().__init__()
        hidden_ch = int(out_ch * expansion)
        self.conv1 = ConvModule(in_ch, hidden_ch, 3, stride=1, padding=1)
        self.conv2 = DepthwiseSeparableConvModule(
            hidden_ch, out_ch, kernel_size, stride=1, padding=kernel_size // 2)
        self.add_identity = add_identity and in_ch == out_ch

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.add_identity:
            return out + identity
        return out


class CSPLayer(nn.Module):
    """Cross Stage Partial Layer with channel attention."""
    def __init__(self, in_ch, out_ch, num_blocks=1, add_identity=True,
                 expand_ratio=0.5, channel_attention=True):
        super().__init__()
        mid_ch = int(out_ch * expand_ratio)
        self.channel_attention = channel_attention
        self.main_conv = ConvModule(in_ch, mid_ch, 1)
        self.short_conv = ConvModule(in_ch, mid_ch, 1)
        self.final_conv = ConvModule(2 * mid_ch, out_ch, 1)
        self.blocks = nn.Sequential(*[
            CSPNeXtBlock(mid_ch, mid_ch, expansion=1.0, add_identity=add_identity)
            for _ in range(num_blocks)
        ])
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_ch)

    def forward(self, x):
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)
        x_final = torch.cat((x_main, x_short), dim=1)
        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)


class SPPBottleneck(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer."""
    def __init__(self, in_ch, out_ch, kernel_sizes=(5, 9, 13)):
        super().__init__()
        mid_ch = in_ch // 2
        self.conv1 = ConvModule(in_ch, mid_ch, 1)
        self.conv2 = ConvModule(mid_ch * (len(kernel_sizes) + 1), out_ch, 1)
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])

    def forward(self, x):
        x = self.conv1(x)
        feats = [x] + [pool(x) for pool in self.poolings]
        x = torch.cat(feats, dim=1)
        x = self.conv2(x)
        return x


class CSPNeXt(nn.Module):
    """CSPNeXt backbone for RTMPose-M (widen_factor=0.75)."""
    # arch_settings P5: [in_ch, out_ch, num_blocks, add_identity, use_spp]
    arch_settings = [
        [64, 128, 3, True, False],   # stage1
        [128, 256, 6, True, False],  # stage2
        [256, 512, 6, True, False],  # stage3
        [512, 1024, 3, False, True], # stage4
    ]

    def __init__(self, widen_factor=0.75, deepen_factor=0.67, out_indices=(4,)):
        super().__init__()
        self.out_indices = out_indices

        # Stem: 3 -> 24 -> 24 -> 48 (with widen_factor=0.75)
        base_ch = int(self.arch_settings[0][0] * widen_factor)  # 48
        stem_ch = base_ch // 2  # 24
        self.stem = nn.Sequential(
            ConvModule(3, stem_ch, 3, stride=2, padding=1),       # 3 -> 24
            ConvModule(stem_ch, stem_ch, 3, stride=1, padding=1), # 24 -> 24
            ConvModule(stem_ch, base_ch, 3, stride=1, padding=1), # 24 -> 48
        )

        # Build stages
        for i, (in_ch, out_ch, num_blocks, add_identity, use_spp) in enumerate(self.arch_settings):
            in_ch = int(in_ch * widen_factor)
            out_ch = int(out_ch * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)

            stage = nn.Sequential()
            # Downsample conv
            stage.add_module('0', ConvModule(in_ch, out_ch, 3, stride=2, padding=1))
            # SPP (only for stage4)
            if use_spp:
                stage.add_module('1', SPPBottleneck(out_ch, out_ch))
                stage.add_module('2', CSPLayer(out_ch, out_ch, num_blocks, add_identity))
            else:
                stage.add_module('1', CSPLayer(out_ch, out_ch, num_blocks, add_identity))
            setattr(self, f'stage{i+1}', stage)

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i in range(4):
            stage = getattr(self, f'stage{i+1}')
            x = stage(x)
            if (i + 1) in self.out_indices:
                outs.append(x)
        return tuple(outs)


class RTMCCHead(nn.Module):
    """RTMPose head with SimCC output - mmpose compatible."""
    def __init__(self, in_channels=768, out_channels=133, input_size=(256, 192),
                 in_featuremap_size=(8, 6), simcc_split_ratio=2.0,
                 final_layer_kernel_size=7, gau_cfg=None):
        super().__init__()
        if gau_cfg is None:
            gau_cfg = dict(hidden_dims=256, s=128, expansion_factor=2,
                           dropout_rate=0., drop_path=0., act_fn='SiLU',
                           use_rel_bias=False, pos_enc=False)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio

        flatten_dims = in_featuremap_size[0] * in_featuremap_size[1]

        # Final layer: conv2d with kernel_size padding to preserve spatial dims
        self.final_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size=final_layer_kernel_size,
            stride=1, padding=final_layer_kernel_size // 2)

        # MLP: ScaleNorm + Linear
        self.mlp = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, gau_cfg['hidden_dims'], bias=False))

        W = int(input_size[1] * simcc_split_ratio)  # 192 * 2 = 384
        H = int(input_size[0] * simcc_split_ratio)  # 256 * 2 = 512

        # GAU (Gated Attention Unit)
        self.gau = RTMCCBlock(
            out_channels, gau_cfg['hidden_dims'], gau_cfg['hidden_dims'],
            s=gau_cfg['s'], expansion_factor=gau_cfg['expansion_factor'],
            dropout_rate=gau_cfg['dropout_rate'], drop_path=gau_cfg.get('drop_path', 0.),
            attn_type='self-attn', act_fn=gau_cfg['act_fn'],
            use_rel_bias=gau_cfg['use_rel_bias'], pos_enc=gau_cfg['pos_enc'])

        # Classification layers for x and y coordinates
        self.cls_x = nn.Linear(gau_cfg['hidden_dims'], W, bias=False)
        self.cls_y = nn.Linear(gau_cfg['hidden_dims'], H, bias=False)

    def forward(self, feats):
        """Forward pass."""
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]

        feats = self.final_layer(feats)      # B, K, H, W
        feats = torch.flatten(feats, 2)       # B, K, H*W
        feats = self.mlp(feats)               # B, K, hidden
        feats = self.gau(feats)               # B, K, hidden
        pred_x = self.cls_x(feats)            # B, K, W*ratio
        pred_y = self.cls_y(feats)            # B, K, H*ratio
        return pred_x, pred_y


class RTMPoseModel(nn.Module):
    """RTMPose model (mmpose compatible) for whole-body pose estimation."""
    def __init__(self, num_keypoints=133, input_size=(256, 192)):
        super().__init__()
        self.input_size = input_size  # H, W
        self.num_keypoints = num_keypoints

        # CSPNeXt-M backbone (widen_factor=0.75, deepen_factor=0.67)
        self.backbone = CSPNeXt(widen_factor=0.75, deepen_factor=0.67, out_indices=(4,))

        # Feature map size after backbone (stride=32)
        feat_h = input_size[0] // 32  # 256 / 32 = 8
        feat_w = input_size[1] // 32  # 192 / 32 = 6
        backbone_out_ch = int(1024 * 0.75)  # 768

        # RTMCCHead
        self.head = RTMCCHead(
            in_channels=backbone_out_ch,
            out_channels=num_keypoints,
            input_size=input_size,
            in_featuremap_size=(feat_h, feat_w),
            simcc_split_ratio=2.0,
            final_layer_kernel_size=7,
            gau_cfg=dict(hidden_dims=256, s=128, expansion_factor=2,
                         dropout_rate=0., act_fn='SiLU',
                         use_rel_bias=False, pos_enc=False))

    def forward(self, x):
        feats = self.backbone(x)
        pred_x, pred_y = self.head(feats)
        return pred_x, pred_y

    def decode(self, pred_x, pred_y, simcc_split_ratio=2.0):
        """Decode SimCC predictions to keypoints."""
        # Get argmax locations
        x_locs = torch.argmax(pred_x, dim=-1).float() / simcc_split_ratio
        y_locs = torch.argmax(pred_y, dim=-1).float() / simcc_split_ratio

        # Get confidence from softmax max values
        x_conf = torch.softmax(pred_x, dim=-1).max(dim=-1)[0]
        y_conf = torch.softmax(pred_y, dim=-1).max(dim=-1)[0]
        conf = (x_conf + y_conf) / 2

        return x_locs, y_locs, conf


# ============================================================
# Hand Tracking Pipeline
# ============================================================

class HandTrackingPipeline:
    """RTMPose-based hand tracking (Pure PyTorch, mmpose compatible)."""

    def __init__(self, precision: str = "fp16", device: str = None):
        self.precision = precision
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.input_size = (256, 192)  # H, W

        print(f"[HandTracking] Device: {self.device}, Precision: {precision}")
        print(f"[HandTracking] Loading RTMPose model...")

        # Initialize model
        self.model = RTMPoseModel(num_keypoints=133, input_size=self.input_size)

        # Load weights
        if MODEL_PATH.exists():
            try:
                state_dict = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=False)
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]

                # Load with strict=False to handle any minor mismatches
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"[HandTracking] Missing keys: {len(missing)}")
                if unexpected:
                    print(f"[HandTracking] Unexpected keys: {len(unexpected)}")
                print(f"[HandTracking] Loaded weights from {MODEL_PATH.name}")
            except Exception as e:
                print(f"[HandTracking] Warning: Could not load weights: {e}")
                print(f"[HandTracking] Using random initialization")
        else:
            print(f"[HandTracking] Warning: Model not found at {MODEL_PATH}")
            print(f"[HandTracking] Using random initialization")

        self.model = self.model.to(self.device)
        self.model.eval()

        # FP16 for faster inference
        if precision == "fp16" and self.device.type == "cuda":
            self.model = self.model.half()
            self.use_fp16 = True
        else:
            self.use_fp16 = False

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for model input."""
        # Resize to input size (W, H)
        img = cv2.resize(frame, (self.input_size[1], self.input_size[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        # To tensor (B, C, H, W)
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)

        if self.use_fp16:
            tensor = tensor.half()

        return tensor

    def process_frame(self, frame: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray, float, Optional[np.ndarray]]:
        """Process frame and return hand landmarks + face info.

        Returns:
            landmarks_list: List of hand landmarks arrays
            detections: Detection boxes
            mar: Mouth aspect ratio (for smile detection)
            mouth_center: Mouth center position [x, y] or None
        """
        h, w = frame.shape[:2]

        with torch.no_grad():
            # Preprocess
            input_tensor = self._preprocess(frame)

            # Inference
            pred_x, pred_y = self.model(input_tensor)

            # Decode
            x_locs, y_locs, conf = self.model.decode(pred_x, pred_y)

            # Convert to numpy
            x_locs = x_locs[0].cpu().numpy()
            y_locs = y_locs[0].cpu().numpy()
            conf = conf[0].cpu().numpy()

        # Scale to original frame size
        scale_x = w / self.input_size[1]
        scale_y = h / self.input_size[0]

        landmarks_list = []
        detections = []

        # Extract hand keypoints
        for hand_indices in [LEFT_HAND_INDICES, RIGHT_HAND_INDICES]:
            hand_conf = conf[hand_indices]

            # Skip if low confidence
            if hand_conf.mean() < 0.3:
                continue

            # Get hand landmarks
            hand_x = x_locs[hand_indices] * scale_x
            hand_y = y_locs[hand_indices] * scale_y
            hand_landmarks = np.stack([hand_x, hand_y, hand_conf], axis=1)

            landmarks_list.append(hand_landmarks)

            # Create detection box
            x_min, x_max = hand_x.min(), hand_x.max()
            y_min, y_max = hand_y.min(), hand_y.max()
            detections.append([x_min, y_min, x_max, y_max, hand_conf.mean()])

        # Extract face info (MAR and mouth center)
        mar = 0.0
        mouth_center = None

        # Check face confidence (use average of mouth keypoints)
        mouth_indices = [MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT]
        mouth_conf = conf[mouth_indices].mean()

        if mouth_conf > 0.3:
            # Calculate MAR (mouth aspect ratio)
            top_y = y_locs[MOUTH_TOP] * scale_y
            bottom_y = y_locs[MOUTH_BOTTOM] * scale_y
            left_x = x_locs[MOUTH_LEFT] * scale_x
            right_x = x_locs[MOUTH_RIGHT] * scale_x

            mouth_height = abs(bottom_y - top_y)
            mouth_width = abs(right_x - left_x)

            if mouth_width > 1:
                mar = mouth_height / mouth_width

            # Calculate mouth center
            center_x = np.mean([x_locs[i] for i in MOUTH_CENTER_INDICES]) * scale_x
            center_y = np.mean([y_locs[i] for i in MOUTH_CENTER_INDICES]) * scale_y
            mouth_center = np.array([center_x, center_y])

        return landmarks_list, np.array(detections), mar, mouth_center

    def print_stats(self):
        """Print pipeline statistics."""
        print("\n" + "=" * 50)
        print("Hand Tracking - RTMPose (Pure PyTorch)")
        print("=" * 50)
        print(f"Device:     {self.device}")
        print(f"Precision:  {self.precision}")
        print(f"Input size: {self.input_size[1]}x{self.input_size[0]}")
        print(f"FP16:       {'Yes' if self.use_fp16 else 'No'}")
        print("=" * 50 + "\n")


# Alias
BlazeHandTrackingPipeline = HandTrackingPipeline


if __name__ == "__main__":
    pipeline = HandTrackingPipeline(precision="fp32")
    pipeline.print_stats()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Press ESC to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame[:, ::-1].copy()
        start = time.time()
        landmarks, detections, _, _ = pipeline.process_frame(frame)
        fps = 1.0 / (time.time() - start + 1e-6)

        for lm in landmarks:
            draw_landmarks(frame, lm)
        for det in detections:
            draw_detections(frame, np.array([det]), 1.0, (0, 0))

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("RTMPose Hand", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
