"""
CrackAwareNet: A continuity-aware lightweight network for fine crack segmentation

论文: Lin et al. (2026), Structures, Vol. 88
     https://www.sciencedirect.com/science/article/pii/S2352012426007605

架构:
  1. Hierarchical VMamba Encoder — 视觉状态空间模型 (线性复杂度 O(N))
  2. Cross-scale Dynamic Attention Module (CDAM) — 跨尺度动态融合
  3. Selective Gated Fusion Module (SGFM) — 选择性门控融合

参数量: ~17.8M (Tiny 配置)
适配:   RTX 4060 Laptop 8GB, 384×384 输入
依赖:   torch >= 1.12, einops (可选: mamba_ssm 加速)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple


# ============================================================
# 工具函数
# ============================================================

_HAS_MAMBA_SSM = False
try:
    import mamba_ssm
    _HAS_MAMBA_SSM = True
except ImportError:
    pass


# ============================================================
# 1. Cross Scan & Cross Merge (2D ←→ 4×1D)
# ============================================================

def cross_scan_2d(x: torch.Tensor) -> torch.Tensor:
    """
    二维交叉扫描 — 沿 4 个方向展开为 1D 序列
    方向: 行优先, 行逆序, 列优先, 列逆序

    参数:
        x: (B, C, H, W)
    返回:
        (B, 4, C, H*W)
    """
    B, C, H, W = x.shape
    # 方向 1: 行优先
    s1 = x.view(B, C, -1)
    # 方向 2: 行逆序
    s2 = torch.flip(s1, dims=[-1])
    # 方向 3: 列优先 (转置后行优先)
    x_t = x.transpose(-2, -1).contiguous()  # (B, C, W, H)
    s3 = x_t.view(B, C, -1)
    # 方向 4: 列逆序
    s4 = torch.flip(s3, dims=[-1])
    return torch.stack([s1, s2, s3, s4], dim=1)


def cross_merge_2d(y: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    二维交叉合并 — 4 方向 1D 序列平均合并回 2D

    参数:
        y: (B, 4, C, H*W)
        H, W: 原始空间尺寸
    返回:
        (B, C, H, W)
    """
    B, _, C, _ = y.shape
    # 方向 1: reshape 回 (H, W)
    o1 = y[:, 0].view(B, C, H, W)
    # 方向 2: 翻转回来再 reshape
    o2 = torch.flip(y[:, 1], dims=[-1]).view(B, C, H, W)
    # 方向 3: reshape (W, H) 再转置
    o3 = y[:, 2].view(B, C, W, H).transpose(-2, -1)
    # 方向 4: 翻转 → reshape (W, H) → 转置
    o4 = torch.flip(y[:, 3], dims=[-1]).view(B, C, W, H).transpose(-2, -1)
    return (o1 + o2 + o3 + o4) / 4.0


# ============================================================
# 2. Selective Scan (纯 PyTorch + CUDA 回退)
# ============================================================

def selective_scan_pytorch(
    u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
    B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
    z: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    选择性扫描 S6 — 纯 PyTorch 循环实现

    h(t) = exp(Δ·A) · h(t-1) + Δ·B(t) · x(t)
    y(t) = C(t) · h(t) + D · x(t)

    参数:
        u:     (B, D, L)   输入
        delta: (B, D, L)   步长
        A:     (D, N)      状态矩阵
        B:     (B, N, L)   输入投影
        C:     (B, N, L)   输出投影
        D:     (D,)        跳跃连接
        z:     (B, D, L)   可选门控
    返回:
        y: (B, D, L)
    """
    B_batch, D_dim, L = u.shape  # D_dim = d_inner (NOT the D param!)
    N = A.shape[-1]
    device, dtype = u.device, u.dtype

    # 确保 delta 为正（等效于 mamba_ssm 的 delta_softplus=True）
    delta = F.softplus(delta)

    # 离散化: dA = exp(delta * A), dB = delta * B
    dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(-2))  # (B,D_dim,L,N)
    # B: (B,N,L) → (B,1,L,N) 用于广播
    B_exp = B.unsqueeze(1).permute(0, 1, 3, 2)  # (B,1,L,N)
    dB = delta.unsqueeze(-1) * B_exp  # (B,D_dim,L,N)

    # 循环扫描
    h = torch.zeros(B_batch, D_dim, N, device=device, dtype=dtype)
    outs = []
    for t in range(L):
        h = dA[:, :, t] * h + dB[:, :, t] * u[:, :, t].unsqueeze(-1)
        # C[:,t]: (B,N) → (B,1,N)
        y_t = (h * C.unsqueeze(1)[:, :, :, t]).sum(dim=-1)
        if D is not None:
            y_t = y_t + D.unsqueeze(0) * u[:, :, t]
        outs.append(y_t)

    y = torch.stack(outs, dim=-1)  # (B,D,L)
    if z is not None:
        y = y * F.silu(z)
    return y


def selective_scan_cuda(u, delta, A, B, C, D, z=None):
    """CUDA 加速版本 (mamba_ssm) — 统一 dtype 处理"""
    from mamba_ssm import selective_scan_fn
    # AMP 下 depthwise conv→float32, Linear→float16, 需对齐
    if delta.dtype != u.dtype:
        delta = delta.to(u.dtype)
    if isinstance(B, torch.Tensor) and B.dtype != u.dtype:
        B = B.to(u.dtype)
    if isinstance(C, torch.Tensor) and C.dtype != u.dtype:
        C = C.to(u.dtype)
    # A, D 必须为 float32（mamba_ssm 硬性要求）
    if A.dtype != torch.float32:
        A = A.to(torch.float32)
    if isinstance(D, torch.Tensor) and D.dtype != torch.float32:
        D = D.to(torch.float32)
    return selective_scan_fn(u, delta, A, B, C, D, z=z, delta_bias=None, delta_softplus=True)


class SelectiveScanFn(nn.Module):
    """选择性扫描 — 自动选择 CUDA/纯PyTorch"""
    def forward(self, u, delta, A, B, C, D, z=None):
        if _HAS_MAMBA_SSM and u.is_cuda:
            return selective_scan_cuda(u, delta, A, B, C, D, z)
        return selective_scan_pytorch(u, delta, A, B, C, D, z)


# ============================================================
# 3. SS2D: 2D Selective Scan Module
# ============================================================

class SS2D(nn.Module):
    """
    2D 选择性扫描模块 (VMamba 核心)

    流程: x → in_proj → split(x,z)
                x → conv1d → SiLU → x_proj → split(dt,B,C)
                → selective_scan → × SiLU(z) → out_proj

    参数:
        d_model:   输入/输出通道
        d_state:   SSM 状态维度 (default: 16)
        dt_rank:   步长秩 (default: 0 → auto = ceil(d_model/16))
        d_conv:    1D 卷积核大小 (default: 3)
        expand:    内部扩展因子 (default: 1 → 无扩展)
    """
    def __init__(self, d_model: int, d_state: int = 16,
                 dt_rank: int = 0, d_conv: int = 3, expand: int = 1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(d_model * expand)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank <= 0 else dt_rank

        # 输入投影: d_model → 2 * d_inner (x + z 两条路径)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D 深度可分离卷积
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            kernel_size=d_conv, groups=self.d_inner,
            padding=d_conv - 1, bias=False,
        )

        # SSM 参数投影: d_inner → (dt_rank + 2 * d_state)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)

        # dt 投影: dt_rank → d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A 参数: log 空间初始化 (鼓励长期记忆)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.repeat(self.d_inner, 1)  # (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A))

        # D 参数: 跳跃连接
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.scan = SelectiveScanFn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (B, C, H, W)  (C == d_model)
        返回:
            (B, C, H, W)
        """
        B, C, H, W = x.shape
        L = H * W

        # 交叉扫描: 2D → 4×1D
        seqs = cross_scan_2d(x)  # (B, 4, C, L)

        outputs = []
        for d in range(4):
            s = seqs[:, d]  # (B, C, L)

            # in_proj: (B, C, L) → Linear → (B, 2*d_inner, L)
            s_t = s.permute(0, 2, 1)  # (B, L, C)
            xz = self.in_proj(s_t)   # (B, L, 2*d_inner)
            xz = xz.permute(0, 2, 1)  # (B, 2*d_inner, L)
            x_in, z = xz.chunk(2, dim=1)  # 各 (B, d_inner, L)

            # 1D 卷积 + SiLU
            x_conv = self.conv1d(x_in)[:, :, :L]
            x_conv = F.silu(x_conv)  # (B, d_inner, L)

            # 生成 dt, B, C 参数
            x_dbl = x_conv.permute(0, 2, 1)  # (B, L, d_inner)
            x_proj_out = self.x_proj(x_dbl)  # (B, L, dt_rank+2*d_state)
            dt_in, B_in, C_in = torch.split(
                x_proj_out, [self.dt_rank, self.d_state, self.d_state], dim=-1
            )

            # dt: 序列均值 → 投影（softplus 由 mamba_ssm 内部处理）
            dt = self.dt_proj(dt_in.mean(dim=1))  # (B, d_inner)
            dt = dt.unsqueeze(-1).expand(-1, -1, L)  # (B, d_inner, L)

            # B, C: (B, L, d_state) → (B, d_state, L)
            B_mat = B_in.permute(0, 2, 1).contiguous()
            C_mat = C_in.permute(0, 2, 1).contiguous()

            # A: (d_inner, d_state), 负值确保稳定性
            A_mat = -torch.exp(self.A_log)  # (d_inner, d_state)

            # 选择性扫描
            y = self.scan(x_conv, dt, A_mat, B_mat, C_mat, self.D)  # (B, d_inner, L)

            # 门控 + 输出投影
            y = y * F.silu(z)  # (B, d_inner, L)
            y = y.permute(0, 2, 1)  # (B, L, d_inner)
            y = self.out_proj(y)   # (B, L, C)
            y = y.permute(0, 2, 1)  # (B, C, L)

            outputs.append(y)

        # 交叉合并: 4×1D → 2D
        y_stack = torch.stack(outputs, dim=1)  # (B, 4, C, L)
        return cross_merge_2d(y_stack, H, W)  # (B, C, H, W)


# ============================================================
# 4. VSS Block + ConvFFN
# ============================================================

class ConvFFN(nn.Module):
    """
    卷积前馈网络 (含深度可分离卷积)

    Linear → DWConv → SiLU → Linear

    参数:
        hidden_dim:   输入/输出通道
        mlp_ratio:    扩展比率 (default: 2)
    """
    def __init__(self, hidden_dim: int, mlp_ratio: float = 2.0):
        super().__init__()
        mlp_hidden = int(hidden_dim * mlp_ratio)

        self.fc1 = nn.Linear(hidden_dim, mlp_hidden, bias=False)
        self.dwconv = nn.Conv2d(
            mlp_hidden, mlp_hidden, kernel_size=3, padding=1,
            groups=mlp_hidden, bias=False
        )
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(mlp_hidden, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (B, C, H, W)
        返回:
            (B, C, H, W)
        """
        B, C, H, W = x.shape

        # fc1: (C → mlp_hidden) in Linear
        x = x.view(B, C, -1).transpose(1, 2)  # (B, L, C)
        x = self.fc1(x)  # (B, L, mlp_hidden)
        x = x.transpose(1, 2).view(B, -1, H, W)  # (B, mlp_hidden, H, W)

        # DWConv + Activation
        x = self.dwconv(x)
        x = self.act(x)

        # fc2: (mlp_hidden → C)
        x = x.view(B, -1, H * W).transpose(1, 2)  # (B, L, mlp_hidden)
        x = self.fc2(x)  # (B, L, C)
        return x.transpose(1, 2).view(B, C, H, W)  # (B, C, H, W)


class VSSBlock(nn.Module):
    """
    Visual State Space Block

    LN → SS2D → + → LN → ConvFFN → +

    参数:
        hidden_dim: 隐藏通道数
        mlp_ratio:  FFN 扩展比率
        **ss2d_kwargs: SS2D 参数
    """
    def __init__(self, hidden_dim: int, mlp_ratio: float = 2.0, **ss2d_kwargs):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ss2d = SS2D(d_model=hidden_dim, **ss2d_kwargs)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = ConvFFN(hidden_dim, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # SS2D 分支
        identity = x
        x_n = x.view(B, C, -1).transpose(1, 2)  # (B, L, C)
        x_n = self.norm1(x_n)
        x_n = x_n.transpose(1, 2).view(B, C, H, W)
        x = identity + self.ss2d(x_n)

        # ConvFFN 分支
        identity = x
        x_n = x.view(B, C, -1).transpose(1, 2)
        x_n = self.norm2(x_n)
        x_n = x_n.transpose(1, 2).view(B, C, H, W)
        x = identity + self.ffn(x_n)

        return x


# ============================================================
# 5. Patch Embedding & Merging
# ============================================================

class PatchEmbed(nn.Module):
    """图像 → Patch 嵌入 (4×4 卷积)"""
    def __init__(self, in_chs: int = 3, embed_dim: int = 96, patch_size: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(in_chs, embed_dim, patch_size, patch_size, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2)  # (B, L, C)
        x = self.norm(x)
        return x.transpose(1, 2).view(B, C, H, W)


class PatchMerging(nn.Module):
    """2×2 邻域合并, 通道翻倍"""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"尺寸 {H}×{W} 须为偶数"

        # 2×2 邻域展开
        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x_cat = torch.cat([x0, x1, x2, x3], dim=1)  # (B, 4C, H/2, W/2)

        x_f = x_cat.view(B, 4 * C, -1).transpose(1, 2)
        x_f = self.norm(x_f)
        x_f = self.reduction(x_f)
        return x_f.transpose(1, 2).view(B, 2 * C, H // 2, W // 2)


# ============================================================
# 6. Hierarchical VMamba Encoder
# ============================================================

class HierarchicalVMambaEncoder(nn.Module):
    """
    4 阶段分层编码器
    Stage1: PatchEmbed(×4) → VSSBlock×d1  (dim)
    Stage2: PatchMerging  → VSSBlock×d2  (dim×2)
    Stage3: PatchMerging  → VSSBlock×d3  (dim×4)
    Stage4: PatchMerging  → VSSBlock×d4  (dim×8)
    """
    def __init__(self, in_chs: int = 3, embed_dim: int = 96,
                 depths: List[int] = [2, 2, 6, 2], d_state: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.depths = depths
        num_stages = len(depths)

        self.patch_embed = PatchEmbed(in_chs, embed_dim, patch_size=4)

        self.stages = nn.ModuleList()
        for i in range(num_stages):
            stage = nn.ModuleDict()

            # Patch Merging (除 stage 0 外)
            if i == 0:
                dim_in = embed_dim
                stage['merging'] = nn.Identity()
            else:
                dim_in = embed_dim * (2 ** (i - 1))
                stage['merging'] = PatchMerging(dim=dim_in)

            dim_out = embed_dim * (2 ** i)

            blocks = nn.ModuleList([
                VSSBlock(hidden_dim=dim_out) for _ in range(depths[i])
            ])
            stage['blocks'] = blocks
            self.stages.append(stage)

        self.out_channels = [embed_dim * (2 ** i) for i in range(num_stages)]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        feats = []
        for stage in self.stages:
            x = stage['merging'](x)
            for block in stage['blocks']:
                x = block(x)
            feats.append(x)
        return feats  # [F1, F2, F3, F4]


# ============================================================
# 7. CDAM: Cross-scale Dynamic Attention Module
# ============================================================

class CDAM(nn.Module):
    """
    跨尺度动态注意力融合

    将低/中/高 3 个尺度的特征融合，用动态注意力权重自适应组合
    """
    def __init__(self, in_chs: List[int], out_ch: int):
        super().__init__()
        n = len(in_chs)
        self.align = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(True),
            ) for c in in_chs
        ])
        # 注意力权重生成
        self.attn_conv = nn.Sequential(
            nn.Conv2d(out_ch * n, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, n, 1),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
        )

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # 对齐通道 + 上采样到最高分辨率
        target_h, target_w = feats[-1].shape[2:]
        aligned = []
        for f, conv in zip(feats, self.align):
            f = conv(f)
            if f.shape[2] != target_h or f.shape[3] != target_w:
                f = F.interpolate(f, (target_h, target_w), mode='bilinear', align_corners=False)
            aligned.append(f)

        # 动态注意力权重 (在尺度维 softmax)
        concat = torch.cat(aligned, dim=1)
        attn = F.softmax(self.attn_conv(concat), dim=1)

        # 加权融合
        fused = sum(attn[:, i:i+1] * aligned[i] for i in range(len(feats)))
        return self.fusion(fused)


# ============================================================
# 8. SGFM: Selective Gated Fusion Module
# ============================================================

class SGFM(nn.Module):
    """
    选择性门控融合

    编码器跳跃连接(边缘细节) 与 解码器(语义) 的门控融合:
        gate = σ(conv(cat(skip, main)))
        out = gate · skip_proj + (1-gate) · main_proj
    """
    def __init__(self, skip_ch: int, main_ch: int, out_ch: int):
        super().__init__()
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.main_proj = nn.Sequential(
            nn.Conv2d(main_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.gate = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 1), nn.Sigmoid())
        self.fusion = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True))

    def forward(self, skip: torch.Tensor, main: torch.Tensor) -> torch.Tensor:
        if skip.shape[2:] != main.shape[2:]:
            skip = F.interpolate(skip, main.shape[2:], mode='bilinear', align_corners=False)
        s = self.skip_proj(skip)
        m = self.main_proj(main)
        g = self.gate(torch.cat([s, m], dim=1))
        return self.fusion(g * s + (1 - g) * m)


# ============================================================
# 9. CrackAwareNet (完整模型)
# ============================================================

class CrackAwareNet(nn.Module):
    """
    CrackAwareNet — 完整裂缝分割网络

    编码器 (Hierarchical VMamba 4 阶段):
        H/4 → H/8 → H/16 → H/32 → [F1, F2, F3, F4]

    解码器 (SGFM + 渐进上采样):
        F4 → Up → SGFM(F3, D4) → D3 (H/16)
        D3 → Up → SGFM(F2, D3) → D2 (H/8)
        D2 → Up → SGFM(F1, D2) → D1 (H/4)

    CDAM: 融合 3 个解码器尺度 → Head → 输出

    输入: (B, 3, H, W)   H,W 须为 32 的倍数
    输出: (B, 1, H, W)
    """
    def __init__(self, in_chs: int = 3, out_chs: int = 1,
                 embed_dim: int = 96, depths: List[int] = [2, 2, 6, 2],
                 dec_chs: List[int] = None):
        super().__init__()
        if dec_chs is None:
            dec_chs = [64, 128, 256]

        # 编码器
        self.encoder = HierarchicalVMambaEncoder(in_chs, embed_dim, depths)
        enc_chs = self.encoder.out_channels  # [96, 192, 384, 768]

        # 解码器 (3 层 SGFM)
        self.decoder = nn.ModuleList()
        in_ch = enc_chs[3]
        for i in reversed(range(3)):
            skip_ch = enc_chs[i]
            dch = dec_chs[2 - i]
            layer = nn.ModuleDict({
                'up': nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                'up_conv': nn.Sequential(
                    nn.Conv2d(in_ch, dch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(dch), nn.ReLU(True)),
                'sgfm': SGFM(skip_ch, dch, dch),
            })
            self.decoder.append(layer)
            in_ch = dch

        # CDAM
        self.cdam = CDAM(dec_chs, dec_chs[0])

        # 输出头 (×4 上采样)
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dec_chs[0], dec_chs[0] // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(dec_chs[0] // 2), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dec_chs[0] // 2, out_chs, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        assert H % 32 == 0 and W % 32 == 0, f"尺寸 {H}×{W} 须为32的倍数"

        # 编码
        f1, f2, f3, f4 = self.encoder(x)

        # 解码 + SGFM (dec_feats 按分辨率从低到高: [H/16, H/8, H/4])
        dec_feats = []
        xd = f4
        for i, layer in enumerate(self.decoder):
            xd = layer['up'](xd)
            xd = layer['up_conv'](xd)
            skip = [f3, f2, f1][i]
            xd = layer['sgfm'](skip, xd)
            dec_feats.append(xd)

        # CDAM: 在最高分辨率 H/4 融合 3 个尺度的特征
        target_res = dec_feats[2].shape[2:]  # H/4
        d_high = dec_feats[2]                  # 256ch, H/4
        d_mid  = F.interpolate(dec_feats[1], size=target_res, mode='bilinear', align_corners=False)  # 128ch, H/4
        d_low  = F.interpolate(dec_feats[0], size=target_res, mode='bilinear', align_corners=False)  # 64ch, H/4
        fused = self.cdam([d_low, d_mid, d_high])  # 对齐通道: [64, 128, 256]

        return self.head(fused)


# ============================================================
# 10. 工厂函数
# ============================================================

def crackawarenet_tiny(**kwargs) -> CrackAwareNet:
    """CrackAwareNet-Tiny ~17.8M 参数, 适合 8GB 显存"""
    return CrackAwareNet(embed_dim=96, depths=[2, 2, 6, 2], **kwargs)


def crackawarenet_small(**kwargs) -> CrackAwareNet:
    """CrackAwareNet-Small ~29M 参数, 适合 16GB+ 显存"""
    return CrackAwareNet(embed_dim=96, depths=[2, 2, 18, 2],
                         dec_chs=[96, 192, 384], **kwargs)


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"mamba_ssm: {'available' if _HAS_MAMBA_SSM else 'not available (pure PyTorch mode)'}")

    model = crackawarenet_tiny(in_chs=3, out_chs=1).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"CrackAwareNet-Tiny")
    print(f"{'='*60}")
    print(f"总参数量:     {total:>10,}")
    print(f"可训练参数:   {trainable:>10,}")
    print(f"模型大小:     {total * 4 / 1024**2:>8.1f} MB (FP32)")

    # 前向测试
    x = torch.randn(2, 3, 384, 384).to(device)
    with torch.no_grad():
        out = model(x)
    print(f"输入:  {x.shape}")
    print(f"输出:  {out.shape}")

    # 各层输出
    model.eval()
    with torch.no_grad():
        feats = model.encoder(x)
        names = ['F1(×4)', 'F2(×8)', 'F3(×16)', 'F4(×32)']
        print(f"\n编码器特征:")
        for n, f in zip(names, feats):
            print(f"  {n}: {f.shape}")

    # 显存
    if torch.cuda.is_available():
        print(f"\n显存: {torch.cuda.memory_allocated(0) / 1024**2:.0f} MB")

    print(f"\nTest passed!")
