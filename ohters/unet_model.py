import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """基础卷积块：Conv + BN + ReLU ×2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    注意力门控模块 (Attention Gate)

    来源: "Attention U-Net: Learning Where to Look for the Pancreas" (Oktay et al., 2018)

    工作原理:
    1. 分别处理上采样特征(gating signal)和跳跃连接特征(skip connection)
    2. 通过加性注意力计算特征间的相关性
    3. 生成0-1范围的注意力系数，对跳跃连接特征进行软加权
    4. 使模型聚焦于裂缝区域，抑制背景噪声

    参数:
        F_g: 门控信号(上采样特征)的通道数
        F_l: 跳跃连接特征的通道数
        F_int: 中间特征通道数(降维用，减少计算量)
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        # 处理门控信号(上采样特征) → 降维到 F_int
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        # 处理跳跃连接特征 → 降维到 F_int
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        # 生成注意力系数图 → 单通道输出
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()  # 将注意力系数限制在 [0, 1]
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        参数:
            g: 门控信号 — 来自解码器上采样路径 (决定"关注哪里")
            x: 跳跃连接特征 — 来自编码器 (提供"要关注什么")
        返回:
            加权后的跳跃连接特征，形状与 x 相同
        """
        # Step 1: 对齐通道维度
        g1 = self.W_g(g)   # (B, F_int, H, W)
        x1 = self.W_x(x)   # (B, F_int, H, W)

        # Step 2: 加性融合 + 激活
        psi = self.relu(g1 + x1)

        # Step 3: 生成注意力系数 α ∈ [0, 1]
        alpha = self.psi(psi)

        # Step 4: 用注意力系数加权跳跃连接特征
        return x * alpha


class AttentionUpBlock(nn.Module):
    """
    带注意力门控的上采样块

    流程: 上采样 → 注意力门控 → 拼接 → 卷积
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # 注意力门控: 用上采样特征作为门控信号，筛选跳跃连接特征
        self.att = AttentionGate(F_g=out_channels, F_l=out_channels, F_int=out_channels // 2)
        # 拼接后通道数 = out_channels(skip) + out_channels(up) = 2 * out_channels
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        """
        参数:
            x1: 深层解码器特征 (低分辨率，语义信息丰富)
            x2: 浅层编码器特征 (高分辨率，空间细节丰富)
        返回:
            融合后的特征图
        """
        # 上采样
        x1 = self.up(x1)

        # 尺寸对齐 (处理奇偶尺寸差异)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # 注意力门控: 用上采样特征指导跳跃连接特征的筛选
        x2 = self.att(g=x1, x=x2)

        # 拼接 + 卷积
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net — 专为裂缝检测优化

    架构总览:
    =========

    编码器 (下采样路径):
        down1: ConvBlock(3 → 64)    256×256
        down2: ConvBlock(64 → 128)  128×128
        down3: ConvBlock(128 → 256)  64×64
        down4: ConvBlock(256 → 512)  32×32
        bottleneck: ConvBlock(512 → 1024)  16×16

    解码器 (上采样路径 + 注意力门控):
        up4: AttentionUpBlock(1024 → 512)  32×32
        up3: AttentionUpBlock(512 → 256)   64×64
        up2: AttentionUpBlock(256 → 128)  128×128
        up1: AttentionUpBlock(128 → 64)   256×256

    输出头:
        out: Conv2d(64 → 1)  256×256

    参数量: ~34.9M (适合 8GB 显存训练)

    与原始 UNet++ 的关键区别:
    1. 注意力门控 → 模型学会"关注"裂缝区域，抑制背景
    2. 标准跳跃连接 + 注意力筛选 → 比嵌套跳跃连接更轻量
    3. 加性注意力 → 计算高效，梯度稳定
    """
    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ===== 编码器 (Encoder) =====
        self.down1 = ConvBlock(in_channels, base_channels)        # 64ch
        self.pool = nn.MaxPool2d(2)

        self.down2 = ConvBlock(base_channels, base_channels * 2)   # 128ch
        self.down3 = ConvBlock(base_channels * 2, base_channels * 4)  # 256ch
        self.down4 = ConvBlock(base_channels * 4, base_channels * 8)  # 512ch

        # ===== 瓶颈层 (Bottleneck) =====
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)  # 1024ch

        # ===== 解码器 (Decoder with Attention Gates) =====
        self.up4 = AttentionUpBlock(base_channels * 16, base_channels * 8)   # 1024→512
        self.up3 = AttentionUpBlock(base_channels * 8, base_channels * 4)    # 512→256
        self.up2 = AttentionUpBlock(base_channels * 4, base_channels * 2)    # 256→128
        self.up1 = AttentionUpBlock(base_channels * 2, base_channels)         # 128→64

        # ===== 输出头 =====
        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # --- 编码 ---
        x1 = self.down1(x)                          # (B, 64,   H,   W)
        x2 = self.down2(self.pool(x1))              # (B, 128,  H/2, W/2)
        x3 = self.down3(self.pool(x2))              # (B, 256,  H/4, W/4)
        x4 = self.down4(self.pool(x3))              # (B, 512,  H/8, W/8)
        x5 = self.bottleneck(self.pool(x4))          # (B, 1024, H/16, W/16)

        # --- 解码 (Attention Gates 在跳跃连接上) ---
        d4 = self.up4(x5, x4)                       # (B, 512,  H/8, W/8)
        d3 = self.up3(d4, x3)                       # (B, 256,  H/4, W/4)
        d2 = self.up2(d3, x2)                       # (B, 128,  H/2, W/2)
        d1 = self.up1(d2, x1)                       # (B, 64,   H,   W)

        # --- 输出 ---
        out = self.out(d1)                           # (B, 1, H, W)
        return out


# 别名: 保持与旧代码的兼容性
UNet = AttentionUNet


if __name__ == "__main__":
    # 测试网络前向传播
    model = AttentionUNet(in_channels=3, out_channels=1)
    test_x = torch.randn(4, 3, 384, 384)
    pred = model(test_x)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"输入shape: {test_x.shape}")
    print(f"输出分割图shape: {pred.shape}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 验证在 384×384 输入下的各层尺寸
    print("\n=== 各层输出尺寸 (384×384 输入) ===")
    model.eval()
    with torch.no_grad():
        x1 = model.down1(test_x)
        print(f"down1:       {x1.shape}")
        x2 = model.down2(model.pool(x1))
        print(f"down2:       {x2.shape}")
        x3 = model.down3(model.pool(x2))
        print(f"down3:       {x3.shape}")
        x4 = model.down4(model.pool(x3))
        print(f"down4:       {x4.shape}")
        x5 = model.bottleneck(model.pool(x4))
        print(f"bottleneck:  {x5.shape}")
        d4 = model.up4(x5, x4)
        print(f"up4:         {d4.shape}")
        d3 = model.up3(d4, x3)
        print(f"up3:         {d3.shape}")
        d2 = model.up2(d3, x2)
        print(f"up2:         {d2.shape}")
        d1 = model.up1(d2, x1)
        print(f"up1:         {d1.shape}")
