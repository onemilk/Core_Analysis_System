"""深度学习裂缝检测模型 — CrackAwareNet / Attention U-Net 单例加载器

模型来源: ohters/ 目录下的 crackawarenet.py / unet_model.py
权重文件: best_model.pth (80MB, 项目根目录)
自动识别模型类型（根据 state_dict 键名前缀），懒加载避免重复加载。
"""

import os, sys
import cv2
import numpy as np

# 尝试导入 PyTorch（非强制依赖，无 PyTorch 时回退到纯 CV 模式）
_HAS_TORCH = False
try:
    import torch
    from torchvision import transforms
    _HAS_TORCH = True
except ImportError:
    pass


class FractureDLModel:
    """深度学习裂缝检测模型 — 单例模式，避免重复加载 80MB 权重文件

    使用方式:
        # 检查是否可用
        if FractureDLModel.is_available():
            mask = FractureDLModel.predict(bgr_image, threshold=0.3)

        # 或在 FractureAnalyzer 中自动融合（默认行为）
    """

    _model = None           # 加载后的 PyTorch 模型
    _device = None          # torch device (cuda/cpu)
    _model_type = None      # 'crackawarenet' 或 'attention_unet'
    _initialized = False    # 是否已尝试初始化（避免重复尝试失败）
    _transform = None       # torchvision 预处理变换（缓存复用）

    # ============================================================
    # 模型加载（懒加载单例）
    # ============================================================

    @classmethod
    def _ensure_model(cls):
        """确保模型已加载，返回 True/False。

        懒加载策略：首次调用时加载 80MB 权重，后续复用已加载模型。
        加载失败时记录原因，不再重复尝试。
        """
        if cls._model is not None:
            return True
        if cls._initialized:
            return False  # 已尝试过但失败了
        cls._initialized = True

        if not _HAS_TORCH:
            print("[FractureDLModel] PyTorch 未安装，使用纯CV模式")
            return False

        try:
            # 路径解析
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "best_model.pth")
            ohters_dir = os.path.join(base_dir, "ohters")

            if not os.path.exists(model_path):
                print(f"[FractureDLModel] 模型文件不存在: {model_path}")
                return False

            # 将 ohters 目录加入 Python 搜索路径（允许 import crackawarenet / unet_model）
            if ohters_dir not in sys.path:
                sys.path.insert(0, ohters_dir)

            cls._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"[FractureDLModel] 使用设备: {cls._device}")

            # 加载权重并自动识别模型类型
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            has_encoder = any(k.startswith('encoder.') for k in state_dict.keys())
            has_down = any(k.startswith('down') for k in state_dict.keys())

            if has_encoder and not has_down:
                # CrackAwareNet 的 state_dict 有 "encoder." 前缀但无 "down" 前缀
                from crackawarenet import crackawarenet_tiny
                cls._model = crackawarenet_tiny(in_chs=3, out_chs=1)
                cls._model_type = 'crackawarenet'
                print("[FractureDLModel] 检测到 CrackAwareNet 模型 (~17.8M 参数)")
            else:
                # Attention U-Net 的 state_dict 有 "down" 前缀
                from unet_model import UNet
                cls._model = UNet(in_channels=3, out_channels=1)
                cls._model_type = 'attention_unet'
                print("[FractureDLModel] 检测到 Attention U-Net 模型 (~34.9M 参数)")

            cls._model.load_state_dict(state_dict)
            cls._model.to(cls._device)
            cls._model.eval()

            # 缓存预处理变换
            cls._transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])

            # 释放 state_dict 内存
            del state_dict
            if cls._device.type == 'cuda':
                torch.cuda.empty_cache()

            print(f"[FractureDLModel] 模型加载成功，显存占用约 {torch.cuda.memory_allocated(0)/1024**2:.0f}MB" if cls._device.type == 'cuda' else "[FractureDLModel] 模型加载成功 (CPU)")
            return True

        except Exception as e:
            print(f"[FractureDLModel] 模型加载失败: {e}")
            cls._model = None
            return False

    # ============================================================
    # 裂缝预测
    # ============================================================

    @classmethod
    def predict(cls, image_bgr: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """对 BGR 图像进行深度学习裂缝检测，返回二值掩码。

        参数:
            image_bgr:  BGR 格式图像 (H, W, 3)，uint8
            threshold:  二值化阈值 (0~1)，默认 0.3（偏敏感，配合 CV 管线使用，
                        宁可多检不漏检，后续形态学+形状过滤会清理误检）
        返回:
            binary_mask: (H, W) uint8 二值掩码，255=裂缝，0=背景。
                         模型不可用时返回全零掩码。
        """
        if not cls._ensure_model():
            return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

        h, w = image_bgr.shape[:2]

        try:
            # BGR → RGB → PIL → 预处理 → 推理
            img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # 使用 PIL 直接从 numpy 创建（避免 opencv→PIL 格式问题）
            from PIL import Image
            img_pil = Image.fromarray(img_rgb)
            img_tensor = cls._transform(img_pil).unsqueeze(0).to(cls._device)

            # 前向推理
            with torch.no_grad():
                output = cls._model(img_tensor)
                prob = torch.sigmoid(output).squeeze().cpu().numpy()

            # 二值化 + 中值滤波去噪（与 predict.py 后处理一致）
            binary = (prob > threshold).astype(np.uint8) * 255

            # 缩放回原始图像尺寸（最近邻插值，保持锐利边界）
            if binary.shape != (h, w):
                binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)

            return binary

        except Exception as e:
            print(f"[FractureDLModel] 预测失败: {e}")
            return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    # ============================================================
    # 状态查询
    # ============================================================

    @classmethod
    def is_available(cls) -> bool:
        """检查深度学习模型是否已加载并可用"""
        return cls._ensure_model()

    @classmethod
    def get_model_type(cls) -> str:
        """返回模型类型: 'crackawarenet' / 'attention_unet' / 'none'"""
        if cls._ensure_model():
            return cls._model_type or 'unknown'
        return 'none'
