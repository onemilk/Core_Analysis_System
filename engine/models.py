"""Core data models — pure Python dataclasses, no dependencies."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Category:
    """分类节点（盆地/区块/构造/井号 — 自引用树）"""
    id: Optional[int] = None
    name: str = ""
    parent_id: Optional[int] = None
    type: str = ""  # 'basin'|'block'|'structure'|'well'


@dataclass
class ImageRecord:
    """岩心图像元数据"""
    id: Optional[int] = None
    category_id: Optional[int] = None
    filename: str = ""
    filepath: str = ""
    capture_date: str = ""
    depth_from: Optional[float] = None
    depth_to: Optional[float] = None
    scale_value: float = 1.0  # mm/pixel
    scale_unit: str = "mm"    # 'mm'|'μm'
    dpi: int = 96
    lithology: str = ""
    description: str = ""
    created_at: str = ""


@dataclass
class HoleResult:
    """单孔洞分析结果"""
    id: Optional[int] = None
    image_id: Optional[int] = None
    session_id: Optional[int] = None
    region_index: int = 0
    area_mm2: float = 0.0
    equivalent_d_mm: float = 0.0
    fill_status: str = "未充填"
    fill_material: str = ""
    effectiveness: str = "有效"
    hole_type: str = "溶洞"
    size_category: str = ""
    is_valid: bool = True
    notes: str = ""
    created_at: str = ""


@dataclass
class FractureResult:
    """单裂缝分析结果"""
    id: Optional[int] = None
    image_id: Optional[int] = None
    session_id: Optional[int] = None
    region_index: int = 0
    length_mm: float = 0.0
    width_mm: float = 0.0
    area_mm2: float = 0.0
    porosity: float = 0.0
    fracture_type: str = "构造缝"
    fill_status: str = "张开缝(未充填)"
    fill_material: str = ""
    effectiveness: str = "有效"
    is_valid: bool = True
    notes: str = ""
    created_at: str = ""


@dataclass
class AnalysisSession:
    """分析会话 — 记录每次完整分析的参数和报告"""
    id: Optional[int] = None
    image_id: Optional[int] = None
    analysis_type: str = ""  # 'hole'|'fracture'
    params_json: str = ""
    report_html: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass
class MaskRegion:
    """图像提取区域 — 在 engine 和 UI 之间传递"""
    contour: list = field(default_factory=list)  # list of [x,y] points
    area_px: float = 0.0
    centroid: tuple = (0.0, 0.0)
    bbox: tuple = (0, 0, 0, 0)  # x, y, w, h


@dataclass
class GrainResult:
    """单颗粒分析结果"""
    id: Optional[int] = None
    image_id: Optional[int] = None
    session_id: Optional[int] = None
    region_index: int = 0
    area_mm2: float = 0.0
    equivalent_d_mm: float = 0.0
    perimeter_mm: float = 0.0
    feret_long_mm: float = 0.0
    feret_short_mm: float = 0.0
    circularity: float = 0.0
    size_category: str = ""
    is_valid: bool = True
    notes: str = ""
    created_at: str = ""
