"""
记忆配置模块
"""
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class MemoryConfig(BaseModel):
    """记忆配置"""
    use_short_term: bool = True
    use_long_term: bool = False
    short_term_size: int = 10
    long_term_threshold: float = 0.7  # 长期记忆的相关性阈值
    save_path: Optional[str] = None  # 记忆保存路径
    extra_params: Dict[str, Any] = Field(default_factory=dict) 