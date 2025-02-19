"""
嵌入模型实现模块
"""

from .gte_qwen import GTEQwenEmbedding, GTEQwenEmbeddingConfig
from .custom import CustomEmbedding, CustomEmbeddingConfig

__all__ = [
    'GTEQwenEmbedding',
    'GTEQwenEmbeddingConfig',
    'CustomEmbedding',
    'CustomEmbeddingConfig'
] 