"""
数字人驱动模块，提供数字人动画生成和控制功能
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from pydantic import BaseModel

from ...base.service.speech import AudioSegment


class AvatarAction(str, Enum):
    """数字人动作"""
    IDLE = "idle"
    SPEAKING = "speaking"
    THINKING = "thinking"
    NODDING = "nodding"
    GESTURING = "gesturing"


class AvatarEmotion(str, Enum):
    """数字人情绪"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISED = "surprised"
    CONFUSED = "confused"


@dataclass
class VideoFrame:
    """视频帧"""
    data: bytes
    timestamp: float
    width: int
    height: int
    format: str = "rgb24"


@dataclass
class AvatarState:
    """数字人状态"""
    action: AvatarAction
    emotion: AvatarEmotion
    position: Tuple[float, float, float]  # x, y, z
    rotation: Tuple[float, float, float]  # pitch, yaw, roll
    blend_shapes: Dict[str, float]  # 面部表情混合权重


class AvatarConfig(BaseModel):
    """数字人配置"""
    model_path: str
    texture_path: str
    width: int = 1280
    height: int = 720
    fps: int = 30
    enable_gpu: bool = True
    extra_params: Dict[str, Any] = {}


class AvatarDriver(ABC):
    """数字人驱动器抽象基类"""
    
    def __init__(self, config: AvatarConfig):
        self.config = config
        self._current_state = AvatarState(
            action=AvatarAction.IDLE,
            emotion=AvatarEmotion.NEUTRAL,
            position=(0, 0, 0),
            rotation=(0, 0, 0),
            blend_shapes={}
        )
    
    @property
    def current_state(self) -> AvatarState:
        """获取当前状态"""
        return self._current_state
    
    @abstractmethod
    async def initialize(self) -> None:
        """初始化数字人"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """关闭数字人"""
        pass
    
    @abstractmethod
    async def drive_with_audio(
        self,
        audio: AudioSegment
    ) -> AsyncIterator[VideoFrame]:
        """使用音频驱动数字人"""
        pass
    
    @abstractmethod
    async def drive_with_text(
        self,
        text: str
    ) -> AsyncIterator[VideoFrame]:
        """使用文本驱动数字人"""
        pass
    
    @abstractmethod
    async def set_action(
        self,
        action: AvatarAction
    ) -> None:
        """设置动作"""
        pass
    
    @abstractmethod
    async def set_emotion(
        self,
        emotion: AvatarEmotion
    ) -> None:
        """设置情绪"""
        pass
    
    @abstractmethod
    async def set_blend_shapes(
        self,
        blend_shapes: Dict[str, float]
    ) -> None:
        """设置面部表情"""
        pass


class AvatarRegistry:
    """数字人组件注册表"""
    
    def __init__(self):
        self._drivers: Dict[str, AvatarDriver] = {}
    
    def register(
        self,
        name: str,
        driver: AvatarDriver
    ) -> None:
        """注册驱动器"""
        self._drivers[name] = driver
    
    def get(
        self,
        name: str
    ) -> Optional[AvatarDriver]:
        """获取驱动器"""
        return self._drivers.get(name)
    
    def list_drivers(self) -> List[str]:
        """列出所有驱动器"""
        return list(self._drivers.keys()) 