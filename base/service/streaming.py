"""
流媒体处理模块，提供音视频流处理和混流功能
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from pydantic import BaseModel

from ...examples.service.avatar import VideoFrame
from .speech import AudioFormat, AudioSegment


class StreamFormat(str, Enum):
    """流媒体格式"""
    RTMP = "rtmp"
    HLS = "hls"
    DASH = "dash"
    WEBRTC = "webrtc"


class StreamQuality(str, Enum):
    """流媒体质量"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class MediaStream:
    """媒体流"""
    stream_id: str
    format: StreamFormat
    quality: StreamQuality
    metadata: Dict[str, Any]


class StreamConfig(BaseModel):
    """流媒体配置"""
    format: StreamFormat = StreamFormat.WEBRTC
    quality: StreamQuality = StreamQuality.HIGH
    video_codec: str = "h264"
    audio_codec: str = "aac"
    video_bitrate: int = 2000000  # 2Mbps
    audio_bitrate: int = 128000   # 128kbps
    keyframe_interval: int = 2     # 2秒
    extra_params: Dict[str, Any] = {}


class StreamProcessor(ABC):
    """流媒体处理器抽象基类"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
    
    @abstractmethod
    async def initialize(self) -> None:
        """初始化处理器"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """关闭处理器"""
        pass
    
    @abstractmethod
    async def create_stream(self) -> MediaStream:
        """创建媒体流"""
        pass
    
    @abstractmethod
    async def process_video(
        self,
        frames: AsyncIterator[VideoFrame]
    ) -> AsyncIterator[bytes]:
        """处理视频流"""
        pass
    
    @abstractmethod
    async def process_audio(
        self,
        audio: AsyncIterator[AudioSegment]
    ) -> AsyncIterator[bytes]:
        """处理音频流"""
        pass
    
    @abstractmethod
    async def mux_streams(
        self,
        video_stream: AsyncIterator[bytes],
        audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[bytes]:
        """混流"""
        pass


class StreamRegistry:
    """流媒体组件注册表"""
    
    def __init__(self):
        self._processors: Dict[str, StreamProcessor] = {}
        self._active_streams: Dict[str, MediaStream] = {}
    
    def register_processor(
        self,
        name: str,
        processor: StreamProcessor
    ) -> None:
        """注册处理器"""
        self._processors[name] = processor
    
    def get_processor(
        self,
        name: str
    ) -> Optional[StreamProcessor]:
        """获取处理器"""
        return self._processors.get(name)
    
    def register_stream(
        self,
        stream: MediaStream
    ) -> None:
        """注册媒体流"""
        self._active_streams[stream.stream_id] = stream
    
    def get_stream(
        self,
        stream_id: str
    ) -> Optional[MediaStream]:
        """获取媒体流"""
        return self._active_streams.get(stream_id)
    
    def list_processors(self) -> List[str]:
        """列出所有处理器"""
        return list(self._processors.keys())
    
    def list_streams(self) -> List[str]:
        """列出所有媒体流"""
        return list(self._active_streams.keys()) 