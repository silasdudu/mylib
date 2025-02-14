"""
语音处理模块，提供语音识别和合成功能
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from pydantic import BaseModel


class AudioFormat(str, Enum):
    """音频格式"""
    WAV = "wav"
    MP3 = "mp3"
    PCM = "pcm"
    OGG = "ogg"


class VoiceType(str, Enum):
    """声音类型"""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


@dataclass
class AudioSegment:
    """音频片段"""
    data: bytes
    format: AudioFormat
    sample_rate: int
    duration: float
    metadata: Dict[str, Any] = None


class SpeechConfig(BaseModel):
    """语音配置"""
    language: str = "zh-CN"
    sample_rate: int = 16000
    audio_format: AudioFormat = AudioFormat.WAV
    voice_type: VoiceType = VoiceType.NEUTRAL
    extra_params: Dict[str, Any] = {}


class SpeechRecognizer(ABC):
    """语音识别器抽象基类"""
    
    def __init__(self, config: SpeechConfig):
        self.config = config
    
    @abstractmethod
    async def recognize(
        self,
        audio: AudioSegment
    ) -> str:
        """识别音频内容"""
        pass
    
    @abstractmethod
    async def recognize_stream(
        self,
        audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[str]:
        """流式识别音频"""
        pass


class SpeechSynthesizer(ABC):
    """语音合成器抽象基类"""
    
    def __init__(self, config: SpeechConfig):
        self.config = config
    
    @abstractmethod
    async def synthesize(
        self,
        text: str
    ) -> AudioSegment:
        """合成音频"""
        pass
    
    @abstractmethod
    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        """流式合成音频"""
        pass


class SpeechRegistry:
    """语音组件注册表"""
    
    def __init__(self):
        self._recognizers: Dict[str, SpeechRecognizer] = {}
        self._synthesizers: Dict[str, SpeechSynthesizer] = {}
        
    def register_recognizer(
        self,
        name: str,
        recognizer: SpeechRecognizer
    ) -> None:
        """注册识别器"""
        self._recognizers[name] = recognizer
        
    def register_synthesizer(
        self,
        name: str,
        synthesizer: SpeechSynthesizer
    ) -> None:
        """注册合成器"""
        self._synthesizers[name] = synthesizer
        
    def get_recognizer(
        self,
        name: str
    ) -> Optional[SpeechRecognizer]:
        """获取识别器"""
        return self._recognizers.get(name)
        
    def get_synthesizer(
        self,
        name: str
    ) -> Optional[SpeechSynthesizer]:
        """获取合成器"""
        return self._synthesizers.get(name) 