"""
课程生成模块，提供课程内容生成和管理功能
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Union

from pydantic import BaseModel


class CourseType(str, Enum):
    """课程类型"""
    LECTURE = "lecture"  # 讲座
    TUTORIAL = "tutorial"  # 教程
    QA = "qa"  # 问答


class CourseStatus(str, Enum):
    """课程状态"""
    PREPARING = "preparing"  # 准备中
    RUNNING = "running"  # 进行中
    PAUSED = "paused"  # 暂停
    FINISHED = "finished"  # 已完成


@dataclass
class CourseSegment:
    """课程片段"""
    content: str
    duration: float
    emotion: str = "neutral"
    action: str = "speaking"
    metadata: Dict[str, Any] = None


@dataclass
class Question:
    """问题"""
    content: str
    timestamp: float
    answer: Optional[str] = None
    metadata: Dict[str, Any] = None


class CourseConfig(BaseModel):
    """课程配置"""
    type: CourseType = CourseType.LECTURE
    max_duration: int = 3600  # 最大时长(秒)
    segment_max_length: int = 100  # 片段最大长度(字符)
    enable_interaction: bool = True
    language: str = "zh-CN"
    extra_params: Dict[str, Any] = {}


class CourseGenerator(ABC):
    """课程生成器抽象基类"""
    
    def __init__(self, config: CourseConfig):
        self.config = config
        self._status = CourseStatus.PREPARING
        self._segments: List[CourseSegment] = []
        self._questions: List[Question] = []
        self._current_segment_index = 0
    
    @property
    def status(self) -> CourseStatus:
        """获取当前状态"""
        return self._status
    
    @property
    def current_segment(self) -> Optional[CourseSegment]:
        """获取当前片段"""
        if 0 <= self._current_segment_index < len(self._segments):
            return self._segments[self._current_segment_index]
        return None
    
    @abstractmethod
    async def generate_course(
        self,
        topic: str,
        context: Dict[str, Any]
    ) -> List[CourseSegment]:
        """生成课程内容"""
        pass
    
    @abstractmethod
    async def handle_question(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> str:
        """处理问题"""
        pass
    
    @abstractmethod
    async def resume_course(self) -> None:
        """恢复课程"""
        pass
    
    async def pause_course(self) -> None:
        """暂停课程"""
        self._status = CourseStatus.PAUSED
    
    async def next_segment(self) -> Optional[CourseSegment]:
        """获取下一个片段"""
        if self._current_segment_index < len(self._segments) - 1:
            self._current_segment_index += 1
            return self.current_segment
        return None
    
    async def add_question(
        self,
        content: str,
        timestamp: float
    ) -> Question:
        """添加问题"""
        question = Question(
            content=content,
            timestamp=timestamp
        )
        self._questions.append(question)
        return question


class CourseRegistry:
    """课程组件注册表"""
    
    def __init__(self):
        self._generators: Dict[str, CourseGenerator] = {}
        self._active_courses: Dict[str, CourseGenerator] = {}
    
    def register_generator(
        self,
        name: str,
        generator: CourseGenerator
    ) -> None:
        """注册生成器"""
        self._generators[name] = generator
    
    def get_generator(
        self,
        name: str
    ) -> Optional[CourseGenerator]:
        """获取生成器"""
        return self._generators.get(name)
    
    def register_course(
        self,
        course_id: str,
        course: CourseGenerator
    ) -> None:
        """注册课程"""
        self._active_courses[course_id] = course
    
    def get_course(
        self,
        course_id: str
    ) -> Optional[CourseGenerator]:
        """获取课程"""
        return self._active_courses.get(course_id)
    
    def list_generators(self) -> List[str]:
        """列出所有生成器"""
        return list(self._generators.keys())
    
    def list_courses(self) -> List[str]:
        """列出所有课程"""
        return list(self._active_courses.keys()) 