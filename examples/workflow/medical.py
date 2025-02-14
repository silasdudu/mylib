"""
医疗咨询工作流模块，提供医疗问答的专业流程
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from ...base.dialogue.memory import Message
from ...base.rag.retriever import SearchResult
from ...base.search.engine import SearchResult as WebSearchResult


class QuestionType(str, Enum):
    """问题类型"""
    MEDICAL = "medical"  # 医疗类问题
    GENERAL = "general"  # 普通问题
    INVALID = "invalid"  # 无效问题


class ResponseType(str, Enum):
    """回应类型"""
    ENCOURAGE = "encourage"  # 鼓励用户继续分享
    INQUIRY = "inquiry"  # 询问更多细节
    CONFIRM = "confirm"  # 确认用户问题
    ANSWER = "answer"  # 提供专业回答


@dataclass
class MedicalContext:
    """医疗咨询上下文"""
    question: str
    question_type: QuestionType
    chat_history: List[Message]
    doc_results: List[SearchResult]
    web_results: List[WebSearchResult]
    metadata: Dict[str, Any] = None


class QuestionClassifier(ABC):
    """问题分类器抽象基类"""
    
    @abstractmethod
    async def classify(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> QuestionType:
        """分类问题类型"""
        pass
    
    @abstractmethod
    async def validate(
        self,
        question: str
    ) -> bool:
        """验证问题是否合法"""
        pass


class QuestionExpander(ABC):
    """问题扩写器抽象基类"""
    
    @abstractmethod
    async def expand(
        self,
        question: str,
        context: MedicalContext
    ) -> str:
        """扩写问题"""
        pass
    
    @abstractmethod
    async def generate_sub_questions(
        self,
        question: str,
        context: MedicalContext
    ) -> List[str]:
        """生成子问题"""
        pass


class ResponseSelector(ABC):
    """回应选择器抽象基类"""
    
    @abstractmethod
    async def select_type(
        self,
        context: MedicalContext
    ) -> ResponseType:
        """选择回应类型"""
        pass
    
    @abstractmethod
    async def generate_response(
        self,
        response_type: ResponseType,
        context: MedicalContext
    ) -> str:
        """生成回应内容"""
        pass


class MedicalWorkflow:
    """医疗咨询工作流"""
    
    def __init__(
        self,
        classifier: QuestionClassifier,
        expander: QuestionExpander,
        selector: ResponseSelector
    ):
        self.classifier = classifier
        self.expander = expander
        self.selector = selector
    
    async def process(
        self,
        question: str,
        context: MedicalContext
    ) -> str:
        """处理用户问题"""
        # 1. 验证和分类问题
        if not await self.classifier.validate(question):
            return "抱歉，您的问题似乎不太清晰，能否重新描述一下？"
            
        question_type = await self.classifier.classify(question)
        context.question_type = question_type
        
        if question_type == QuestionType.INVALID:
            return "抱歉，您的问题可能不太恰当，建议重新组织语言。"
        elif question_type == QuestionType.GENERAL:
            return "这似乎不是医疗相关的问题，我会用一般方式回答..."
            
        # 2. 扩写问题
        expanded_question = await self.expander.expand(question, context)
        sub_questions = await self.expander.generate_sub_questions(
            expanded_question,
            context
        )
        
        # 3. 选择回应类型
        response_type = await self.selector.select_type(context)
        
        # 4. 生成回应
        response = await self.selector.generate_response(
            response_type,
            context
        )
        
        return response


class MedicalConfig(BaseModel):
    """医疗咨询配置"""
    max_history_turns: int = 10
    min_confidence: float = 0.7
    require_confirmation: bool = True
    extra_params: Dict[str, Any] = {} 