"""
股票咨询工作流模块，提供股票问答的专业流程
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel

from ...base.dialogue.memory import Message
from ...examples.agent.finance.tools import DataSource, MarketInfo, StockData
from ...base.rag.retriever import SearchResult
from ...base.search.engine import SearchResult as WebSearchResult


class QuestionType(str, Enum):
    """问题类型"""
    STOCK = "stock"  # 股票相关
    MARKET = "market"  # 市场相关
    GENERAL = "general"  # 普通问题
    INVALID = "invalid"  # 无效问题


class IntentType(str, Enum):
    """意图类型"""
    PRICE_QUERY = "price_query"  # 价格查询
    TREND_ANALYSIS = "trend_analysis"  # 趋势分析
    FINANCIAL_REPORT = "financial_report"  # 财报分析
    MARKET_SENTIMENT = "market_sentiment"  # 市场情绪
    COMPANY_INFO = "company_info"  # 公司信息
    NEWS_QUERY = "news_query"  # 新闻查询


@dataclass
class Entity:
    """金融实体"""
    name: str
    type: str
    value: str
    start: int
    end: int
    metadata: Dict[str, Any] = None


@dataclass
class StockContext:
    """股票咨询上下文"""
    question: str
    question_type: QuestionType
    intent: Optional[IntentType]
    entities: List[Entity]
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


class IntentRecognizer(ABC):
    """意图识别器抽象基类"""
    
    @abstractmethod
    async def recognize(
        self,
        question: str,
        context: StockContext
    ) -> IntentType:
        """识别意图"""
        pass
    
    @abstractmethod
    async def get_confidence(
        self,
        question: str,
        intent: IntentType
    ) -> float:
        """获取意图置信度"""
        pass


class EntityExtractor(ABC):
    """实体提取器抽象基类"""
    
    @abstractmethod
    async def extract(
        self,
        question: str,
        context: StockContext
    ) -> List[Entity]:
        """提取实体"""
        pass
    
    @abstractmethod
    async def normalize(
        self,
        entity: Entity
    ) -> Entity:
        """标准化实体"""
        pass


class QuestionExpander(ABC):
    """问题扩写器抽象基类"""
    
    @abstractmethod
    async def expand(
        self,
        question: str,
        context: StockContext
    ) -> str:
        """扩写问题"""
        pass
    
    @abstractmethod
    async def generate_sub_questions(
        self,
        question: str,
        context: StockContext
    ) -> List[str]:
        """生成子问题"""
        pass


class StockWorkflow:
    """股票咨询工作流"""
    
    def __init__(
        self,
        classifier: QuestionClassifier,
        intent_recognizer: IntentRecognizer,
        entity_extractor: EntityExtractor,
        expander: QuestionExpander
    ):
        self.classifier = classifier
        self.intent_recognizer = intent_recognizer
        self.entity_extractor = entity_extractor
        self.expander = expander
    
    async def process(
        self,
        question: str,
        context: Optional[StockContext] = None
    ) -> StockContext:
        """处理用户问题"""
        # 1. 验证和分类问题
        if not await self.classifier.validate(question):
            raise ValueError("问题不合法")
            
        question_type = await self.classifier.classify(question)
        
        if question_type == QuestionType.INVALID:
            raise ValueError("问题不合法")
            
        # 2. 识别意图
        intent = await self.intent_recognizer.recognize(
            question,
            context
        ) if question_type == QuestionType.STOCK else None
        
        # 3. 提取实体
        entities = await self.entity_extractor.extract(
            question,
            context
        )
        
        # 4. 扩写问题
        expanded_question = await self.expander.expand(
            question,
            context
        )
        
        # 5. 更新上下文
        new_context = StockContext(
            question=expanded_question,
            question_type=question_type,
            intent=intent,
            entities=entities,
            chat_history=context.chat_history if context else [],
            doc_results=[],
            web_results=[],
            metadata={"original_question": question}
        )
        
        return new_context


class StockConfig(BaseModel):
    """股票咨询配置"""
    max_history_turns: int = 10
    min_confidence: float = 0.7
    allowed_data_sources: Set[DataSource] = {
        DataSource.API,
        DataSource.DATABASE,
        DataSource.EXCEL,
        DataSource.ONLINE
    }
    extra_params: Dict[str, Any] = {} 