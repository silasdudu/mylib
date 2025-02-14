"""
股票咨询Agent模块，提供智能的股票问答服务
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel

from ..finance.tools import (DataSource, DatabaseQuery, ExcelProcessor,
                           MarketInfo, StockAPI, StockData)
from ..search.engine import SearchEngine, SearchResult
from ...examples.workflow.stock import IntentType, StockContext


class ActionType(str, Enum):
    """动作类型"""
    QUERY_PRICE = "query_price"  # 查询价格
    ANALYZE_TREND = "analyze_trend"  # 分析趋势
    READ_FINANCIAL = "read_financial"  # 读取财报
    SEARCH_NEWS = "search_news"  # 搜索新闻
    QUERY_DATABASE = "query_database"  # 查询数据库
    READ_EXCEL = "read_excel"  # 读取Excel
    COMBINE_INFO = "combine_info"  # 整合信息


@dataclass
class Action:
    """Agent动作"""
    type: ActionType
    parameters: Dict[str, Any]
    description: str
    data_source: DataSource
    priority: int = 0
    metadata: Dict[str, Any] = None


@dataclass
class ActionResult:
    """动作执行结果"""
    action: Action
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class PlanningStrategy(ABC):
    """规划策略抽象基类"""
    
    @abstractmethod
    async def plan_actions(
        self,
        context: StockContext
    ) -> List[Action]:
        """规划动作序列"""
        pass
    
    @abstractmethod
    async def adjust_plan(
        self,
        original_plan: List[Action],
        failed_action: Action,
        error: str
    ) -> List[Action]:
        """调整规划"""
        pass


class ToolSelector(ABC):
    """工具选择器抽象基类"""
    
    @abstractmethod
    async def select_tool(
        self,
        action: Action,
        available_tools: Dict[DataSource, Any]
    ) -> Any:
        """选择合适的工具"""
        pass
    
    @abstractmethod
    async def validate_tool(
        self,
        tool: Any,
        action: Action
    ) -> bool:
        """验证工具是否适用"""
        pass


class ResponseGenerator(ABC):
    """回应生成器抽象基类"""
    
    @abstractmethod
    async def generate(
        self,
        context: StockContext,
        action_results: List[ActionResult]
    ) -> str:
        """生成回应"""
        pass
    
    @abstractmethod
    async def format_result(
        self,
        result: ActionResult
    ) -> str:
        """格式化结果"""
        pass


class StockAgent:
    """股票咨询Agent"""
    
    def __init__(
        self,
        planning_strategy: PlanningStrategy,
        tool_selector: ToolSelector,
        response_generator: ResponseGenerator,
        stock_api: StockAPI,
        database: DatabaseQuery,
        excel_processor: ExcelProcessor,
        search_engine: SearchEngine
    ):
        self.planning_strategy = planning_strategy
        self.tool_selector = tool_selector
        self.response_generator = response_generator
        self.tools = {
            DataSource.API: stock_api,
            DataSource.DATABASE: database,
            DataSource.EXCEL: excel_processor,
            DataSource.ONLINE: search_engine
        }
    
    async def process(
        self,
        context: StockContext
    ) -> str:
        """处理用户查询"""
        # 1. 规划动作序列
        actions = await self.planning_strategy.plan_actions(context)
        results = []
        
        # 2. 执行动作
        for action in actions:
            try:
                # 选择工具
                tool = await self.tool_selector.select_tool(
                    action,
                    self.tools
                )
                
                # 验证工具
                if not await self.tool_selector.validate_tool(tool, action):
                    raise ValueError(f"工具 {tool.__class__.__name__} 不适用于动作 {action.type}")
                
                # 执行动作
                result = await self._execute_action(action, tool)
                results.append(result)
                
                # 如果执行失败，尝试调整计划
                if not result.success:
                    actions = await self.planning_strategy.adjust_plan(
                        actions,
                        action,
                        result.error
                    )
                    
            except Exception as e:
                results.append(ActionResult(
                    action=action,
                    success=False,
                    data=None,
                    error=str(e)
                ))
                
                # 调整计划
                actions = await self.planning_strategy.adjust_plan(
                    actions,
                    action,
                    str(e)
                )
        
        # 3. 生成回应
        response = await self.response_generator.generate(context, results)
        return response
    
    async def _execute_action(
        self,
        action: Action,
        tool: Any
    ) -> ActionResult:
        """执行具体动作"""
        try:
            if action.type == ActionType.QUERY_PRICE:
                data = await tool.get_real_time_quote(
                    action.parameters["code"]
                )
            elif action.type == ActionType.ANALYZE_TREND:
                data = await tool.get_historical_data(
                    action.parameters["code"],
                    action.parameters["start_date"],
                    action.parameters["end_date"]
                )
            elif action.type == ActionType.READ_FINANCIAL:
                data = await tool.query_financial_report(
                    action.parameters["code"],
                    action.parameters["report_type"]
                )
            elif action.type == ActionType.SEARCH_NEWS:
                data = await tool.search(action.parameters["query"])
            elif action.type == ActionType.QUERY_DATABASE:
                data = await tool.query_stock_info(
                    action.parameters["code"]
                )
            elif action.type == ActionType.READ_EXCEL:
                data = await tool.read_stock_data(
                    action.parameters["file_path"],
                    action.parameters["sheet_name"]
                )
            else:
                raise ValueError(f"不支持的动作类型：{action.type}")
                
            return ActionResult(
                action=action,
                success=True,
                data=data
            )
            
        except Exception as e:
            return ActionResult(
                action=action,
                success=False,
                data=None,
                error=str(e)
            )


class AgentConfig(BaseModel):
    """Agent配置"""
    max_retries: int = 3
    timeout: float = 30.0
    parallel_actions: bool = True
    error_tolerance: float = 0.3
    extra_params: Dict[str, Any] = {} 