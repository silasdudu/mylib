"""
股票咨询Agent的策略类实现
"""
from typing import Any, Dict, List, Optional

from finance.tools import DataSource
from ...examples.workflow.stock import IntentType, QuestionType, StockContext
from .stock import Action, ActionResult, ActionType, PlanningStrategy, ResponseGenerator, ToolSelector


class IntentBasedPlanner(PlanningStrategy):
    """基于意图的规划策略"""
    
    def __init__(self, default_priority: int = 0):
        self.default_priority = default_priority
        self.intent_action_map = {
            IntentType.PRICE_QUERY: [
                Action(
                    type=ActionType.QUERY_PRICE,
                    parameters={},
                    description="查询实时价格",
                    data_source=DataSource.API,
                    priority=3
                )
            ],
            IntentType.TREND_ANALYSIS: [
                Action(
                    type=ActionType.ANALYZE_TREND,
                    parameters={},
                    description="分析价格趋势",
                    data_source=DataSource.API,
                    priority=2
                ),
                Action(
                    type=ActionType.SEARCH_NEWS,
                    parameters={},
                    description="搜索相关新闻",
                    data_source=DataSource.ONLINE,
                    priority=1
                )
            ],
            IntentType.FINANCIAL_REPORT: [
                Action(
                    type=ActionType.READ_FINANCIAL,
                    parameters={},
                    description="读取财务报表",
                    data_source=DataSource.DATABASE,
                    priority=3
                ),
                Action(
                    type=ActionType.QUERY_DATABASE,
                    parameters={},
                    description="查询补充信息",
                    data_source=DataSource.DATABASE,
                    priority=2
                )
            ],
            IntentType.MARKET_SENTIMENT: [
                Action(
                    type=ActionType.SEARCH_NEWS,
                    parameters={},
                    description="搜索市场新闻",
                    data_source=DataSource.ONLINE,
                    priority=3
                ),
                Action(
                    type=ActionType.ANALYZE_TREND,
                    parameters={},
                    description="分析大盘走势",
                    data_source=DataSource.API,
                    priority=2
                )
            ],
            IntentType.COMPANY_INFO: [
                Action(
                    type=ActionType.QUERY_DATABASE,
                    parameters={},
                    description="查询公司信息",
                    data_source=DataSource.DATABASE,
                    priority=3
                ),
                Action(
                    type=ActionType.SEARCH_NEWS,
                    parameters={},
                    description="搜索公司新闻",
                    data_source=DataSource.ONLINE,
                    priority=2
                )
            ],
            IntentType.NEWS_QUERY: [
                Action(
                    type=ActionType.SEARCH_NEWS,
                    parameters={},
                    description="搜索相关新闻",
                    data_source=DataSource.ONLINE,
                    priority=3
                )
            ]
        }
    
    async def plan_actions(
        self,
        context: StockContext
    ) -> List[Action]:
        """根据意图规划动作序列"""
        if context.question_type != QuestionType.STOCK:
            return [
                Action(
                    type=ActionType.SEARCH_NEWS,
                    parameters={"query": context.question},
                    description="搜索一般信息",
                    data_source=DataSource.ONLINE,
                    priority=self.default_priority
                )
            ]
            
        if not context.intent:
            return []
            
        actions = self.intent_action_map.get(context.intent, [])
        
        # 填充动作参数
        for action in actions:
            if action.type == ActionType.QUERY_PRICE:
                for entity in context.entities:
                    if entity.type == "stock_code":
                        action.parameters["code"] = entity.value
                        
            elif action.type == ActionType.ANALYZE_TREND:
                for entity in context.entities:
                    if entity.type == "stock_code":
                        action.parameters["code"] = entity.value
                    elif entity.type == "date_range":
                        action.parameters["start_date"] = entity.metadata.get("start_date")
                        action.parameters["end_date"] = entity.metadata.get("end_date")
                        
            elif action.type == ActionType.READ_FINANCIAL:
                for entity in context.entities:
                    if entity.type == "stock_code":
                        action.parameters["code"] = entity.value
                    elif entity.type == "report_type":
                        action.parameters["report_type"] = entity.value
                        
            elif action.type == ActionType.SEARCH_NEWS:
                action.parameters["query"] = context.question
                
            elif action.type == ActionType.QUERY_DATABASE:
                for entity in context.entities:
                    if entity.type == "stock_code":
                        action.parameters["code"] = entity.value
                        
        return sorted(actions, key=lambda x: x.priority, reverse=True)
    
    async def adjust_plan(
        self,
        original_plan: List[Action],
        failed_action: Action,
        error: str
    ) -> List[Action]:
        """调整失败的规划"""
        # 移除失败的动作
        adjusted_plan = [
            action for action in original_plan
            if action.type != failed_action.type
        ]
        
        # 如果API调用失败，尝试使用数据库
        if failed_action.data_source == DataSource.API:
            adjusted_plan.append(Action(
                type=ActionType.QUERY_DATABASE,
                parameters=failed_action.parameters,
                description="从数据库查询备选数据",
                data_source=DataSource.DATABASE,
                priority=failed_action.priority - 1
            ))
            
        # 如果数据库查询失败，尝试使用Excel
        elif failed_action.data_source == DataSource.DATABASE:
            adjusted_plan.append(Action(
                type=ActionType.READ_EXCEL,
                parameters={
                    "file_path": "backup_data.xlsx",
                    "sheet_name": "stock_data"
                },
                description="从Excel读取备选数据",
                data_source=DataSource.EXCEL,
                priority=failed_action.priority - 1
            ))
            
        return sorted(adjusted_plan, key=lambda x: x.priority, reverse=True)


class PriorityToolSelector(ToolSelector):
    """基于优先级的工具选择器"""
    
    async def select_tool(
        self,
        action: Action,
        available_tools: Dict[DataSource, Any]
    ) -> Any:
        """选择最合适的工具"""
        if action.data_source not in available_tools:
            raise ValueError(f"数据源 {action.data_source} 不可用")
            
        return available_tools[action.data_source]
    
    async def validate_tool(
        self,
        tool: Any,
        action: Action
    ) -> bool:
        """验证工具是否适用"""
        if action.data_source == DataSource.API:
            return hasattr(tool, "get_real_time_quote")
        elif action.data_source == DataSource.DATABASE:
            return hasattr(tool, "query_stock_info")
        elif action.data_source == DataSource.EXCEL:
            return hasattr(tool, "read_stock_data")
        elif action.data_source == DataSource.ONLINE:
            return hasattr(tool, "search")
        return False


class TemplateResponseGenerator(ResponseGenerator):
    """基于模板的回应生成器"""
    
    def __init__(self):
        self.templates = {
            ActionType.QUERY_PRICE: "当前{stock_name}({stock_code})的价格是{price}元，"
                                  "较前一交易日{change_direction}{change_value}%。",
            ActionType.ANALYZE_TREND: "根据分析，{stock_name}在{date_range}期间的"
                                    "价格趋势{trend_description}。{trend_analysis}",
            ActionType.READ_FINANCIAL: "{stock_name}的{report_type}显示：{financial_summary}",
            ActionType.SEARCH_NEWS: "相关新闻：\n{news_summary}",
            ActionType.QUERY_DATABASE: "公司基本信息：{company_info}",
            ActionType.READ_EXCEL: "历史数据显示：{historical_data}",
            ActionType.COMBINE_INFO: "{combined_analysis}"
        }
    
    async def generate(
        self,
        context: StockContext,
        action_results: List[ActionResult]
    ) -> str:
        """生成最终回应"""
        if not action_results:
            return "抱歉，我无法处理这个问题。"
            
        response_parts = []
        
        for result in action_results:
            if result.success:
                formatted = await self.format_result(result)
                if formatted:
                    response_parts.append(formatted)
                    
        if not response_parts:
            return "抱歉，我无法获取相关信息。"
            
        # 组合所有部分
        response = " ".join(response_parts)
        
        # 添加免责声明
        response += "\n\n免责声明：以上信息仅供参考，不构成投资建议。"
        
        return response
    
    async def format_result(
        self,
        result: ActionResult
    ) -> str:
        """格式化单个结果"""
        template = self.templates.get(result.action.type)
        if not template or not result.data:
            return ""
            
        try:
            if result.action.type == ActionType.QUERY_PRICE:
                stock_data = result.data
                return template.format(
                    stock_name=stock_data.name,
                    stock_code=stock_data.code,
                    price=stock_data.price,
                    change_direction="上涨" if stock_data.change > 0 else "下跌",
                    change_value=abs(stock_data.change)
                )
                
            elif result.action.type == ActionType.SEARCH_NEWS:
                news_list = result.data
                news_summary = "\n".join([
                    f"- {news.title}"
                    for news in news_list[:3]
                ])
                return template.format(news_summary=news_summary)
                
            # 其他类型的结果格式化...
            return str(result.data)
            
        except Exception as e:
            return f"结果格式化失败：{str(e)}" 