"""
股票咨询Agent使用示例
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List

from base.agent.stock import StockAgent
from base.agent.stock_strategies import (IntentBasedPlanner, PriorityToolSelector,
                                       TemplateResponseGenerator)
from base.dialogue.memory import Message, MessageRole, ShortTermMemory
from base.finance.tools import (DataSource, MarketInfo, StockAPI, StockData,
                              DatabaseQuery, ExcelProcessor)
from base.search.engine import SearchEngine, SearchResult
from examples.workflow.stock import (Entity, IntentType, QuestionType, StockContext,
                               StockWorkflow)


class DemoStockAPI(StockAPI):
    """示例股票API实现"""
    
    async def get_real_time_quote(self, code: str) -> StockData:
        """获取模拟实时报价"""
        return StockData(
            code=code,
            name="示例股票",
            price=100.0,
            timestamp=datetime.now(),
            change=0.5,
            volume=10000
        )
    
    async def get_historical_data(
        self,
        code: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[StockData]:
        """获取模拟历史数据"""
        return [
            StockData(
                code=code,
                name="示例股票",
                price=100.0 + i,
                timestamp=start_date + timedelta(days=i),
                change=0.1 * i,
                volume=10000 + i * 100
            )
            for i in range((end_date - start_date).days + 1)
        ]
    
    async def get_market_status(self) -> MarketInfo:
        """获取模拟市场状态"""
        return MarketInfo(
            index="上证指数",
            value=3000.0,
            change=0.3,
            timestamp=datetime.now()
        )


class DemoDatabaseQuery(DatabaseQuery):
    """示例数据库查询实现"""
    
    async def query_stock_info(self, code: str) -> Dict[str, Any]:
        """查询模拟股票信息"""
        return {
            "code": code,
            "name": "示例股票",
            "industry": "科技",
            "market_cap": 1000000000,
            "pe_ratio": 20.5
        }
    
    async def query_financial_report(
        self,
        code: str,
        report_type: str
    ) -> Dict[str, Any]:
        """查询模拟财务报表"""
        return {
            "revenue": 1000000,
            "profit": 100000,
            "assets": 5000000,
            "liabilities": 2000000
        }
    
    async def save_query_history(
        self,
        query: str,
        result: Dict[str, Any]
    ) -> None:
        """保存查询历史"""
        pass


class DemoExcelProcessor(ExcelProcessor):
    """示例Excel处理器实现"""
    
    async def read_stock_data(
        self,
        file_path: str,
        sheet_name: str
    ) -> List[Dict[str, Any]]:
        """读取模拟股票数据"""
        return [{
            "date": "2024-01-01",
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.0,
            "volume": 10000
        }]
    
    async def export_analysis_result(
        self,
        data: List[Dict[str, Any]],
        file_path: str
    ) -> None:
        """导出分析结果"""
        pass


class DemoSearchEngine(SearchEngine):
    """示例搜索引擎实现"""
    
    async def search(
        self,
        query: str,
        **kwargs
    ) -> List[SearchResult]:
        """执行模拟搜索"""
        return [
            SearchResult(
                title=f"示例新闻{i}",
                url=f"http://example.com/news/{i}",
                snippet=f"这是关于{query}的示例新闻{i}...",
                source="示例网站",
                timestamp=datetime.now().isoformat()
            )
            for i in range(3)
        ]
    
    async def get_content(self, url: str) -> str:
        """获取模拟网页内容"""
        return "这是示例网页内容..."


async def main():
    """主函数"""
    # 创建工具实例
    stock_api = DemoStockAPI()
    database = DemoDatabaseQuery()
    excel_processor = DemoExcelProcessor()
    search_engine = DemoSearchEngine()
    
    # 创建Agent组件
    planner = IntentBasedPlanner()
    tool_selector = PriorityToolSelector()
    response_generator = TemplateResponseGenerator()
    
    # 创建Agent
    agent = StockAgent(
        planning_strategy=planner,
        tool_selector=tool_selector,
        response_generator=response_generator,
        stock_api=stock_api,
        database=database,
        excel_processor=excel_processor,
        search_engine=search_engine
    )
    
    # 创建示例上下文
    context = StockContext(
        question="请查询贵州茅台的最新股价",
        question_type=QuestionType.STOCK,
        intent=IntentType.PRICE_QUERY,
        entities=[
            Entity(
                name="贵州茅台",
                type="stock_code",
                value="600519",
                start=2,
                end=6
            )
        ],
        chat_history=[
            Message(
                role=MessageRole.USER,
                content="请查询贵州茅台的最新股价",
                timestamp=datetime.now()
            )
        ],
        doc_results=[],
        web_results=[]
    )
    
    # 处理查询
    response = await agent.process(context)
    print(f"问题：{context.question}")
    print(f"回答：{response}")
    
    # 示例2：查询公司信息
    context2 = StockContext(
        question="请介绍一下贵州茅台的基本情况",
        question_type=QuestionType.STOCK,
        intent=IntentType.COMPANY_INFO,
        entities=[
            Entity(
                name="贵州茅台",
                type="stock_code",
                value="600519",
                start=5,
                end=9
            )
        ],
        chat_history=[],
        doc_results=[],
        web_results=[]
    )
    
    response2 = await agent.process(context2)
    print(f"\n问题：{context2.question}")
    print(f"回答：{response2}")


if __name__ == "__main__":
    asyncio.run(main()) 