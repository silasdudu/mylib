"""
金融工具模块，提供股票数据查询和分析功能
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class DataSource(str, Enum):
    """数据源类型"""
    API = "api"
    DATABASE = "database"
    EXCEL = "excel"
    ONLINE = "online"


@dataclass
class StockData:
    """股票数据"""
    code: str
    name: str
    price: float
    timestamp: datetime
    change: float
    volume: int
    metadata: Dict[str, Any] = None


@dataclass
class MarketInfo:
    """市场信息"""
    index: str
    value: float
    change: float
    timestamp: datetime
    metadata: Dict[str, Any] = None


class StockAPI(ABC):
    """股票API抽象基类"""
    
    @abstractmethod
    async def get_real_time_quote(
        self,
        code: str
    ) -> StockData:
        """获取实时报价"""
        pass
    
    @abstractmethod
    async def get_historical_data(
        self,
        code: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[StockData]:
        """获取历史数据"""
        pass
    
    @abstractmethod
    async def get_market_status(
        self
    ) -> MarketInfo:
        """获取市场状态"""
        pass


class DatabaseQuery(ABC):
    """数据库查询抽象基类"""
    
    @abstractmethod
    async def query_stock_info(
        self,
        code: str
    ) -> Dict[str, Any]:
        """查询股票信息"""
        pass
    
    @abstractmethod
    async def query_financial_report(
        self,
        code: str,
        report_type: str
    ) -> Dict[str, Any]:
        """查询财务报表"""
        pass
    
    @abstractmethod
    async def save_query_history(
        self,
        query: str,
        result: Dict[str, Any]
    ) -> None:
        """保存查询历史"""
        pass


class ExcelProcessor(ABC):
    """Excel处理器抽象基类"""
    
    @abstractmethod
    async def read_stock_data(
        self,
        file_path: str,
        sheet_name: str
    ) -> List[Dict[str, Any]]:
        """读取股票数据"""
        pass
    
    @abstractmethod
    async def export_analysis_result(
        self,
        data: List[Dict[str, Any]],
        file_path: str
    ) -> None:
        """导出分析结果"""
        pass


class FinanceConfig(BaseModel):
    """金融工具配置"""
    api_key: str = ""
    database_url: str = ""
    excel_dir: str = ""
    cache_ttl: int = 300  # 缓存时间（秒）
    extra_params: Dict[str, Any] = {} 