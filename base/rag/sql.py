"""
SQL处理模块，提供SQL生成、执行和结果处理的功能
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class SQLConfig(BaseModel):
    """SQL配置"""
    max_results: int = 1000
    timeout: float = 30.0
    safe_mode: bool = True  # 是否启用安全模式（限制危险操作）
    extra_params: Dict[str, Any] = {}


@dataclass
class SQLResult:
    """SQL执行结果"""
    columns: List[str]
    rows: List[List[Any]]
    affected_rows: int = 0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None


class SQLGenerator(ABC):
    """SQL生成器抽象基类"""
    
    @abstractmethod
    async def generate_sql(
        self,
        query: str,
        schema: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """根据自然语言查询生成SQL"""
        pass
    
    @abstractmethod
    async def validate_sql(
        self,
        sql: str,
        schema: Dict[str, Any]
    ) -> bool:
        """验证SQL语句的安全性和正确性"""
        pass


class SQLExecutor(ABC):
    """SQL执行器抽象基类"""
    
    def __init__(self, config: SQLConfig):
        self.config = config
    
    @abstractmethod
    async def execute(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None
    ) -> SQLResult:
        """执行SQL语句"""
        pass
    
    @abstractmethod
    async def execute_batch(
        self,
        sqls: List[str],
        params_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[SQLResult]:
        """批量执行SQL语句"""
        pass


class SQLResultFormatter(ABC):
    """SQL结果格式化器抽象基类"""
    
    @abstractmethod
    async def format_result(
        self,
        result: SQLResult,
        format_type: str = "text"
    ) -> str:
        """将SQL结果格式化为可读形式"""
        pass
    
    @abstractmethod
    async def merge_results(
        self,
        results: List[SQLResult]
    ) -> SQLResult:
        """合并多个SQL结果"""
        pass


class SQLContext(BaseModel):
    """SQL上下文，用于存储数据库schema和执行环境信息"""
    tables: Dict[str, Dict[str, Any]]  # 表结构信息
    relationships: List[Dict[str, Any]] = []  # 表关系
    constraints: List[Dict[str, Any]] = []  # 约束条件
    metadata: Dict[str, Any] = {}  # 其他元数据


class SQLRegistry:
    """SQL组件注册表"""
    
    def __init__(self):
        self._generators: Dict[str, SQLGenerator] = {}
        self._executors: Dict[str, SQLExecutor] = {}
        self._formatters: Dict[str, SQLResultFormatter] = {}
        
    def register_generator(self, name: str, generator: SQLGenerator) -> None:
        """注册SQL生成器"""
        self._generators[name] = generator
        
    def register_executor(self, name: str, executor: SQLExecutor) -> None:
        """注册SQL执行器"""
        self._executors[name] = executor
        
    def register_formatter(self, name: str, formatter: SQLResultFormatter) -> None:
        """注册结果格式化器"""
        self._formatters[name] = formatter
        
    def get_generator(self, name: str) -> Optional[SQLGenerator]:
        """获取SQL生成器"""
        return self._generators.get(name)
        
    def get_executor(self, name: str) -> Optional[SQLExecutor]:
        """获取SQL执行器"""
        return self._executors.get(name)
        
    def get_formatter(self, name: str) -> Optional[SQLResultFormatter]:
        """获取结果格式化器"""
        return self._formatters.get(name) 