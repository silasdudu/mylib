"""
Agent基类模块，定义Agent的基本接口和行为
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar

from pydantic import BaseModel

from ..core.exceptions import BaseError

T = TypeVar("T")


class AgentStatus(str, Enum):
    """Agent状态"""
    IDLE = "IDLE"
    THINKING = "THINKING"
    ACTING = "ACTING"
    WAITING = "WAITING"
    ERROR = "ERROR"


class AgentAction(BaseModel):
    """Agent行动"""
    action_type: str
    parameters: Dict[str, Any] = {}
    description: Optional[str] = None


@dataclass
class AgentState:
    """Agent状态"""
    status: AgentStatus
    current_task: Optional[str] = None
    memory: Dict[str, Any] = None
    last_action: Optional[AgentAction] = None


class AgentConfig(BaseModel):
    """Agent配置"""
    name: str
    description: Optional[str] = None
    max_steps: int = 10
    timeout: float = 60.0
    extra_params: Dict[str, Any] = {}


class AgentError(BaseError):
    """Agent异常"""
    pass


class Agent(ABC):
    """Agent抽象基类"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.state = AgentState(status=AgentStatus.IDLE)
        
    @abstractmethod
    async def think(self) -> AgentAction:
        """思考下一步行动"""
        pass
    
    @abstractmethod
    async def act(self, action: AgentAction) -> Any:
        """执行行动"""
        pass
    
    @abstractmethod
    async def observe(self, observation: Any) -> None:
        """观察环境"""
        pass
    
    async def step(self) -> Any:
        """执行一个完整的思考-行动循环"""
        try:
            self.state.status = AgentStatus.THINKING
            action = await self.think()
            
            self.state.status = AgentStatus.ACTING
            self.state.last_action = action
            result = await self.act(action)
            
            await self.observe(result)
            self.state.status = AgentStatus.IDLE
            
            return result
        except Exception as e:
            self.state.status = AgentStatus.ERROR
            raise AgentError(str(e))


class Tool(ABC):
    """工具抽象基类"""
    
    @abstractmethod
    async def run(self, **kwargs) -> Any:
        """运行工具"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """工具参数说明"""
        pass


class ToolRegistry:
    """工具注册器"""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        
    def register(self, name: str, tool: Tool) -> None:
        """注册工具"""
        self._tools[name] = tool
        
    def get(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """列出所有工具"""
        return list(self._tools.keys())
    
    def get_descriptions(self) -> Dict[str, str]:
        """获取所有工具的描述"""
        return {name: tool.description for name, tool in self._tools.items()} 