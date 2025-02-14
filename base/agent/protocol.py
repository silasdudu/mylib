"""
Agent通信协议模块，定义Agent之间的消息格式和通信规则
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel


class MessageType(str, Enum):
    """消息类型"""
    QUERY = "query"           # 查询请求
    RESPONSE = "response"     # 响应
    PROPOSAL = "proposal"     # 提议
    ACCEPT = "accept"         # 接受
    REJECT = "reject"         # 拒绝
    INFORM = "inform"         # 通知
    REQUEST = "request"       # 请求
    FEEDBACK = "feedback"     # 反馈
    STATUS = "status"         # 状态更新
    ERROR = "error"          # 错误


class MessagePriority(str, Enum):
    """消息优先级"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class AgentMessage:
    """Agent消息"""
    msg_id: str
    msg_type: MessageType
    sender_id: str
    receiver_id: str
    content: Any
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = None
    correlation_id: Optional[str] = None  # 关联消息ID
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class MessagePattern(BaseModel):
    """消息模式"""
    msg_type: MessageType
    schema: Dict[str, Any]  # 消息内容的JSON Schema
    required_fields: Set[str]
    validators: Dict[str, str]  # 字段验证规则


class CommunicationProtocol(ABC):
    """通信协议抽象基类"""
    
    @abstractmethod
    async def send_message(
        self,
        message: AgentMessage
    ) -> bool:
        """发送消息"""
        pass
    
    @abstractmethod
    async def receive_message(
        self,
        timeout: Optional[float] = None
    ) -> Optional[AgentMessage]:
        """接收消息"""
        pass
    
    @abstractmethod
    async def broadcast(
        self,
        message: AgentMessage,
        receivers: List[str]
    ) -> Dict[str, bool]:
        """广播消息"""
        pass


class MessageBroker(ABC):
    """消息代理抽象基类"""
    
    @abstractmethod
    async def connect(self) -> None:
        """连接到消息系统"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """断开连接"""
        pass
    
    @abstractmethod
    async def create_channel(
        self,
        channel_id: str
    ) -> None:
        """创建通信通道"""
        pass
    
    @abstractmethod
    async def subscribe(
        self,
        channel_id: str,
        callback: callable
    ) -> None:
        """订阅通道"""
        pass
    
    @abstractmethod
    async def unsubscribe(
        self,
        channel_id: str
    ) -> None:
        """取消订阅"""
        pass


class AgentRegistry:
    """Agent注册表"""
    
    def __init__(self):
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._capabilities: Dict[str, Set[str]] = {}
        self._channels: Dict[str, Set[str]] = {}
    
    def register_agent(
        self,
        agent_id: str,
        info: Dict[str, Any],
        capabilities: Set[str]
    ) -> None:
        """注册Agent"""
        self._agents[agent_id] = info
        self._capabilities[agent_id] = capabilities
    
    def unregister_agent(
        self,
        agent_id: str
    ) -> None:
        """注销Agent"""
        self._agents.pop(agent_id, None)
        self._capabilities.pop(agent_id, None)
        
        # 清理通道订阅
        for channel in list(self._channels.keys()):
            self._channels[channel].discard(agent_id)
    
    def get_agent_info(
        self,
        agent_id: str
    ) -> Optional[Dict[str, Any]]:
        """获取Agent信息"""
        return self._agents.get(agent_id)
    
    def find_agents_by_capability(
        self,
        capability: str
    ) -> List[str]:
        """查找具有特定能力的Agent"""
        return [
            agent_id
            for agent_id, caps in self._capabilities.items()
            if capability in caps
        ]
    
    def register_channel(
        self,
        channel_id: str,
        subscribers: Set[str]
    ) -> None:
        """注册通信通道"""
        self._channels[channel_id] = subscribers
    
    def get_channel_subscribers(
        self,
        channel_id: str
    ) -> Set[str]:
        """获取通道订阅者"""
        return self._channels.get(channel_id, set())


class Conversation:
    """会话管理"""
    
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.messages: List[AgentMessage] = []
        self.participants: Set[str] = set()
        self.status: str = "active"
        self.metadata: Dict[str, Any] = {}
    
    def add_message(
        self,
        message: AgentMessage
    ) -> None:
        """添加消息"""
        self.messages.append(message)
        self.participants.add(message.sender_id)
        self.participants.add(message.receiver_id)
    
    def get_messages(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[AgentMessage]:
        """获取消息历史"""
        filtered = self.messages
        if start:
            filtered = [m for m in filtered if m.timestamp >= start]
        if end:
            filtered = [m for m in filtered if m.timestamp <= end]
        return filtered
    
    def get_participant_messages(
        self,
        participant_id: str
    ) -> List[AgentMessage]:
        """获取参与者的消息"""
        return [
            m for m in self.messages
            if m.sender_id == participant_id or m.receiver_id == participant_id
        ] 