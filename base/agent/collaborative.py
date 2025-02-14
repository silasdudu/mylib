"""
协作Agent模块，提供多Agent协作的基础框架
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

from .base import Agent, AgentAction, AgentConfig, AgentState
from .protocol import (AgentMessage, AgentRegistry, CommunicationProtocol,
                      Conversation, MessageType)


class CollaborativeAgentConfig(AgentConfig):
    """协作Agent配置"""
    team_id: str  # 团队ID
    role: str     # 角色
    capabilities: Set[str] = set()  # 能力集
    can_delegate: bool = True  # 是否可以委派任务
    can_coordinate: bool = True  # 是否可以协调其他Agent
    communication_timeout: float = 30.0  # 通信超时时间(秒)


class CollaborativeAgent(Agent):
    """协作Agent基类"""
    
    def __init__(
        self,
        config: CollaborativeAgentConfig,
        protocol: CommunicationProtocol,
        registry: AgentRegistry
    ):
        super().__init__(config)
        self.protocol = protocol
        self.registry = registry
        self.conversations: Dict[str, Conversation] = {}
        
        # 注册Agent
        self.registry.register_agent(
            self.config.name,
            {
                "team_id": config.team_id,
                "role": config.role,
                "status": self.state.status
            },
            config.capabilities
        )
    
    async def delegate_task(
        self,
        task: Any,
        required_capability: str
    ) -> Optional[str]:
        """委派任务给其他Agent"""
        if not self.config.can_delegate:
            return None
            
        # 查找具有所需能力的Agent
        candidates = self.registry.find_agents_by_capability(required_capability)
        if not candidates:
            return None
            
        # 创建任务委派消息
        message = AgentMessage(
            msg_id=f"task_{task}",
            msg_type=MessageType.REQUEST,
            sender_id=self.config.name,
            receiver_id=candidates[0],  # 简单起见，选择第一个候选者
            content={
                "task": task,
                "required_capability": required_capability
            }
        )
        
        # 发送消息
        success = await self.protocol.send_message(message)
        return candidates[0] if success else None
    
    async def coordinate_team(
        self,
        task: Any,
        team_members: List[str]
    ) -> bool:
        """协调团队成员"""
        if not self.config.can_coordinate:
            return False
            
        # 创建协调消息
        message = AgentMessage(
            msg_id=f"coord_{task}",
            msg_type=MessageType.PROPOSAL,
            sender_id=self.config.name,
            receiver_id="team",  # 广播给团队
            content={
                "task": task,
                "coordinator": self.config.name,
                "team_members": team_members
            }
        )
        
        # 广播消息
        results = await self.protocol.broadcast(message, team_members)
        return all(results.values())
    
    async def handle_message(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        """处理接收到的消息"""
        # 更新会话
        conversation_id = message.correlation_id or message.msg_id
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = Conversation(conversation_id)
        self.conversations[conversation_id].add_message(message)
        
        # 根据消息类型处理
        if message.msg_type == MessageType.REQUEST:
            return await self._handle_request(message)
        elif message.msg_type == MessageType.PROPOSAL:
            return await self._handle_proposal(message)
        elif message.msg_type == MessageType.QUERY:
            return await self._handle_query(message)
        else:
            return await self._handle_other(message)
    
    @abstractmethod
    async def _handle_request(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        """处理请求消息"""
        pass
    
    @abstractmethod
    async def _handle_proposal(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        """处理提议消息"""
        pass
    
    @abstractmethod
    async def _handle_query(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        """处理查询消息"""
        pass
    
    @abstractmethod
    async def _handle_other(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        """处理其他类型消息"""
        pass
    
    async def run_conversation_loop(self) -> None:
        """运行会话循环"""
        while True:
            # 接收消息
            message = await self.protocol.receive_message(
                timeout=self.config.communication_timeout
            )
            if message:
                # 处理消息
                response = await self.handle_message(message)
                if response:
                    # 发送响应
                    await self.protocol.send_message(response)
    
    async def cleanup(self) -> None:
        """清理资源"""
        # 注销Agent
        self.registry.unregister_agent(self.config.name)
        # 清理会话
        self.conversations.clear() 