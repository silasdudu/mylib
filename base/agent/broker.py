"""
消息代理实现模块，提供基于Redis的消息传递机制
"""
import asyncio
import json
from typing import Any, Dict, List, Optional, Set

import aioredis
from loguru import logger

from .protocol import AgentMessage, CommunicationProtocol, MessageBroker


class RedisMessageBroker(MessageBroker):
    """基于Redis的消息代理实现"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "agent"
    ):
        self.redis_url = f"redis://{host}:{port}/{db}"
        self.password = password
        self.prefix = prefix
        self.redis: Optional[aioredis.Redis] = None
        self.pubsub: Optional[aioredis.client.PubSub] = None
        self.callbacks: Dict[str, callable] = {}
        self.running_tasks: Set[asyncio.Task] = set()
    
    async def connect(self) -> None:
        """连接到Redis"""
        try:
            self.redis = await aioredis.from_url(
                self.redis_url,
                password=self.password,
                decode_responses=True
            )
            self.pubsub = self.redis.pubsub()
            logger.info(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self) -> None:
        """断开Redis连接"""
        # 取消所有运行中的任务
        for task in self.running_tasks:
            task.cancel()
        await asyncio.gather(*self.running_tasks, return_exceptions=True)
        self.running_tasks.clear()
        
        # 关闭pubsub和redis连接
        if self.pubsub:
            await self.pubsub.close()
        if self.redis:
            await self.redis.close()
        
        logger.info("Disconnected from Redis")
    
    def _get_channel_key(self, channel_id: str) -> str:
        """获取通道键名"""
        return f"{self.prefix}:channel:{channel_id}"
    
    async def create_channel(self, channel_id: str) -> None:
        """创建通信通道"""
        if not self.redis:
            raise RuntimeError("Not connected to Redis")
            
        channel_key = self._get_channel_key(channel_id)
        # 使用Redis Set存储通道订阅者
        await self.redis.sadd(f"{channel_key}:subscribers", "")
        logger.info(f"Created channel {channel_id}")
    
    async def subscribe(
        self,
        channel_id: str,
        callback: callable
    ) -> None:
        """订阅通道"""
        if not self.redis or not self.pubsub:
            raise RuntimeError("Not connected to Redis")
            
        channel_key = self._get_channel_key(channel_id)
        
        # 添加到订阅者集合
        await self.redis.sadd(f"{channel_key}:subscribers", channel_id)
        
        # 保存回调函数
        self.callbacks[channel_id] = callback
        
        # 订阅通道
        await self.pubsub.subscribe(channel_key)
        
        # 启动消息处理任务
        task = asyncio.create_task(self._process_messages(channel_id))
        self.running_tasks.add(task)
        task.add_done_callback(self.running_tasks.discard)
        
        logger.info(f"Subscribed to channel {channel_id}")
    
    async def unsubscribe(self, channel_id: str) -> None:
        """取消订阅"""
        if not self.redis or not self.pubsub:
            raise RuntimeError("Not connected to Redis")
            
        channel_key = self._get_channel_key(channel_id)
        
        # 从订阅者集合中移除
        await self.redis.srem(f"{channel_key}:subscribers", channel_id)
        
        # 取消订阅
        await self.pubsub.unsubscribe(channel_key)
        
        # 移除回调
        self.callbacks.pop(channel_id, None)
        
        logger.info(f"Unsubscribed from channel {channel_id}")
    
    async def _process_messages(self, channel_id: str) -> None:
        """处理订阅消息"""
        if not self.pubsub:
            return
            
        channel_key = self._get_channel_key(channel_id)
        callback = self.callbacks.get(channel_id)
        
        try:
            async for message in self.pubsub.listen():
                if (
                    message["type"] == "message" and
                    message["channel"] == channel_key and
                    callback
                ):
                    try:
                        # 解析消息
                        data = json.loads(message["data"])
                        # 调用回调处理
                        await callback(data)
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
        except asyncio.CancelledError:
            logger.info(f"Message processing for {channel_id} cancelled")
        except Exception as e:
            logger.error(f"Message processing error: {e}")
    
    async def publish(
        self,
        channel_id: str,
        message: Dict[str, Any]
    ) -> bool:
        """发布消息到通道"""
        if not self.redis:
            raise RuntimeError("Not connected to Redis")
            
        channel_key = self._get_channel_key(channel_id)
        try:
            # 序列化消息
            message_data = json.dumps(message)
            # 发布到通道
            await self.redis.publish(channel_key, message_data)
            return True
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False


class RedisProtocol(CommunicationProtocol):
    """基于Redis的通信协议实现"""
    
    def __init__(
        self,
        broker: RedisMessageBroker,
        agent_id: str
    ):
        self.broker = broker
        self.agent_id = agent_id
        self._message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
    
    async def _message_handler(self, message_data: Dict[str, Any]) -> None:
        """处理接收到的消息"""
        try:
            # 将消息放入队列
            message = AgentMessage(**message_data)
            await self._message_queue.put(message)
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def send_message(
        self,
        message: AgentMessage
    ) -> bool:
        """发送消息"""
        try:
            # 发布消息到接收者的通道
            return await self.broker.publish(
                message.receiver_id,
                message.__dict__
            )
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    async def receive_message(
        self,
        timeout: Optional[float] = None
    ) -> Optional[AgentMessage]:
        """接收消息"""
        try:
            # 从队列获取消息
            return await asyncio.wait_for(
                self._message_queue.get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None
    
    async def broadcast(
        self,
        message: AgentMessage,
        receivers: List[str]
    ) -> Dict[str, bool]:
        """广播消息"""
        results = {}
        for receiver in receivers:
            # 为每个接收者创建新消息
            msg = AgentMessage(
                msg_id=message.msg_id,
                msg_type=message.msg_type,
                sender_id=self.agent_id,
                receiver_id=receiver,
                content=message.content,
                priority=message.priority,
                correlation_id=message.correlation_id,
                metadata=message.metadata
            )
            # 发送消息
            results[receiver] = await self.send_message(msg)
        return results 