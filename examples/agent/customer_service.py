"""
智能客服团队示例，展示多个Agent协作处理用户查询
"""
import asyncio
import uuid
from typing import Any, Dict, List, Optional, Set

from loguru import logger

from base.agent.broker import RedisMessageBroker, RedisProtocol
from base.agent.collaborative import CollaborativeAgent, CollaborativeAgentConfig
from base.agent.protocol import AgentMessage, AgentRegistry, MessageType
from base.rag.document import Document, DocumentType
from base.rag.retriever import SearchResult


class QueryDispatcherAgent(CollaborativeAgent):
    """查询分发Agent，负责接收和分发用户查询"""
    
    async def _handle_request(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        """处理用户查询请求"""
        query = message.content.get("query")
        if not query:
            return AgentMessage(
                msg_id=str(uuid.uuid4()),
                msg_type=MessageType.ERROR,
                sender_id=self.config.name,
                receiver_id=message.sender_id,
                content={"error": "查询内容不能为空"}
            )
            
        # 委派给知识库Agent进行检索
        kb_agent_id = await self.delegate_task(
            task={"action": "retrieve", "query": query},
            required_capability="knowledge_base"
        )
        
        if not kb_agent_id:
            return AgentMessage(
                msg_id=str(uuid.uuid4()),
                msg_type=MessageType.ERROR,
                sender_id=self.config.name,
                receiver_id=message.sender_id,
                content={"error": "未找到可用的知识库Agent"}
            )
            
        return AgentMessage(
            msg_id=str(uuid.uuid4()),
            msg_type=MessageType.INFORM,
            sender_id=self.config.name,
            receiver_id=message.sender_id,
            content={"status": "查询已分发", "handler": kb_agent_id}
        )
    
    async def _handle_proposal(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        return None
    
    async def _handle_query(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        return None
    
    async def _handle_other(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        return None


class KnowledgeBaseAgent(CollaborativeAgent):
    """知识库Agent，负责检索相关文档"""
    
    async def _handle_request(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        """处理检索请求"""
        task = message.content.get("task", {})
        if task.get("action") != "retrieve":
            return None
            
        query = task.get("query")
        # 模拟检索过程
        results = [
            SearchResult(
                chunk=Document(
                    content="这是一个示例回答",
                    metadata={"doc_id": "1", "doc_type": DocumentType.TEXT}
                ),
                score=0.9,
                metadata={"source": "FAQ"}
            )
        ]
        
        # 委派给回答生成Agent
        generator_id = await self.delegate_task(
            task={
                "action": "generate",
                "query": query,
                "context": results
            },
            required_capability="answer_generation"
        )
        
        if not generator_id:
            return AgentMessage(
                msg_id=str(uuid.uuid4()),
                msg_type=MessageType.ERROR,
                sender_id=self.config.name,
                receiver_id=message.sender_id,
                content={"error": "未找到可用的生成Agent"}
            )
            
        return AgentMessage(
            msg_id=str(uuid.uuid4()),
            msg_type=MessageType.INFORM,
            sender_id=self.config.name,
            receiver_id=message.sender_id,
            content={
                "status": "检索完成",
                "handler": generator_id,
                "results_count": len(results)
            }
        )
    
    async def _handle_proposal(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        return None
    
    async def _handle_query(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        return None
    
    async def _handle_other(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        return None


class AnswerGeneratorAgent(CollaborativeAgent):
    """回答生成Agent，负责生成最终答案"""
    
    async def _handle_request(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        """处理生成请求"""
        task = message.content.get("task", {})
        if task.get("action") != "generate":
            return None
            
        query = task.get("query")
        context = task.get("context", [])
        
        # 模拟答案生成
        answer = "这是基于检索结果生成的回答。"
        
        # 委派给质量检查Agent
        qa_id = await self.delegate_task(
            task={
                "action": "check",
                "query": query,
                "answer": answer,
                "context": context
            },
            required_capability="quality_check"
        )
        
        if not qa_id:
            return AgentMessage(
                msg_id=str(uuid.uuid4()),
                msg_type=MessageType.ERROR,
                sender_id=self.config.name,
                receiver_id=message.sender_id,
                content={"error": "未找到可用的质检Agent"}
            )
            
        return AgentMessage(
            msg_id=str(uuid.uuid4()),
            msg_type=MessageType.INFORM,
            sender_id=self.config.name,
            receiver_id=message.sender_id,
            content={
                "status": "答案已生成",
                "handler": qa_id
            }
        )
    
    async def _handle_proposal(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        return None
    
    async def _handle_query(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        return None
    
    async def _handle_other(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        return None


class QualityCheckerAgent(CollaborativeAgent):
    """质量检查Agent，负责检查答案质量"""
    
    async def _handle_request(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        """处理质检请求"""
        task = message.content.get("task", {})
        if task.get("action") != "check":
            return None
            
        query = task.get("query")
        answer = task.get("answer")
        context = task.get("context", [])
        
        # 模拟质量检查
        quality_score = 0.95
        feedback = "答案质量良好"
        
        return AgentMessage(
            msg_id=str(uuid.uuid4()),
            msg_type=MessageType.RESPONSE,
            sender_id=self.config.name,
            receiver_id=message.sender_id,
            content={
                "answer": answer,
                "quality_score": quality_score,
                "feedback": feedback
            }
        )
    
    async def _handle_proposal(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        return None
    
    async def _handle_query(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        return None
    
    async def _handle_other(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        return None


async def main():
    """运行示例"""
    # 创建消息代理
    broker = RedisMessageBroker(
        host="localhost",
        port=6379,
        prefix="customer_service"
    )
    await broker.connect()
    
    # 创建Agent注册表
    registry = AgentRegistry()
    
    # 创建各个Agent
    dispatcher = QueryDispatcherAgent(
        config=CollaborativeAgentConfig(
            name="dispatcher",
            team_id="customer_service",
            role="dispatcher",
            capabilities={"query_dispatch"}
        ),
        protocol=RedisProtocol(broker, "dispatcher"),
        registry=registry
    )
    
    kb_agent = KnowledgeBaseAgent(
        config=CollaborativeAgentConfig(
            name="kb_agent",
            team_id="customer_service",
            role="knowledge_base",
            capabilities={"knowledge_base"}
        ),
        protocol=RedisProtocol(broker, "kb_agent"),
        registry=registry
    )
    
    generator = AnswerGeneratorAgent(
        config=CollaborativeAgentConfig(
            name="generator",
            team_id="customer_service",
            role="generator",
            capabilities={"answer_generation"}
        ),
        protocol=RedisProtocol(broker, "generator"),
        registry=registry
    )
    
    qa_agent = QualityCheckerAgent(
        config=CollaborativeAgentConfig(
            name="qa_agent",
            team_id="customer_service",
            role="qa",
            capabilities={"quality_check"}
        ),
        protocol=RedisProtocol(broker, "qa_agent"),
        registry=registry
    )
    
    # 启动所有Agent的会话循环
    tasks = [
        asyncio.create_task(dispatcher.run_conversation_loop()),
        asyncio.create_task(kb_agent.run_conversation_loop()),
        asyncio.create_task(generator.run_conversation_loop()),
        asyncio.create_task(qa_agent.run_conversation_loop())
    ]
    
    # 模拟用户查询
    query_message = AgentMessage(
        msg_id=str(uuid.uuid4()),
        msg_type=MessageType.REQUEST,
        sender_id="user",
        receiver_id="dispatcher",
        content={"query": "如何重置密码？"}
    )
    
    # 发送查询
    await dispatcher.protocol.send_message(query_message)
    
    try:
        # 运行一段时间
        await asyncio.sleep(10)
    finally:
        # 取消所有任务
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # 清理资源
        await dispatcher.cleanup()
        await kb_agent.cleanup()
        await generator.cleanup()
        await qa_agent.cleanup()
        await broker.disconnect()


if __name__ == "__main__":
    asyncio.run(main()) 