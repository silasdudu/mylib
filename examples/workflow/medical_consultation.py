"""
医疗咨询示例应用，展示如何使用基础工具库构建医疗问答系统
"""
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from base.dialogue.memory import (Memory, MemoryConfig, Message, MessageRole,
                                ShortTermMemory)
from base.model.interface import LargeModel, ModelConfig, ModelResponse
from base.rag.chunking import ChunkerConfig, TextChunker
from base.rag.document import DocumentParser, DocumentType, TextDocument
from base.model.embedding import DenseEmbeddingModel, EmbeddingConfig
from base.rag.generator import GeneratorConfig, RAGGenerator
from base.rag.retriever import RetrieverConfig, VectorRetriever
from base.search.engine import SearchConfig, SearchEngine, SearchResult
from examples.workflow.medical import (MedicalConfig, MedicalContext,
                                 MedicalWorkflow, QuestionClassifier,
                                 QuestionExpander, QuestionType, ResponseSelector,
                                 ResponseType)


class SimpleMedicalClassifier(QuestionClassifier):
    """简单的医疗问题分类器"""
    
    async def classify(self, question: str, context=None) -> QuestionType:
        """基于关键词的简单分类"""
        medical_keywords = ["症状", "疼痛", "治疗", "医生", "药", "病", "检查"]
        
        if any(kw in question for kw in medical_keywords):
            return QuestionType.MEDICAL
        return QuestionType.GENERAL
    
    async def validate(self, question: str) -> bool:
        """简单的问题验证"""
        return len(question.strip()) >= 5


class SimpleMedicalExpander(QuestionExpander):
    """简单的医疗问题扩写器"""
    
    async def expand(self, question: str, context: MedicalContext) -> str:
        """扩写问题，添加上下文信息"""
        history = context.chat_history[-3:] if context.chat_history else []
        history_text = "\n".join(f"{msg.role}: {msg.content}" for msg in history)
        
        return f"""基于以下上下文回答问题：

历史对话：
{history_text}

当前问题：{question}"""
    
    async def generate_sub_questions(
        self,
        question: str,
        context: MedicalContext
    ) -> List[str]:
        """生成相关的子问题"""
        return [
            f"这个症状持续多久了？",
            f"是否进行过相关检查？",
            f"有其他并发症状吗？"
        ]


class SimpleMedicalSelector(ResponseSelector):
    """简单的医疗回应选择器"""
    
    async def select_type(self, context: MedicalContext) -> ResponseType:
        """基于简单规则选择回应类型"""
        question = context.question.lower()
        
        if len(context.chat_history) <= 1:
            return ResponseType.ENCOURAGE
        elif "?" in question or "？" in question:
            return ResponseType.CONFIRM
        elif len(question) < 20:
            return ResponseType.INQUIRY
        else:
            return ResponseType.ANSWER
    
    async def generate_response(
        self,
        response_type: ResponseType,
        context: MedicalContext
    ) -> str:
        """生成对应类型的回应"""
        if response_type == ResponseType.ENCOURAGE:
            return "感谢您的咨询。为了更好地帮助您，能否详细描述一下您的具体症状？"
        elif response_type == ResponseType.INQUIRY:
            return "您提到的情况我了解了。请问：\n1. 这种情况持续多久了？\n2. 有没有其他不适？"
        elif response_type == ResponseType.CONFIRM:
            return f"""让我确认一下，您是说"{context.question}"，对吗？"""
        else:
            # 这里应该使用RAG和搜索结果生成专业回答
            return "基于您的描述，我的建议是..."


class SimpleSearchEngine(SearchEngine):
    """简单的搜索引擎实现"""
    
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """模拟搜索结果"""
        return [
            SearchResult(
                title="示例医疗文章",
                url="https://example.com/medical/1",
                snippet="这是一个相关的医疗信息片段...",
                source="示例来源",
                timestamp=datetime.now().isoformat()
            )
        ]
    
    async def get_content(self, url: str) -> str:
        """获取网页内容"""
        return "这是示例网页的内容..."


async def main():
    # 初始化配置
    medical_config = MedicalConfig()
    memory_config = MemoryConfig()
    search_config = SearchConfig()
    
    # 初始化组件
    memory = ShortTermMemory(max_messages=memory_config.short_term_size)
    search_engine = SimpleSearchEngine(search_config)
    
    # 初始化医疗工作流
    workflow = MedicalWorkflow(
        classifier=SimpleMedicalClassifier(),
        expander=SimpleMedicalExpander(),
        selector=SimpleMedicalSelector()
    )
    
    print("医疗咨询助手已启动，请描述您的问题（输入'q'退出）：")
    
    while True:
        # 获取用户输入
        user_input = input("\n您：").strip()
        if user_input.lower() == 'q':
            break
            
        # 记录用户消息
        user_message = Message(
            role=MessageRole.USER,
            content=user_input,
            timestamp=datetime.now()
        )
        await memory.add(user_message)
        
        # 准备上下文
        context = MedicalContext(
            question=user_input,
            question_type=QuestionType.GENERAL,
            chat_history=await memory.get_recent(5),
            doc_results=[],  # 这里应该添加实际的文档检索结果
            web_results=await search_engine.search(user_input)
        )
        
        # 处理问题
        response = await workflow.process(user_input, context)
        
        # 记录助手回复
        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=response,
            timestamp=datetime.now()
        )
        await memory.add(assistant_message)
        
        # 输出回复
        print(f"\n助手：{response}")


if __name__ == "__main__":
    asyncio.run(main()) 