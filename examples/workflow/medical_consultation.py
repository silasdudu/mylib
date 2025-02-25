"""
医疗咨询应用程序示例

这个示例展示了如何构建一个医疗问答系统，结合了以下组件：
- 医疗搜索引擎（支持多种搜索方式）
- 医疗RAG系统（检索增强生成）
- 医疗工作流（包括分类、扩展和响应选择）
- 对话记忆管理
"""

import os
import sys
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入模块
from module.parsers.pdf import PDFParser
from module.chunkers.text import TextChunker
from module.models.embedding.custom import CustomEmbedding
from module.vectordbs.chroma import ChromaVectorDB, ChromaVectorDBConfig
from module.models.llm.custom import CustomLLM
from module.models.rerank.custom import CustomReranker

# 导入异步日志类
from module.logging import AsyncLogger
from module.logging.console_logger import ColoredConsoleHandler
from module.logging.file_logger import FileHandler

from examples.workflow.medical.search import MedicalSearchEngine, MedicalSearchConfig
from examples.workflow.medical.retrieval import MedicalRAGSystem
from examples.workflow.medical import (
    MedicalQueryClassifier,
    MedicalQueryExpander,
    MedicalResponseSelector
)

# 配置日志
# 创建日志目录
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# 创建异步日志记录器
async_logger = AsyncLogger(module_name="examples.workflow.medical_consultation")

async def setup_async_logger():
    """设置异步日志记录器"""
    # 添加彩色控制台处理器
    console_handler = ColoredConsoleHandler(use_color=True)
    async_logger.add_handler(console_handler)
    
    # 添加文件处理器
    log_file = os.path.join(LOG_DIR, 'medical_consultation.log')
    file_handler = FileHandler(log_file)
    async_logger.add_handler(file_handler)
    
    # 启动日志记录器
    await async_logger.start()
    return async_logger


def load_medical_documents(rag_system: MedicalRAGSystem, docs_dir: str) -> None:
    """加载医疗文档到RAG系统
    
    Args:
        rag_system: RAG系统实例
        docs_dir: 文档目录
    """
    if not os.path.exists(docs_dir):
        async_logger.warning(f"文档目录不存在: {docs_dir}")
        return
    
    async_logger.info(f"开始加载医疗文档: {docs_dir}")
    
    # 创建PDF解析器和文本分块器
    pdf_parser = PDFParser()
    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    
    # 遍历目录中的PDF文件
    for filename in os.listdir(docs_dir):
        print(f"开始处理文件: {filename}")
        if filename.endswith('.pdf'):
            file_path = os.path.join(docs_dir, filename)
            # 解析PDF文件
            async_logger.info(f"解析PDF文件: {filename}")
            text = pdf_parser.parse(file_path)
            
            # 分块
            chunks = chunker.create_chunks(text)
            async_logger.info(f"从 {filename} 创建了 {len(chunks)} 个文本块")
            
            # 添加到RAG系统
            for chunk in chunks:
                # 设置元数据
                chunk.metadata['source'] = filename
                # 异步添加文档
                asyncio.create_task(rag_system.add_document(chunk.text, chunk.metadata))
                
    async_logger.info("医疗文档加载完成")


async def main():
    """主函数"""
    # 初始化日志
    await setup_async_logger()
    
    async_logger.info("医疗咨询应用启动")
    
    print("开始初始化嵌入模型...")
    # 初始化嵌入模型
    embedding_model = CustomEmbedding()
    print("嵌入模型初始化完成")
    
    print("开始初始化向量数据库...")
    # 初始化向量数据库
    vector_db_config = ChromaVectorDBConfig(
        collection_name="medical_docs",
        persist_directory=os.path.join(os.path.dirname(__file__), "data/vector_db"),
        dimension=3584,  # 添加必需的向量维度参数
        top_k=3,
        distance_metric="cosine",
    )
    vector_db = ChromaVectorDB(
        config=vector_db_config,
        logger=async_logger
    )
    print("向量数据库初始化完成")
    
    print("开始初始化RAG系统...")
    # 初始化RAG系统
    rag_system = MedicalRAGSystem(embedding_model=embedding_model, vector_db=vector_db)
    print("RAG系统初始化完成")
    
    print("开始加载医疗文档...")
    # 加载医疗文档
    docs_dir = os.path.join(os.path.dirname(__file__), "data/medical")
    load_medical_documents(rag_system, docs_dir)
    print("医疗文档加载完成")
    
    print("开始初始化LLM和重排序模型...")
    # 初始化LLM和重排序模型
    llm = CustomLLM()
    reranker = CustomReranker()
    print("LLM和重排序模型初始化完成")
    
    print("开始初始化搜索引擎...")
    # 初始化搜索引擎
    search_config = MedicalSearchConfig(
        primary_engine="basic",
        use_llm_filter=True,
        use_reranker=True
    )
    search_engine = MedicalSearchEngine(config=search_config, llm=llm, reranker=reranker)
    print("搜索引擎初始化完成")
    
    print("开始初始化查询分类器和扩展器...")
    # 初始化查询分类器和扩展器
    query_classifier = MedicalQueryClassifier(llm=llm)
    query_expander = MedicalQueryExpander(llm=llm)
    print("查询分类器和扩展器初始化完成")
    
    print("开始初始化响应选择器...")
    # 初始化响应选择器
    response_selector = MedicalResponseSelector(llm=llm)
    print("响应选择器初始化完成")
    
    print("开始进入主对话循环...")
    # 主对话循环
    while True:
        # 获取用户输入
        user_query = input("\n请输入您的医疗问题（输入'退出'结束对话）: ")
        
        # 检查是否退出
        if user_query.lower() in ["退出", "exit", "quit"]:
            break
            
        print(f"处理查询: {user_query}")
        async_logger.info(f"用户查询: {user_query}")
        
        # 分类查询
        print("开始分类查询...")
        query_type = await query_classifier.classify(user_query)
        print(f"查询类型: {query_type}")
        async_logger.info(f"查询类型: {query_type}")
        
        # 根据查询类型处理
        if query_type == "factual":
            print("处理事实型查询...")
            # 扩展查询
            expanded_query = await query_expander.expand(user_query)
            async_logger.info(f"扩展查询: {expanded_query}")
            
            # 从RAG系统检索相关文档
            print("从RAG系统检索相关文档...")
            rag_results = await rag_system.search(expanded_query, limit=5)
            
            # 从搜索引擎获取结果
            print("从搜索引擎获取结果...")
            search_results = await search_engine.search(expanded_query)
            
            # 合并上下文
            contexts = []
            for result in rag_results:
                contexts.append(result.chunk.text)
            for result in search_results:
                contexts.append(result.content)
            
            # 生成响应
            print("生成响应...")
            response = await response_selector.generate_factual_response(user_query, contexts)
            
        elif query_type == "personal":
            print("处理个人医疗咨询...")
            response = await response_selector.generate_personal_response(user_query)
            
        elif query_type == "general":
            print("处理一般医疗问题...")
            response = await response_selector.generate_general_response(user_query)
            
        else:  # non_medical
            print("处理非医疗问题...")
            response = "这似乎不是一个医疗相关的问题。我是一个医疗咨询助手，主要回答与医疗健康相关的问题。"
        
        # 输出响应
        print("\n回答:", response)
        async_logger.info(f"系统响应: {response[:100]}...")  # 只记录响应的前100个字符
    
    async_logger.info("医疗咨询应用正常关闭")
    print("医疗咨询应用已关闭")
    
    # 关闭日志
    await async_logger.stop()


if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main()) 