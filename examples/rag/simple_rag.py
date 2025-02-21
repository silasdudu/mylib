"""
简单的RAG pipeline示例，展示如何使用基础工具库构建一个完整的RAG系统
"""
import asyncio
import os
from pathlib import Path
from base.core.logging import AsyncLogger, ConsoleLogHandler, LogLevel
from base.rag.chunking import ChunkerConfig
from base.rag.document import (Document, DocumentType, DocumentParserRegistry)
from base.rag.generator import GeneratorInput
from module.generators.rag import RAGGenerator, RAGGeneratorConfig
from module.parsers import (ExcelParser, MarkdownParser, PDFParser, TextParser,
                        WordParser)
from module.chunkers import SentenceChunker
from module.models.embedding import GTEQwenEmbedding
from module.models.llm.custom import CustomLLM, CustomLLMConfig
from module.vectordbs.chroma import ChromaVectorDB, ChromaVectorDBConfig
from module.prompts.rag import DEFAULT_RAG_TEMPLATE
from dotenv import load_dotenv

# 加载环境变量
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

# 获取通义千问配置
api_key = os.getenv('QWEN_72B_API_KEY')
base_url = os.getenv('QWEN_72B_BASE_URL')
model_name = os.getenv('QWEN_72B_MODEL_NAME')

if not all([api_key, base_url, model_name]):
    raise ValueError(
        "必需的环境变量未设置。请在 .env 文件中设置以下变量：\n"
        "- QWEN_72B_API_KEY\n"
        "- QWEN_72B_BASE_URL\n"
        "- QWEN_72B_MODEL_NAME"
    )



def setup_parser_registry(logger: AsyncLogger) -> DocumentParserRegistry:
    """设置文档解析器注册表
    
    Args:
        logger: 日志记录器
        
    Returns:
        配置好的文档解析器注册表
    """
    registry = DocumentParserRegistry()
    
    # 注册文本解析器
    registry.register(
        DocumentType.TEXT,
        TextParser(logger=logger),
        ['.txt']
    )
    
    # 注册PDF解析器
    registry.register(
        DocumentType.PDF,
        PDFParser(logger=logger, extract_images=False),
        ['.pdf']
    )
    
    # 注册Word解析器
    registry.register(
        DocumentType.WORD,
        WordParser(logger=logger),
        ['.docx']
    )
    
    # 注册Excel解析器
    registry.register(
        DocumentType.EXCEL,
        ExcelParser(logger=logger),
        ['.xlsx', '.xls']
    )
    
    # 注册Markdown解析器
    registry.register(
        DocumentType.MARKDOWN,
        MarkdownParser(logger=logger),
        ['.md']
    )
    
    return registry


async def process_document(file_path: Path, registry: DocumentParserRegistry, logger: AsyncLogger) -> Document:
    """处理单个文档
    
    Args:
        file_path: 文档路径
        registry: 文档解析器注册表
        logger: 日志记录器
        
    Returns:
        解析后的文档对象
        
    Raises:
        ValueError: 当文件类型不支持时抛出
    """
    try:
        parser, doc_type = registry.get_parser_for_file(file_path)
        return await parser.parse(file_path, doc_type)
    except ValueError as e:
        await logger.log(LogLevel.ERROR, str(e))
        raise


async def main():
    # 设置日志
    logger = AsyncLogger()
    logger.add_handler(ConsoleLogHandler())
    await logger.log(LogLevel.INFO, "初始化RAG系统...")
    
    # 初始化解析器注册表
    parser_registry = setup_parser_registry(logger)
    await logger.log(LogLevel.INFO, f"支持的文件类型: {parser_registry.list_supported_extensions()}")
    
    # 初始化其他组件
    chunker = SentenceChunker(
        config=ChunkerConfig(
            chunk_size=1000,
            chunk_overlap=100
        ),
        sentence_end_chars='.!?。！？',  # 支持中英文标点
        min_sentence_length=10,  # 最小句子长度
        logger=logger
    )
    
    # 使用 CustomEmbedding
    embedding_model = GTEQwenEmbedding()  # 使用默认配置，从 .env 加载
    
    # 使用 ChromaVectorDB
    persist_directory = Path(__file__).parent / "data" / "chroma"
    vector_db = ChromaVectorDB(
        config=ChromaVectorDBConfig(
            dimension=3584,  # 与嵌入模型维度匹配
            top_k=3,
            distance_metric="cosine",
            persist_directory=str(persist_directory),
            collection_name="document_chunks"
        ),
        logger=logger
    )
    
    # 使用 CustomLLM
    large_model = CustomLLM(
        config=CustomLLMConfig(
            model_name=model_name,
            api_url=base_url,
            api_key=api_key,
            temperature=0.7,
            max_tokens=2048
        )
    )
    
    generator = RAGGenerator(
        config=RAGGeneratorConfig(
            max_input_tokens=4096,
            max_output_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            prompt_template=DEFAULT_RAG_TEMPLATE["prompt_template"],
            context_format=DEFAULT_RAG_TEMPLATE["context_format"],
            context_separator=DEFAULT_RAG_TEMPLATE["context_separator"]
        ),
        model=large_model
    )
    
    # 检查是否需要重新构建索引
    docs_dir = Path(__file__).parent / "docs"
    if not docs_dir.exists():
        await logger.log(LogLevel.ERROR, f"文档目录 {docs_dir} 不存在")
        return
    
    # 如果数据库为空，则重新构建索引
    if not vector_db.is_ready:
        await logger.log(LogLevel.INFO, "数据库为空，开始构建索引...")
        
        # 解析所有支持的文档
        documents = []
        for file_path in docs_dir.iterdir():
            if parser_registry.is_supported_extension(file_path.suffix):
                try:
                    doc = await process_document(file_path, parser_registry, logger)
                    documents.append(doc)
                except Exception as e:
                    await logger.log(LogLevel.ERROR, f"处理文件 {file_path} 时出错: {str(e)}")
                    continue
        
        await logger.log(LogLevel.INFO, f"成功加载了 {len(documents)} 个文档")
        
        # 文档分块
        all_chunks = []
        await logger.log(LogLevel.INFO, "开始文档分块...")
        for doc in documents:
            await logger.log(LogLevel.INFO, f"处理文档: {doc.metadata.doc_id}")
            chunks = await chunker.split(doc)
            await logger.log(LogLevel.INFO, f"文档 {doc.metadata.doc_id} 生成了 {len(chunks)} 个块")
            all_chunks.extend(chunks)
        
        await logger.log(LogLevel.INFO, f"总共生成了 {len(all_chunks)} 个文档块")
        
        # 生成嵌入向量并构建索引
        await logger.log(LogLevel.INFO, "开始生成嵌入向量...")
        embeddings = await embedding_model.embed_chunks(all_chunks)
        vectors = [e.vector for e in embeddings]
        
        await logger.log(LogLevel.INFO, "开始构建索引...")
        await vector_db.create_index(vectors, all_chunks)
        await logger.log(LogLevel.INFO, "索引构建完成")
    else:
        await logger.log(LogLevel.INFO, "使用现有数据库，跳过索引构建")
    
    # 处理用户查询
    while True:
        query = input("\n请输入问题（输入'q'退出）: ").strip()
        if query.lower() == 'q':
            break
            
        # 生成查询向量
        query_embedding = await embedding_model.embed(query)
        
        # 检索相关文档
        results = await vector_db.search(query_embedding.vector)
        await logger.log(LogLevel.INFO, f"找到 {len(results)} 个相关文档块")
        
        # 生成回答
        generator_input = GeneratorInput(
            query=query,
            context=results
        )
        
        # 流式输出回答
        print("\n回答：", end="", flush=True)
        async for token in generator.generate_stream(generator_input):
            print(token, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    asyncio.run(main())