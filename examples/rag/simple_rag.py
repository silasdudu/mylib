"""
简单的RAG pipeline示例，展示如何使用基础工具库构建一个完整的RAG系统
"""
import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Type

from base.core.logging import AsyncLogger, ConsoleLogHandler, LogLevel
from base.model.interface import LargeModel, ModelConfig, ModelResponse
from base.rag.chunking import ChunkerConfig
from base.rag.document import (Document, DocumentType, DocumentParserRegistry)
from base.model.embedding import DenseEmbeddingModel, EmbeddingConfig, EmbeddingOutput
from base.rag.generator import GeneratorConfig, GeneratorInput, RAGGenerator
from base.rag.retriever import RetrieverConfig, VectorRetriever, SearchResult
from rag.parsers import (ExcelParser, MarkdownParser, PDFParser, TextParser,
                        WordParser)
from rag.chunkers import SentenceChunker


class SimpleEmbeddingModel(DenseEmbeddingModel):
    """简单的嵌入模型（仅作示例）"""
    
    async def embed(self, text: str) -> EmbeddingOutput:
        """生成文本嵌入（这里使用随机向量作为示例）"""
        import numpy as np
        vector = np.random.randn(self.config.dimension)
        if self.config.normalize:
            vector = self.normalize_vector(vector)
        return EmbeddingOutput(vector=vector.tolist())


class SimpleLargeModel(LargeModel):
    """简单的大模型（仅作示例）"""
    
    async def generate(
        self,
        prompt: str,
        config: ModelConfig = None
    ) -> ModelResponse:
        """生成文本（示例实现）"""
        # 这里应该实现实际的大模型调用
        # 示例中仅返回一个固定回答
        return ModelResponse(
            text="这是一个示例回答。在实际应用中，这里应该是大模型生成的回答。",
            tokens_used=10,
            finish_reason="completed"
        )
    
    async def generate_stream(
        self,
        prompt: str,
        config: ModelConfig = None
    ):
        """流式生成（示例实现）"""
        response = await self.generate(prompt, config)
        yield response.text
    
    async def embed(self, text: str):
        """生成文本嵌入（示例实现）"""
        raise NotImplementedError()


class SimpleVectorRetriever(VectorRetriever):
    """简单的向量检索器（仅作示例）"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._index = {}  # 简单的内存存储
    
    async def index(self, chunks, embeddings=None):
        """索引文档块"""
        if not embeddings:
            embeddings = await self.embedding_model.embed_chunks(chunks)
        
        for chunk, embedding in zip(chunks, embeddings):
            self._index[chunk.metadata.chunk_id] = {
                "chunk": chunk,
                "vector": embedding.vector
            }
    
    async def search(self, query: str, top_k: int = None):
        """搜索相关内容（示例实现：随机返回结果）"""
        import random
        k = top_k or self.config.top_k
        results = []
        
        # 随机选择k个结果
        items = list(self._index.values())
        if items:
            selected = random.sample(items, min(k, len(items)))
            for item in selected:
                results.append(SearchResult(
                    chunk=item["chunk"],
                    score=random.random()
                ))
        
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    async def delete(self, chunk_ids: List[str]):
        """删除索引"""
        for chunk_id in chunk_ids:
            self._index.pop(chunk_id, None)


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
    
    embedding_model = SimpleEmbeddingModel(EmbeddingConfig(
        model_name="simple_embedding",
        dimension=768
    ))
    retriever = SimpleVectorRetriever(
        config=RetrieverConfig(top_k=3),
        embedding_model=embedding_model
    )
    large_model = SimpleLargeModel()
    generator = RAGGenerator(
        config=GeneratorConfig(
            max_input_tokens=4096,
            max_output_tokens=1024
        ),
        model=large_model
    )
    
    # 加载和处理文档
    docs_dir = Path(__file__).parent / "docs"
    if not docs_dir.exists():
        await logger.log(LogLevel.ERROR, f"文档目录 {docs_dir} 不存在")
        return
        
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
    
    # 构建索引
    await logger.log(LogLevel.INFO, "开始构建索引...")
    await retriever.index(all_chunks)
    await logger.log(LogLevel.INFO, "索引构建完成")
    
    # 处理用户查询
    while True:
        query = input("\n请输入问题（输入'q'退出）: ").strip()
        if query.lower() == 'q':
            break
            
        # 检索相关文档
        results = await retriever.search(query)
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