"""
简单的RAG pipeline示例，展示如何使用基础工具库构建一个完整的RAG系统
"""
import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import List

from base.core.logging import AsyncLogger, ConsoleLogHandler, LogLevel
from base.model.interface import LargeModel, ModelConfig, ModelResponse
from base.rag.chunking import ChunkerConfig, TextChunker
from base.rag.document import (Document, DocumentMetadata, DocumentParser,
                             DocumentType, TextDocument)
from base.rag.embedding import DenseEmbeddingModel, EmbeddingConfig, EmbeddingOutput
from base.rag.generator import GeneratorConfig, GeneratorInput, RAGGenerator
from base.rag.retriever import RetrieverConfig, VectorRetriever, SearchResult


class SimpleDocumentParser(DocumentParser):
    """简单的文本文档解析器"""
    
    async def parse(self, file_path: Path, doc_type: DocumentType) -> Document:
        """解析文本文件"""
        if doc_type != DocumentType.TEXT:
            raise ValueError("仅支持文本文档")
            
        # 读取文件内容
        content = file_path.read_text(encoding="utf-8")
        
        # 创建元数据
        metadata = DocumentMetadata(
            doc_id=file_path.stem,
            doc_type=DocumentType.TEXT,
            source=str(file_path),
            created_at=datetime.now().isoformat()
        )
        
        return TextDocument(content, metadata)
    
    async def parse_batch(
        self,
        file_paths: List[Path],
        doc_types: List[DocumentType]
    ) -> List[Document]:
        """批量解析文档"""
        documents = []
        for file_path, doc_type in zip(file_paths, doc_types):
            doc = await self.parse(file_path, doc_type)
            documents.append(doc)
        return documents


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


async def main():
    # 设置日志
    logger = AsyncLogger()
    logger.add_handler(ConsoleLogHandler())
    await logger.log(LogLevel.INFO, "初始化RAG系统...")
    
    # 初始化组件
    doc_parser = SimpleDocumentParser()
    chunker = TextChunker(ChunkerConfig(
        chunk_size=1000,
        chunk_overlap=100
    ))
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
    docs_dir = Path("docs")  # 文档目录
    if not docs_dir.exists():
        await logger.log(LogLevel.ERROR, f"文档目录 {docs_dir} 不存在")
        return
        
    # 解析文档
    documents = []
    for file_path in docs_dir.glob("*.txt"):
        doc = await doc_parser.parse(file_path, DocumentType.TEXT)
        documents.append(doc)
    
    await logger.log(LogLevel.INFO, f"加载了 {len(documents)} 个文档")
    
    # 文档分块
    all_chunks = []
    for doc in documents:
        chunks = await chunker.split(doc)
        all_chunks.extend(chunks)
    
    await logger.log(LogLevel.INFO, f"生成了 {len(all_chunks)} 个文档块")
    
    # 构建索引
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