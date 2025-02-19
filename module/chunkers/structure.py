"""
基于文档结构的切分器，支持 Markdown、JSON、代码块和 HTML 等结构化文档的切分
"""
import json
import re
from typing import List, Optional, Dict, Any
from bs4 import BeautifulSoup
from markdown import markdown

from base.core.logging import AsyncLogger, LogLevel
from base.rag.chunking import Chunk, ChunkMetadata, Chunker, ChunkerConfig
from base.rag.document import Document, DocumentType


class StructureChunker(Chunker):
    """基于文档结构的切分器，根据文档的特定结构进行切分"""
    
    def __init__(
        self,
        config: ChunkerConfig,
        preserve_structure: bool = True,
        logger: Optional[AsyncLogger] = None
    ):
        """初始化切分器
        
        Args:
            config: 切分器配置
            preserve_structure: 是否在切分时保留文档结构标记
            logger: 可选的日志记录器
        """
        super().__init__(config, logger)
        self.preserve_structure = preserve_structure
    
    async def _split_markdown(self, text: str) -> List[str]:
        """切分 Markdown 文档
        
        Args:
            text: Markdown 文本
            
        Returns:
            文本块列表
        """
        # 分割标题
        header_pattern = r'^#{1,6}\s+.*$'
        # 分割代码块
        code_block_pattern = r'```[\s\S]*?```'
        # 分割列表
        list_pattern = r'^[-*+]\s+.*$'
        
        # 将文本按照这些模式分割
        parts = []
        current_part = []
        
        for line in text.split('\n'):
            if (re.match(header_pattern, line) or 
                re.match(code_block_pattern, line) or 
                re.match(list_pattern, line)):
                if current_part:
                    parts.append('\n'.join(current_part))
                    current_part = []
            current_part.append(line)
            
        if current_part:
            parts.append('\n'.join(current_part))
            
        return parts
    
    async def _split_json(self, text: str) -> List[str]:
        """切分 JSON 文档
        
        Args:
            text: JSON 文本
            
        Returns:
            文本块列表
        """
        try:
            data = json.loads(text)
            parts = []
            
            def process_value(value: Any, path: str = '') -> None:
                if isinstance(value, dict):
                    for k, v in value.items():
                        new_path = f"{path}.{k}" if path else k
                        if isinstance(v, (dict, list)):
                            process_value(v, new_path)
                        else:
                            parts.append(f"{new_path}: {str(v)}")
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        new_path = f"{path}[{i}]"
                        if isinstance(item, (dict, list)):
                            process_value(item, new_path)
                        else:
                            parts.append(f"{new_path}: {str(item)}")
                else:
                    parts.append(f"{path}: {str(value)}")
            
            process_value(data)
            return parts
        except json.JSONDecodeError:
            await self._log(LogLevel.WARNING, "JSON 解析失败，将按普通文本处理")
            return [text]
    
    async def _split_html(self, text: str) -> List[str]:
        """切分 HTML 文档
        
        Args:
            text: HTML 文本
            
        Returns:
            文本块列表
        """
        soup = BeautifulSoup(text, 'html.parser')
        parts = []
        
        # 处理标题
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            parts.append(tag.get_text())
        
        # 处理段落
        for p in soup.find_all('p'):
            parts.append(p.get_text())
        
        # 处理列表
        for ul in soup.find_all(['ul', 'ol']):
            list_text = []
            for li in ul.find_all('li'):
                list_text.append(f"- {li.get_text()}")
            parts.append('\n'.join(list_text))
        
        # 处理表格
        for table in soup.find_all('table'):
            table_text = []
            for row in table.find_all('tr'):
                cells = [cell.get_text().strip() for cell in row.find_all(['th', 'td'])]
                table_text.append(' | '.join(cells))
            parts.append('\n'.join(table_text))
        
        return parts
    
    async def _split_code(self, text: str) -> List[str]:
        """切分代码文档
        
        Args:
            text: 代码文本
            
        Returns:
            文本块列表
        """
        parts = []
        current_part = []
        
        # 函数/类定义模式
        definition_pattern = r'^(def|class)\s+\w+.*:$'
        # 注释块模式
        comment_block_pattern = r'^(\s*#.*|\s*/\*[\s\S]*?\*/|\s*"""[\s\S]*?""")$'
        
        for line in text.split('\n'):
            # 如果遇到新的函数/类定义或注释块
            if (re.match(definition_pattern, line) or 
                re.match(comment_block_pattern, line)):
                if current_part:
                    parts.append('\n'.join(current_part))
                    current_part = []
            current_part.append(line)
            
        if current_part:
            parts.append('\n'.join(current_part))
            
        return parts
    
    async def split(self, document: Document) -> List[Chunk]:
        """根据文档类型和结构进行切分
        
        Args:
            document: 要切分的文档
            
        Returns:
            切分后的文本块列表
        """
        await self._log(LogLevel.INFO, f"开始按结构切分文档: {document.metadata.doc_id}")
        
        # 获取文本内容
        text = await document.to_text()
        if not text:
            await self._log(LogLevel.WARNING, "文档内容为空")
            return []
            
        # 根据文档类型选择切分方法
        doc_type = document.metadata.doc_type
        parts = []
        
        if doc_type == DocumentType.MARKDOWN:
            parts = await self._split_markdown(text)
        elif doc_type == DocumentType.HTML:
            parts = await self._split_html(text)
        else:
            # 尝试检测文档格式
            if text.startswith('{') and text.endswith('}'):
                parts = await self._split_json(text)
            elif '```' in text or 'def ' in text or 'class ' in text:
                parts = await self._split_code(text)
            else:
                # 如果无法识别结构，按段落切分
                parts = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        await self._log(LogLevel.DEBUG, f"初步分割得到 {len(parts)} 个结构块")
        
        # 将结构块组合成最终的文本块
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0
        start_char = 0
        
        for part in parts:
            part_length = len(part)
            
            # 如果当前块加上新部分超过最大长度，创建新块
            if current_length + part_length > self.config.chunk_size and current_chunk:
                # 创建当前块
                chunk_text = '\n\n'.join(current_chunk)
                metadata = ChunkMetadata(
                    chunk_id=f"{document.metadata.doc_id}_chunk_{chunk_id}",
                    doc_id=document.metadata.doc_id,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    text_len=len(chunk_text)
                )
                
                chunk = Chunk(text=chunk_text, metadata=metadata)
                chunks.append(chunk)
                await self._log(LogLevel.DEBUG, f"创建块 {metadata.chunk_id}, 长度: {metadata.text_len}")
                
                # 重置当前块
                current_chunk = []
                current_length = 0
                chunk_id += 1
                start_char = start_char + len(chunk_text)
            
            # 添加新部分到当前块
            current_chunk.append(part)
            current_length += part_length
        
        # 处理最后一个块
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            metadata = ChunkMetadata(
                chunk_id=f"{document.metadata.doc_id}_chunk_{chunk_id}",
                doc_id=document.metadata.doc_id,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                text_len=len(chunk_text)
            )
            
            chunk = Chunk(text=chunk_text, metadata=metadata)
            chunks.append(chunk)
            await self._log(LogLevel.DEBUG, f"创建最后一个块 {metadata.chunk_id}, 长度: {metadata.text_len}")
        
        await self._log(LogLevel.INFO, f"文档切分完成，共生成 {len(chunks)} 个块")
        return chunks

    async def split_batch(self, documents: List[Document]) -> List[List[Chunk]]:
        """批量处理多个文档
        
        Args:
            documents: 要处理的文档列表
            
        Returns:
            每个文档对应的文本块列表
        """
        await self._log(LogLevel.INFO, f"开始批量处理 {len(documents)} 个文档")
        results = []
        
        for doc in documents:
            try:
                chunks = await self.split(doc)
                results.append(chunks)
                await self._log(LogLevel.DEBUG, f"文档 {doc.metadata.doc_id} 生成了 {len(chunks)} 个块")
            except Exception as e:
                await self._log(LogLevel.ERROR, f"处理文档 {doc.metadata.doc_id} 时出错: {str(e)}")
                results.append([])
                
        await self._log(LogLevel.INFO, "批量处理完成")
        return results
        
    async def merge(self, chunks: List[Chunk]) -> Chunk:
        """合并多个文本块
        
        Args:
            chunks: 要合并的文本块列表
            
        Returns:
            合并后的文本块
            
        Raises:
            ValueError: 当块列表为空时抛出
        """
        if not chunks:
            raise ValueError("无法合并空的块列表")
            
        # 按照块的顺序合并文本，使用双换行符保持结构
        merged_text = '\n\n'.join(chunk.text for chunk in chunks)
        
        # 使用第一个块的文档ID
        doc_id = chunks[0].metadata.doc_id
        
        # 创建新的元数据
        metadata = ChunkMetadata(
            chunk_id=f"{doc_id}_merged_{chunks[0].metadata.chunk_id}",
            doc_id=doc_id,
            start_char=chunks[0].metadata.start_char,
            end_char=chunks[-1].metadata.end_char,
            text_len=len(merged_text)
        )
        
        return Chunk(text=merged_text, metadata=metadata) 