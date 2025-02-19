"""
Markdown文档解析器
"""
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import markdown
from bs4 import BeautifulSoup
from base.core.logging import AsyncLogger, LogLevel
from base.rag.document import (Document, DocumentMetadata, DocumentParser,
                             DocumentType, TextDocument)


class MarkdownParser(DocumentParser):
    """Markdown文档解析器，支持.md文件的解析"""
    
    def __init__(self, strip_html: bool = True, logger: Optional[AsyncLogger] = None):
        """初始化Markdown解析器
        
        Args:
            strip_html: 是否移除HTML标签，默认为True
            logger: 可选的日志记录器
        """
        super().__init__(logger)
        self.strip_html = strip_html
        self.md = markdown.Markdown(extensions=['extra', 'tables', 'toc'])
    
    async def parse(self, file_path: Path, doc_type: DocumentType) -> Document:
        """解析单个Markdown文档
        
        Args:
            file_path: Markdown文件路径
            doc_type: 文档类型，必须是 DocumentType.MARKDOWN
            
        Returns:
            解析后的文档对象
            
        Raises:
            ValueError: 当文档类型不是MARKDOWN时抛出
        """
        await self._log(LogLevel.INFO, f"开始解析Markdown文档: {file_path}")
        
        if doc_type != DocumentType.MARKDOWN:
            await self._log(LogLevel.ERROR, f"文档类型不匹配: {doc_type}")
            raise ValueError(f"MarkdownParser只支持MARKDOWN类型文档，收到: {doc_type}")
            
        if not file_path.exists():
            await self._log(LogLevel.ERROR, f"文件不存在: {file_path}")
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        try:
            # 读取文件内容
            await self._log(LogLevel.DEBUG, "读取Markdown文件")
            content = file_path.read_text(encoding='utf-8')
            
            # 转换Markdown为HTML
            await self._log(LogLevel.DEBUG, "转换Markdown为HTML")
            html_content = self.md.convert(content)
            
            # 如果需要移除HTML标签
            if self.strip_html:
                await self._log(LogLevel.DEBUG, "移除HTML标签")
                soup = BeautifulSoup(html_content, 'html.parser')
                # 获取文本并清理空白
                paragraphs = [p.strip() for p in soup.get_text(separator='\n').split('\n') if p.strip()]
                text_content = '\n\n'.join(paragraphs)
            else:
                text_content = html_content
            
            # 提取标题结构
            await self._log(LogLevel.DEBUG, "提取文档结构")
            headers = []
            soup = BeautifulSoup(html_content, 'html.parser')
            for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                headers.append({
                    'level': int(header.name[1]),
                    'text': header.get_text()
                })
            
            # 创建元数据
            metadata = DocumentMetadata(
                doc_id=file_path.stem,
                doc_type=DocumentType.MARKDOWN,
                source=str(file_path),
                created_at=datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                extra={
                    'file_size': file_path.stat().st_size,
                    'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    'headers': headers,
                    'has_tables': bool(soup.find_all('table')),
                    'has_code_blocks': bool(soup.find_all('code'))
                }
            )
            
            await self._log(LogLevel.INFO, f"Markdown文档解析完成: {file_path}")
            return TextDocument(text_content, metadata)
            
        except Exception as e:
            await self._log(LogLevel.ERROR, f"解析Markdown文档时出错: {str(e)}")
            raise RuntimeError(f"解析Markdown文档时出错: {str(e)}")
    
    async def parse_batch(
        self,
        file_paths: List[Path],
        doc_types: List[DocumentType]
    ) -> List[Document]:
        """批量解析Markdown文档
        
        Args:
            file_paths: Markdown文件路径列表
            doc_types: 文档类型列表，必须全部是MARKDOWN类型
            
        Returns:
            解析后的文档对象列表
        """
        await self._log(LogLevel.INFO, f"开始批量解析 {len(file_paths)} 个Markdown文档")
        
        if not all(dt == DocumentType.MARKDOWN for dt in doc_types):
            await self._log(LogLevel.ERROR, "批量解析时发现不支持的文档类型")
            raise ValueError("MarkdownParser只支持MARKDOWN类型文档")
            
        documents = []
        for file_path in file_paths:
            try:
                doc = await self.parse(file_path, DocumentType.MARKDOWN)
                documents.append(doc)
            except Exception as e:
                await self._log(LogLevel.ERROR, f"解析Markdown文档 {file_path} 时出错: {str(e)}")
                continue
                
        await self._log(LogLevel.INFO, f"批量解析完成，成功解析 {len(documents)} 个文件")
        return documents 