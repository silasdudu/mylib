"""
文本文件解析器
"""
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from base.core.logging import AsyncLogger, LogLevel
from base.rag.document import (Document, DocumentMetadata, DocumentParser,
                             DocumentType)
from module.documents import TextDocument

class TextParser(DocumentParser):
    """文本文件解析器，支持.txt文件的解析"""
    
    def __init__(self, logger: Optional[AsyncLogger] = None):
        """初始化解析器
        
        Args:
            logger: 可选的日志记录器
        """
        super().__init__(logger)
    
    async def _log(self, level: LogLevel, message: str):
        """内部日志记录方法"""
        if self.logger:
            await self.logger.log(level, message)
    
    async def parse(self, file_path: Path, doc_type: DocumentType) -> Document:
        """解析单个文本文件
        
        Args:
            file_path: 文件路径
            doc_type: 文档类型，必须是 DocumentType.TEXT
            
        Returns:
            解析后的文档对象
            
        Raises:
            ValueError: 当文档类型不是TEXT时抛出
        """
        await self._log(LogLevel.INFO, f"开始解析文本文件: {file_path}")
        
        if doc_type != DocumentType.TEXT:
            await self._log(LogLevel.ERROR, f"文档类型不匹配: {doc_type}")
            raise ValueError(f"TextParser只支持TEXT类型文档，收到: {doc_type}")
            
        if not file_path.exists():
            await self._log(LogLevel.ERROR, f"文件不存在: {file_path}")
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        # 读取文件内容，尝试不同编码
        content = None
        encodings = ['utf-8', 'gbk', 'gb2312', 'ascii']
        
        for encoding in encodings:
            try:
                await self._log(LogLevel.DEBUG, f"尝试使用 {encoding} 编码读取文件")
                content = file_path.read_text(encoding=encoding)
                await self._log(LogLevel.DEBUG, f"成功使用 {encoding} 编码读取文件")
                break
            except UnicodeDecodeError:
                await self._log(LogLevel.DEBUG, f"{encoding} 编码读取失败，尝试下一个编码")
                continue
                
        if content is None:
            await self._log(LogLevel.ERROR, f"无法用支持的编码解析文件: {file_path}")
            raise UnicodeError(f"无法用支持的编码解析文件: {file_path}")
            
        # 清理文本内容
        # 1. 将多个连续空白字符替换为单个空格
        content = ' '.join(content.split())
        # 2. 将多个连续换行符替换为双换行符
        content = '\n\n'.join(p.strip() for p in content.split('\n') if p.strip())
            
        # 创建文档元数据
        metadata = DocumentMetadata(
            doc_id=file_path.stem,
            doc_type=DocumentType.TEXT,
            source=str(file_path),
            created_at=datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
            extra={
                "file_size": file_path.stat().st_size,
                "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }
        )
        
        await self._log(LogLevel.INFO, f"文本文件解析完成: {file_path}")
        return TextDocument(content, metadata)
    
    async def parse_batch(
        self,
        file_paths: List[Path],
        doc_types: List[DocumentType]
    ) -> List[Document]:
        """批量解析文本文件
        
        Args:
            file_paths: 文件路径列表
            doc_types: 文档类型列表，必须全部是TEXT类型
            
        Returns:
            解析后的文档对象列表
        """
        await self._log(LogLevel.INFO, f"开始批量解析 {len(file_paths)} 个文本文件")
        
        if not all(dt == DocumentType.TEXT for dt in doc_types):
            await self._log(LogLevel.ERROR, "批量解析时发现不支持的文档类型")
            raise ValueError("TextParser只支持TEXT类型文档")
            
        documents = []
        for file_path in file_paths:
            try:
                doc = await self.parse(file_path, DocumentType.TEXT)
                documents.append(doc)
            except Exception as e:
                await self._log(LogLevel.ERROR, f"解析文件 {file_path} 时出错: {str(e)}")
                continue
                
        await self._log(LogLevel.INFO, f"批量解析完成，成功解析 {len(documents)} 个文件")
        return documents 