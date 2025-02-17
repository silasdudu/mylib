"""
Excel文档解析器
"""
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from base.core.logging import AsyncLogger, LogLevel
from base.rag.document import (Document, DocumentMetadata, DocumentParser,
                             DocumentType, TextDocument)


class ExcelParser(DocumentParser):
    """Excel文档解析器，支持.xlsx和.xls文件的解析"""
    
    def __init__(self, sheet_separator: str = "\n\n--- Sheet: {sheet_name} ---\n\n", logger: Optional[AsyncLogger] = None):
        """初始化Excel解析器
        
        Args:
            sheet_separator: 工作表之间的分隔符模板，使用{sheet_name}作为工作表名的占位符
            logger: 可选的日志记录器
        """
        super().__init__(logger)
        self.sheet_separator = sheet_separator
    
    async def parse(self, file_path: Path, doc_type: DocumentType) -> Document:
        """解析单个Excel文档
        
        Args:
            file_path: Excel文件路径
            doc_type: 文档类型，必须是 DocumentType.EXCEL
            
        Returns:
            解析后的文档对象
            
        Raises:
            ValueError: 当文档类型不是EXCEL时抛出
        """
        await self._log(LogLevel.INFO, f"开始解析Excel文档: {file_path}")
        
        if doc_type != DocumentType.EXCEL:
            await self._log(LogLevel.ERROR, f"文档类型不匹配: {doc_type}")
            raise ValueError(f"ExcelParser只支持EXCEL类型文档，收到: {doc_type}")
            
        if not file_path.exists():
            await self._log(LogLevel.ERROR, f"文件不存在: {file_path}")
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        try:
            # 读取Excel文件的所有工作表
            await self._log(LogLevel.DEBUG, "打开Excel文件")
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            content_parts = []
            total_rows = 0
            total_cells = 0
            
            await self._log(LogLevel.DEBUG, f"开始处理工作表，共 {len(sheet_names)} 个工作表")
            for sheet_name in sheet_names:
                await self._log(LogLevel.DEBUG, f"处理工作表: {sheet_name}")
                # 读取工作表
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # 添加工作表分隔符
                content_parts.append(self.sheet_separator.format(sheet_name=sheet_name))
                
                # 将数据框转换为字符串表示
                # 包含列名作为表头
                content_parts.append(df.to_string(index=False))
                
                # 统计信息
                total_rows += len(df)
                total_cells += df.size
                await self._log(LogLevel.DEBUG, f"工作表 {sheet_name} 包含 {len(df)} 行, {df.size} 个单元格")
            
            # 合并所有内容
            full_content = "\n".join(content_parts)
            
            # 创建元数据
            metadata = DocumentMetadata(
                doc_id=file_path.stem,
                doc_type=DocumentType.EXCEL,
                source=str(file_path),
                created_at=datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                extra={
                    'file_size': file_path.stat().st_size,
                    'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    'sheet_count': len(sheet_names),
                    'sheet_names': sheet_names,
                    'total_rows': total_rows,
                    'total_cells': total_cells
                }
            )
            
            await self._log(LogLevel.INFO, f"Excel文档解析完成: {file_path}")
            return TextDocument(full_content, metadata)
            
        except Exception as e:
            await self._log(LogLevel.ERROR, f"解析Excel文档时出错: {str(e)}")
            raise RuntimeError(f"解析Excel文档时出错: {str(e)}")
    
    async def parse_batch(
        self,
        file_paths: List[Path],
        doc_types: List[DocumentType]
    ) -> List[Document]:
        """批量解析Excel文档
        
        Args:
            file_paths: Excel文件路径列表
            doc_types: 文档类型列表，必须全部是EXCEL类型
            
        Returns:
            解析后的文档对象列表
        """
        await self._log(LogLevel.INFO, f"开始批量解析 {len(file_paths)} 个Excel文档")
        
        if not all(dt == DocumentType.EXCEL for dt in doc_types):
            await self._log(LogLevel.ERROR, "批量解析时发现不支持的文档类型")
            raise ValueError("ExcelParser只支持EXCEL类型文档")
            
        documents = []
        for file_path in file_paths:
            try:
                doc = await self.parse(file_path, DocumentType.EXCEL)
                documents.append(doc)
            except Exception as e:
                await self._log(LogLevel.ERROR, f"解析Excel文档 {file_path} 时出错: {str(e)}")
                continue
                
        await self._log(LogLevel.INFO, f"批量解析完成，成功解析 {len(documents)} 个文件")
        return documents 