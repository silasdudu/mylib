"""
PDF文件解析器
"""
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
from base.core.logging import LogLevel
from base.rag.document import (Document, DocumentMetadata, DocumentType,
                             TextDocument, DocumentParser)
from base.core.logging import AsyncLogger


class PDFParser(DocumentParser):
    """PDF文件解析器，支持.pdf文件的解析"""
    
    def __init__(self, extract_images: bool = False, logger: Optional[AsyncLogger] = None):
        """初始化PDF解析器
        
        Args:
            extract_images: 是否提取PDF中的图片，默认为False
            logger: 可选的日志记录器
        """
        super().__init__(logger)
        self.extract_images = extract_images
    
    async def parse(self, file_path: Path, doc_type: DocumentType) -> Document:
        """解析单个PDF文件
        
        Args:
            file_path: PDF文件路径
            doc_type: 文档类型，必须是 DocumentType.PDF
            
        Returns:
            解析后的文档对象
            
        Raises:
            ValueError: 当文档类型不是PDF时抛出
        """
        await self._log(LogLevel.INFO, f"开始解析PDF文件: {file_path}")
        
        if doc_type != DocumentType.PDF:
            await self._log(LogLevel.ERROR, f"文档类型不匹配: {doc_type}")
            raise ValueError(f"PDFParser只支持PDF类型文档，收到: {doc_type}")
            
        if not file_path.exists():
            await self._log(LogLevel.ERROR, f"文件不存在: {file_path}")
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        try:
            # 打开PDF文件
            await self._log(LogLevel.DEBUG, "打开PDF文件")
            pdf_document = fitz.open(file_path)
            
            # 提取文本内容
            content = ""
            images = []
            
            await self._log(LogLevel.INFO, f"开始处理PDF文件，共 {len(pdf_document)} 页")
            for page_num in range(len(pdf_document)):
                await self._log(LogLevel.DEBUG, f"处理第 {page_num + 1} 页")
                page = pdf_document[page_num]
                content += page.get_text()
                
                # 如果需要提取图片
                if self.extract_images:
                    await self._log(LogLevel.DEBUG, f"从第 {page_num + 1} 页提取图片")
                    image_list = page.get_images()
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        if base_image:
                            images.append({
                                'page': page_num + 1,
                                'index': img_index,
                                'data': base_image["image"],
                                'extension': base_image["ext"]
                            })
                    await self._log(LogLevel.DEBUG, f"第 {page_num + 1} 页发现 {len(image_list)} 张图片")
            
            # 创建元数据
            metadata = DocumentMetadata(
                doc_id=file_path.stem,
                doc_type=DocumentType.PDF,
                source=str(file_path),
                created_at=datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                extra={
                    'file_size': file_path.stat().st_size,
                    'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    'page_count': len(pdf_document),
                    'pdf_info': dict(pdf_document.metadata),
                    'has_images': bool(images) if self.extract_images else None
                }
            )
            
            # 关闭PDF文件
            pdf_document.close()
            await self._log(LogLevel.INFO, f"PDF文件解析完成: {file_path}")
            
            return TextDocument(content, metadata)
            
        except Exception as e:
            await self._log(LogLevel.ERROR, f"解析PDF文件时出错: {str(e)}")
            raise RuntimeError(f"解析PDF文件时出错: {str(e)}")
    
    async def parse_batch(
        self,
        file_paths: List[Path],
        doc_types: List[DocumentType]
    ) -> List[Document]:
        """批量解析PDF文件
        
        Args:
            file_paths: PDF文件路径列表
            doc_types: 文档类型列表，必须全部是PDF类型
            
        Returns:
            解析后的文档对象列表
        """
        await self._log(LogLevel.INFO, f"开始批量解析 {len(file_paths)} 个PDF文件")
        
        if not all(dt == DocumentType.PDF for dt in doc_types):
            await self._log(LogLevel.ERROR, "批量解析时发现不支持的文档类型")
            raise ValueError("PDFParser只支持PDF类型文档")
            
        documents = []
        for file_path in file_paths:
            try:
                doc = await self.parse(file_path, DocumentType.PDF)
                documents.append(doc)
            except Exception as e:
                await self._log(LogLevel.ERROR, f"解析PDF文件 {file_path} 时出错: {str(e)}")
                continue
                
        await self._log(LogLevel.INFO, f"批量解析完成，成功解析 {len(documents)} 个文件")
        return documents 