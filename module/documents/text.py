"""
文本文档实现
"""
from base.rag.document import Document, DocumentMetadata


class TextDocument(Document):
    """文本文档"""
    
    def __init__(
        self,
        content: str,
        metadata: DocumentMetadata
    ):
        super().__init__(content, metadata)
        
    async def preprocess(self) -> None:
        """预处理文本，如清理、标准化等"""
        # 实现文本预处理逻辑
        pass
        
    async def to_text(self) -> str:
        """直接返回文本内容"""
        return self.content
        
    def get_size(self) -> int:
        """返回文本长度"""
        return len(self.content) 