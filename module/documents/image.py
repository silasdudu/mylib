"""
图像文档实现
"""
from base.rag.document import Document, DocumentMetadata


class ImageDocument(Document):
    """图像文档"""
    
    def __init__(
        self,
        content: bytes,
        metadata: DocumentMetadata
    ):
        super().__init__(content, metadata)
        
    async def preprocess(self) -> None:
        """预处理图像，如调整大小、标准化等"""
        # 实现图像预处理逻辑
        pass
        
    async def to_text(self) -> str:
        """将图像转换为文本描述（如通过图像识别）"""
        # 实现图像到文本的转换
        raise NotImplementedError()
        
    def get_size(self) -> int:
        """返回图像大小（字节）"""
        return len(self.content) 