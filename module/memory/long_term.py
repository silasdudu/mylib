"""
长期记忆实现，基于向量数据库存储和检索消息
"""
import json
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from base.dialogue.memory import Memory, Message, MessageRole


class LongTermMemory(Memory):
    """长期记忆，使用向量数据库存储和检索消息"""
    
    def __init__(
        self, 
        vector_store: Any, 
        embedding_model: Any,
        collection_name: str = "dialogue_memory",
        relevance_threshold: float = 0.7
    ):
        """初始化长期记忆
        
        Args:
            vector_store: 向量存储实例
            embedding_model: 嵌入模型实例
            collection_name: 集合名称
            relevance_threshold: 相关性阈值
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.relevance_threshold = relevance_threshold
        self._ensure_collection()
    
    def _ensure_collection(self) -> None:
        """确保集合存在"""
        # 根据具体的向量存储实现，确保集合存在
        # 这里是一个示例，实际实现可能需要根据具体的向量存储调整
        if hasattr(self.vector_store, "create_collection") and hasattr(self.vector_store, "has_collection"):
            # 对于支持 create_collection 和 has_collection 方法的向量存储
            if not self.vector_store.has_collection(self.collection_name):
                self.vector_store.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_model.embed_query
                )
        elif hasattr(self.vector_store, "_get_or_create_collection"):
            # 对于 ChromaVectorDB 类，它使用 _get_or_create_collection 方法
            # 这个方法已经在初始化时被调用了，所以这里不需要做任何事情
            pass
        # 如果没有上述方法，假设集合已经存在或者会在添加文档时自动创建
    
    async def _message_to_vector(self, message: Message) -> Dict[str, Any]:
        """将消息转换为向量存储的文档格式
        
        Args:
            message: 消息对象
            
        Returns:
            向量存储的文档格式
        """
        # 获取消息的嵌入向量
        embedding_result = await self.embedding_model.embed(message.content)
        
        # 处理嵌入结果，确保我们获取到正确的向量
        if hasattr(embedding_result, 'vector'):
            # 如果结果是一个具有 vector 属性的对象
            embedding = embedding_result.vector
        elif isinstance(embedding_result, (list, tuple)) and len(embedding_result) > 0:
            # 如果结果是一个列表或元组
            if isinstance(embedding_result[0], tuple) and len(embedding_result[0]) == 2 and embedding_result[0][0] == 'vector':
                # 如果是 [('vector', [...]), ...] 格式
                embedding = embedding_result[0][1]
            else:
                # 否则假设它是向量本身
                embedding = embedding_result
        else:
            # 其他情况，直接使用结果
            embedding = embedding_result
        
        # 构建文档
        return {
            "id": f"{message.role.value}_{message.timestamp.isoformat()}",
            "text": message.content,
            "metadata": {
                "role": message.role.value,
                "timestamp": message.timestamp.isoformat(),
                **message.metadata
            },
            "embedding": embedding
        }
    
    async def add(self, message: Message) -> None:
        """添加消息到记忆
        
        Args:
            message: 要添加的消息
        """
        doc = await self._message_to_vector(message)
        
        # 检查向量存储对象的类型和可用方法
        vector_store_type = type(self.vector_store).__name__
        
        # 对于 ChromaVectorDB 类
        if vector_store_type == "ChromaVectorDB":
            # 创建 Chunk 对象
            from base.rag.chunking import Chunk, ChunkMetadata
            
            # 提取消息ID作为文档ID
            doc_id = f"msg_{message.timestamp.isoformat()}"
            text = doc["text"]
            text_len = len(text)
            
            chunk = Chunk(
                text=text,
                metadata=ChunkMetadata(
                    chunk_id=doc["id"],
                    doc_id=doc_id,  # 添加必需的 doc_id 字段
                    start_char=0,   # 添加必需的 start_char 字段
                    end_char=text_len,  # 添加必需的 end_char 字段
                    text_len=text_len,  # 添加必需的 text_len 字段
                    extra=doc["metadata"]  # 将原始元数据放入 extra 字段
                )
            )
            
            # 确保向量是正确格式的二维数组
            embedding = doc["embedding"]
            
            # 导入 numpy 以便进行数组操作
            import numpy as np
            
            # 处理特殊情况：embedding 是元组列表 [('vector', [...]), ('metadata', {})]
            if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], tuple):
                # 尝试从元组列表中提取向量数据
                vector_data = None
                for item in embedding:
                    if isinstance(item, tuple) and len(item) == 2 and item[0] == 'vector':
                        vector_data = item[1]
                        break
                
                if vector_data is not None:
                    # 使用提取的向量数据
                    vectors = [vector_data]
                else:
                    # 如果无法提取向量数据，使用一个空的向量
                    print(f"警告：无法从元组列表中提取向量数据: {embedding}")
                    vectors = [[0.0] * self.vector_store.config.dimension]
            # 如果 embedding 是列表
            elif isinstance(embedding, list):
                # 确保它是二维数组，形状为 (1, n)
                if not any(isinstance(x, list) for x in embedding):
                    # 如果是一维列表，将其转换为二维列表
                    vectors = [embedding]
                else:
                    # 如果已经是二维列表，直接使用
                    vectors = embedding
            # 如果 embedding 是 numpy 数组
            elif isinstance(embedding, np.ndarray):
                # 检查维度
                if embedding.ndim == 1:
                    # 如果是一维数组，将其转换为二维数组
                    vectors = np.array([embedding])
                else:
                    # 如果已经是二维数组，直接使用
                    vectors = embedding
                # 转换为列表以便传递给 add_vectors
                vectors = vectors.tolist()
            else:
                # 其他情况，尝试转换为二维列表
                try:
                    # 尝试将 embedding 转换为列表
                    embedding_list = list(embedding)
                    # 检查是否是嵌套列表
                    if any(isinstance(x, (list, tuple)) for x in embedding_list):
                        vectors = embedding_list
                    else:
                        vectors = [embedding_list]
                except:
                    # 如果无法转换，使用一个空的二维数组
                    print(f"警告：无法将 embedding 转换为二维数组，类型: {type(embedding)}")
                    vectors = [[0.0] * self.vector_store.config.dimension]
            
            # 使用 add_vectors 方法
            await self.vector_store.add_vectors(vectors, [chunk])
        
        # 对于其他类型的向量存储
        elif hasattr(self.vector_store, "aadd"):
            # 使用 aadd 方法
            await self.vector_store.aadd(
                collection_name=self.collection_name,
                documents=[doc["text"]],
                metadatas=[doc["metadata"]],
                ids=[doc["id"]],
                embeddings=[doc["embedding"]]
            )
        else:
            # 如果没有支持的方法，抛出异常
            raise AttributeError(f"{vector_store_type} 对象没有支持的添加方法")
    
    async def get_recent(self, k: int) -> List[Message]:
        """获取最近k条消息
        
        Args:
            k: 要获取的消息数量
            
        Returns:
            最近的k条消息列表
        """
        # 检查向量存储对象的类型和可用方法
        vector_store_type = type(self.vector_store).__name__
        
        if vector_store_type == "ChromaVectorDB":
            # 对于 ChromaVectorDB 类，我们需要使用其他方法
            # 由于 ChromaVectorDB 不支持按时间戳排序，我们先获取所有消息，然后在内存中排序
            try:
                # 尝试获取所有消息
                # 注意：这种方法可能不适用于大量消息的情况
                all_results = await self.vector_store._collection.get(
                    include=["metadatas", "documents"]
                )
                
                if not all_results or not all_results["metadatas"]:
                    return []
                
                # 构建消息列表
                messages = []
                for i, metadata in enumerate(all_results["metadatas"]):
                    # 从 extra 字段中获取元数据
                    role = metadata.get("role") or metadata.get("extra", {}).get("role")
                    timestamp_str = metadata.get("timestamp") or metadata.get("extra", {}).get("timestamp")
                    
                    if role and timestamp_str:
                        messages.append(Message(
                            role=MessageRole(role),
                            content=all_results["documents"][i],
                            timestamp=datetime.fromisoformat(timestamp_str),
                            metadata={k: v for k, v in metadata.items() if k not in ["role", "timestamp"]}
                        ))
                
                # 按时间戳排序
                messages.sort(key=lambda msg: msg.timestamp, reverse=True)
                
                # 返回最近的k条消息
                return messages[:k]
            except Exception as e:
                print(f"获取最近消息时出错: {str(e)}")
                return []
        elif hasattr(self.vector_store, "aget"):
            # 对于支持 aget 方法的向量存储
            # 按时间戳排序获取最近的消息
            results = await self.vector_store.aget(
                collection_name=self.collection_name,
                limit=k,
                sort="metadata.timestamp",
                sort_order="desc"
            )
            
            if not results or not results["metadatas"]:
                return []
            
            messages = []
            for i, metadata in enumerate(results["metadatas"]):
                messages.append(Message(
                    role=MessageRole(metadata["role"]),
                    content=results["documents"][i],
                    timestamp=datetime.fromisoformat(metadata["timestamp"]),
                    metadata={k: v for k, v in metadata.items() if k not in ["role", "timestamp"]}
                ))
            
            return messages
        else:
            # 如果没有支持的方法，抛出异常
            raise AttributeError(f"{vector_store_type} 对象没有支持的获取最近消息的方法")
    
    async def search(self, query: str, limit: int = 5) -> List[Message]:
        """搜索相关消息
        
        Args:
            query: 搜索查询
            limit: 返回结果数量限制
            
        Returns:
            相关的消息列表
        """
        # 获取查询的嵌入向量
        embedding_result = await self.embedding_model.embed(query)
        
        # 处理嵌入结果，确保我们获取到正确的向量
        if hasattr(embedding_result, 'vector'):
            # 如果结果是一个具有 vector 属性的对象
            query_vector = embedding_result.vector
        elif isinstance(embedding_result, (list, tuple)) and len(embedding_result) > 0:
            # 如果结果是一个列表或元组
            if isinstance(embedding_result[0], tuple) and len(embedding_result[0]) == 2 and embedding_result[0][0] == 'vector':
                # 如果是 [('vector', [...]), ...] 格式
                query_vector = embedding_result[0][1]
            else:
                # 否则假设它是向量本身
                query_vector = embedding_result
        else:
            # 其他情况，直接使用结果
            query_vector = embedding_result
        
        # 检查向量存储对象的类型和可用方法
        vector_store_type = type(self.vector_store).__name__
        
        # 执行向量搜索
        if vector_store_type == "ChromaVectorDB":
            # 对于 ChromaVectorDB 类，使用 search 方法
            # 确保查询向量是正确的格式（一维列表）
            try:
                # 调用 search 方法
                results = await self.vector_store.search(query_vector, limit)
                
                if not results:
                    return []
                
                messages = []
                for result in results:
                    # 只返回相关性超过阈值的结果
                    if result.score >= self.relevance_threshold:
                        # 从 extra 字段中获取元数据
                        metadata = result.chunk.metadata.extra or {}
                        
                        # 检查是否存在必要的字段
                        if "role" not in metadata or "timestamp" not in metadata:
                            print(f"警告：搜索结果缺少必要的元数据字段: {metadata}")
                            continue
                            
                        try:
                            messages.append(Message(
                                role=MessageRole(metadata["role"]),
                                content=result.chunk.text,
                                timestamp=datetime.fromisoformat(metadata["timestamp"]),
                                metadata={k: v for k, v in metadata.items() if k not in ["role", "timestamp"]}
                            ))
                        except Exception as e:
                            print(f"创建消息对象时出错: {str(e)}, 元数据: {metadata}")
                            continue
                
                return messages
            except Exception as e:
                print(f"搜索时出错: {str(e)}")
                return []
        elif hasattr(self.vector_store, "asimilarity_search_with_score_by_vector"):
            # 对于其他类型的向量存储，使用 asimilarity_search_with_score_by_vector 方法
            results = await self.vector_store.asimilarity_search_with_score_by_vector(
                collection_name=self.collection_name,
                embedding=query_vector,
                k=limit
            )
            
            if not results:
                return []
            
            messages = []
            for doc, score in results:
                # 只返回相关性超过阈值的结果
                if score >= self.relevance_threshold:
                    metadata = doc.metadata
                    messages.append(Message(
                        role=MessageRole(metadata["role"]),
                        content=doc.page_content,
                        timestamp=datetime.fromisoformat(metadata["timestamp"]),
                        metadata={k: v for k, v in metadata.items() if k not in ["role", "timestamp"]}
                    ))
            
            return messages
        else:
            # 如果没有支持的方法，抛出异常
            raise AttributeError(f"{vector_store_type} 对象没有支持的搜索方法")
    
    async def clear(self) -> None:
        """清空记忆"""
        # 检查向量存储对象的类型和可用方法
        vector_store_type = type(self.vector_store).__name__
        
        if vector_store_type == "ChromaVectorDB":
            # 对于 ChromaVectorDB 类，使用 clear 方法
            await self.vector_store.clear()
        elif hasattr(self.vector_store, "adelete_collection"):
            # 对于支持 adelete_collection 方法的向量存储
            await self.vector_store.adelete_collection(self.collection_name)
        else:
            # 如果没有支持的方法，抛出异常
            raise AttributeError(f"{vector_store_type} 对象没有支持的清空方法")
            
        # 重新确保集合存在
        self._ensure_collection()
    
    async def get_formatted_history(self, formatter: Optional[Callable[[Message], str]] = None) -> str:
        """获取格式化的历史记录
        
        Args:
            formatter: 自定义格式化函数，如果为None则使用默认格式化
            
        Returns:
            格式化的历史记录字符串
        """
        messages = await self.get_recent(10)  # 获取最近10条消息
        
        if not messages:
            return ""
        
        if formatter:
            return "\n".join(formatter(msg) for msg in messages)
        
        # 默认格式化
        formatted = []
        for msg in messages:
            role_name = {
                MessageRole.SYSTEM: "系统",
                MessageRole.USER: "用户",
                MessageRole.ASSISTANT: "助手",
                MessageRole.FUNCTION: "函数"
            }.get(msg.role, str(msg.role))
            formatted.append(f"{role_name}: {msg.content}")
        
        return "\n".join(formatted)
    
    async def save(self, path: str) -> None:
        """保存记忆到文件
        
        Args:
            path: 保存路径
        """
        # 对于向量存储，通常已经持久化了，这里可以保存一些配置信息
        os.makedirs(os.path.dirname(path), exist_ok=True)
        config = {
            "collection_name": self.collection_name,
            "relevance_threshold": self.relevance_threshold,
            "vector_store_type": self.vector_store.__class__.__name__,
            "embedding_model_type": self.embedding_model.__class__.__name__
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    async def load(self, path: str) -> None:
        """从文件加载记忆配置
        
        Args:
            path: 加载路径
        """
        if not os.path.exists(path):
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            self.collection_name = config.get("collection_name", self.collection_name)
            self.relevance_threshold = config.get("relevance_threshold", self.relevance_threshold)
        
        self._ensure_collection() 