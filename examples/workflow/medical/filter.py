"""
医疗内容过滤器模块

提供基于大模型和重排序的医疗内容过滤功能，用于筛选和排序医疗相关搜索结果
"""

import json
from typing import List, Dict, Any, Optional, Tuple
import asyncio

from base.search.engine import SearchResult
from module.models.llm.custom import CustomLLM
from module.models.rerank.custom import CustomReranker


class LLMBasedMedicalFilter:
    """基于大模型和重排序的医疗内容过滤器
    
    使用大语言模型和重排序模型对搜索结果进行过滤和排序，
    以提高医疗相关内容的质量和相关性。
    """
    
    def __init__(
        self, 
        llm: Optional[CustomLLM] = None,
        reranker: Optional[CustomReranker] = None
    ):
        """初始化医疗内容过滤器
        
        Args:
            llm: 大语言模型实例，如果为None则创建默认实例
            reranker: 重排序模型实例，如果为None则创建默认实例
        """
        self.llm = llm or CustomLLM()
        self.reranker = reranker or CustomReranker()
    
    async def filter_by_llm(
        self, 
        query: str, 
        results: List[SearchResult],
        threshold: float = 0.7,
        max_results: int = 10
    ) -> List[SearchResult]:
        """使用大模型过滤医疗相关内容
        
        通过大模型评估每个搜索结果与医疗查询的相关性，
        只保留相关性得分高于阈值的结果。
        
        Args:
            query: 用户查询
            results: 搜索结果列表
            threshold: 医疗相关性阈值，范围0-1
            max_results: 最大返回结果数
            
        Returns:
            过滤后的搜索结果列表
        """
        if not results:
            return []
            
        # 构建提示，要求LLM评估每个结果的医疗相关性
        prompt = f"""
作为医疗内容评估专家，请评估以下搜索结果与医疗查询的相关性。
查询: "{query}"

请为每个结果评分（0-1分），其中:
- 0分表示完全不相关或非医疗内容
- 0.5分表示部分相关或包含一些医疗信息
- 1分表示高度相关且包含有价值的医疗信息

请以JSON格式返回评分，格式如下:
{{
  "results": [
    {{"id": 0, "score": 0.8, "reason": "相关性原因简述"}},
    {{"id": 1, "score": 0.3, "reason": "相关性原因简述"}},
    ...
  ]
}}

搜索结果:
"""
        
        # 添加搜索结果到提示
        for i, result in enumerate(results[:max_results * 2]):  # 处理更多结果，然后筛选
            title = result.metadata.get("title", "无标题")
            prompt += f"\n[{i}] 标题: {title}\n内容: {result.content[:300]}...\n"
            
        # 调用LLM评估结果
        response = await self.llm.generate(prompt)
        
        # 解析LLM响应
        try:
            # 尝试直接解析JSON
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].strip()
                
            data = json.loads(json_str)
            
            # 提取评分结果
            scored_results = []
            for item in data.get("results", []):
                result_id = item.get("id")
                score = item.get("score", 0)
                
                if result_id is not None and result_id < len(results) and score >= threshold:
                    result = results[result_id]
                    # 更新结果的分数
                    result.score = score
                    scored_results.append(result)
                    
            # 按分数排序
            scored_results.sort(key=lambda x: x.score, reverse=True)
            return scored_results[:max_results]
            
        except (json.JSONDecodeError, KeyError) as e:
            # 如果解析失败，返回原始结果
            print(f"解析LLM响应时出错: {str(e)}")
            return results[:max_results]
    
    async def filter_by_reranker(
        self, 
        query: str, 
        results: List[SearchResult],
        medical_context: str = "医疗健康相关信息",
        top_k: int = 10
    ) -> List[SearchResult]:
        """使用重排序模型过滤医疗相关内容
        
        通过重排序模型对搜索结果进行重新排序，提高医疗相关内容的排名。
        
        Args:
            query: 用户查询
            results: 搜索结果列表
            medical_context: 医疗上下文描述，用于增强查询
            top_k: 返回的结果数量
            
        Returns:
            重排序后的搜索结果列表
        """
        if not results:
            return []
            
        # 增强查询，添加医疗上下文
        enhanced_query = f"{query} {medical_context}"
        
        # 提取文档内容
        docs = [result.content for result in results]
        
        # 使用重排序模型重排序
        reranked_results = await self.reranker.rerank_search_results(
            query=enhanced_query,
            results=results,
            k=top_k
        )
        
        return reranked_results
    
    async def hybrid_filter(
        self, 
        query: str, 
        results: List[SearchResult],
        llm_threshold: float = 0.7,
        rerank_weight: float = 0.7,
        llm_weight: float = 0.3,
        max_results: int = 10
    ) -> List[SearchResult]:
        """混合过滤方法，结合LLM和重排序模型
        
        同时使用LLM和重排序模型对结果进行评估，
        然后根据权重合并两种方法的结果。
        
        Args:
            query: 用户查询
            results: 搜索结果列表
            llm_threshold: LLM过滤的医疗相关性阈值
            rerank_weight: 重排序模型分数权重
            llm_weight: LLM分数权重
            max_results: 最大返回结果数
            
        Returns:
            混合过滤后的搜索结果列表
        """
        if not results:
            return []
            
        # 并行执行两种过滤方法
        llm_results, rerank_results = await asyncio.gather(
            self.filter_by_llm(query, results, llm_threshold, max_results * 2),
            self.filter_by_reranker(query, results, top_k=max_results * 2)
        )
        
        # 创建结果ID到结果的映射
        result_map = {}
        
        # 处理LLM结果
        for result in llm_results:
            result_id = id(result)
            if result_id not in result_map:
                result_map[result_id] = {
                    "result": result,
                    "llm_score": result.score,
                    "rerank_score": 0.0,
                    "final_score": 0.0
                }
            else:
                result_map[result_id]["llm_score"] = result.score
                
        # 处理重排序结果
        for result in rerank_results:
            result_id = id(result)
            if result_id not in result_map:
                result_map[result_id] = {
                    "result": result,
                    "llm_score": 0.0,
                    "rerank_score": result.score,
                    "final_score": 0.0
                }
            else:
                result_map[result_id]["rerank_score"] = result.score
                
        # 计算最终分数
        for item in result_map.values():
            item["final_score"] = (
                item["llm_score"] * llm_weight + 
                item["rerank_score"] * rerank_weight
            )
            # 更新结果分数
            item["result"].score = item["final_score"]
            
        # 按最终分数排序
        final_results = [item["result"] for item in result_map.values()]
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        return final_results[:max_results] 