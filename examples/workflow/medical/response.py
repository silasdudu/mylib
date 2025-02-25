#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
医疗响应选择模块

提供医疗响应生成和选择功能
"""
from typing import List, Dict, Any, Optional, Union


class MedicalResponseSelector:
    """医疗响应选择器
    
    根据不同类型的医疗查询生成合适的响应
    """
    
    def __init__(self, llm):
        """初始化响应选择器
        
        Args:
            llm: 大语言模型实例
        """
        self.llm = llm
    
    async def generate_factual_response(self, query: str, contexts: List[str]) -> str:
        """生成事实型医疗查询的响应
        
        Args:
            query: 用户查询
            contexts: 检索到的相关上下文列表
            
        Returns:
            生成的响应
        """
        # 在实际应用中，这里会使用LLM根据上下文生成响应
        # 这里使用简化的逻辑进行演示
        
        if not contexts:
            return "抱歉，我没有找到与您问题相关的信息。请尝试用不同的方式提问。"
        
        # 简单示例：根据上下文拼接响应
        response = "根据医学资料，"
        
        # 添加来源信息
        sources = []
        for context in contexts[:2]:  # 只使用前两个上下文
            if "来源:" in context:
                source = context.split("来源:")[1].split("\n")[0].strip()
                if source and source not in sources:
                    sources.append(source)
            
            # 从上下文中提取一些内容
            content = context.split("\n")[-1] if "\n" in context else context
            response += f"{content} "
        
        # 添加来源引用
        if sources:
            response += "\n\n信息来源: " + ", ".join(sources)
        
        return response
    
    async def generate_personal_response(self, query: str) -> str:
        """生成个人医疗咨询的响应
        
        Args:
            query: 用户查询
            
        Returns:
            生成的响应
        """
        # 在实际应用中，这里会使用LLM生成个性化响应
        # 这里使用简化的逻辑进行演示
        
        disclaimer = "请注意，我提供的信息仅供参考，不能替代专业医生的诊断和建议。如果您有严重的健康问题，请立即就医。"
        
        if "头痛" in query:
            return f"头痛可能由多种原因引起，包括压力、疲劳、脱水、眼睛疲劳或偏头痛等。建议您保持充分休息，多喝水，并考虑服用非处方止痛药。如果头痛持续或加重，请咨询医生。\n\n{disclaimer}"
        
        if "失眠" in query:
            return f"失眠可能与压力、焦虑、不规律的睡眠习惯或某些药物有关。建议您尝试建立规律的睡眠时间表，睡前避免咖啡因和电子设备，创造舒适的睡眠环境。如果问题持续，请咨询医生。\n\n{disclaimer}"
        
        # 默认响应
        return f"感谢您分享您的健康状况。根据您提供的信息，我建议您密切关注症状变化，保持良好的生活习惯，如果症状持续或加重，请及时就医咨询专业医生的建议。\n\n{disclaimer}"
    
    async def generate_general_response(self, query: str) -> str:
        """生成一般医疗问题的响应
        
        Args:
            query: 用户查询
            
        Returns:
            生成的响应
        """
        # 在实际应用中，这里会使用LLM生成响应
        # 这里使用简化的逻辑进行演示
        
        if "健康" in query or "保健" in query:
            return "保持健康的关键包括均衡饮食、规律运动、充足睡眠、减轻压力和定期体检。建议每天至少进行30分钟中等强度的运动，摄入丰富的蔬菜水果，限制加工食品和糖分的摄入。"
        
        if "饮食" in query or "营养" in query:
            return "健康饮食应包括多样化的食物，如全谷物、蛋白质、健康脂肪、蔬菜和水果。建议限制盐、糖和饱和脂肪的摄入，增加膳食纤维的摄入。每天饮用足够的水也很重要。"
        
        if "运动" in query or "锻炼" in query:
            return "规律运动对身心健康都有益处。建议每周进行至少150分钟的中等强度有氧运动，如快走、游泳或骑自行车，并每周进行两次以上的肌肉强化训练。开始新的运动计划前，最好先咨询医生。"
        
        if "睡眠" in query:
            return "良好的睡眠对健康至关重要。成年人每晚应睡7-9小时。建立规律的睡眠时间表，创造舒适的睡眠环境，睡前避免咖啡因、酒精和电子设备，有助于提高睡眠质量。"
        
        # 默认响应
        return "保持健康的生活方式对预防疾病和提高生活质量非常重要。均衡饮食、规律运动、充足睡眠、减轻压力和定期体检是维护健康的基本要素。如果您有特定的健康问题，建议咨询专业医生的建议。" 