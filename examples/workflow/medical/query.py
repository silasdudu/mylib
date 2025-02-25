#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
医疗查询处理模块

提供医疗查询分类和扩展功能
"""
from typing import List, Dict, Any, Optional, Union


class MedicalQueryClassifier:
    """医疗查询分类器
    
    将用户查询分类为不同类型：
    - factual: 事实型医疗查询（如"糖尿病的症状是什么？"）
    - personal: 个人医疗咨询（如"我最近头痛，可能是什么原因？"）
    - general: 一般医疗问题（如"如何保持健康？"）
    - non_medical: 非医疗问题
    """
    
    def __init__(self, llm):
        """初始化查询分类器
        
        Args:
            llm: 大语言模型实例
        """
        self.llm = llm
    
    async def classify(self, query: str) -> str:
        """分类用户查询
        
        Args:
            query: 用户查询文本
            
        Returns:
            查询类型: factual, personal, general, non_medical
        """
        # 在实际应用中，这里会使用LLM进行分类
        # 这里使用简化的规则进行演示
        
        query = query.lower()
        
        # 个人医疗咨询通常包含"我"、"我的"等第一人称
        if any(word in query for word in ["我", "我的", "我们", "我有", "我感觉"]):
            return "personal"
            
        # 事实型医疗查询通常包含特定疾病名称或医学术语
        medical_terms = ["症状", "治疗", "药物", "疾病", "诊断", "预防", "病因", 
                         "糖尿病", "高血压", "癌症", "心脏病", "肺炎", "感冒", "发烧"]
        if any(term in query for term in medical_terms):
            return "factual"
            
        # 一般医疗问题
        general_terms = ["健康", "饮食", "运动", "睡眠", "保健", "营养", "锻炼"]
        if any(term in query for term in general_terms):
            return "general"
            
        # 默认为非医疗问题
        return "non_medical"


class MedicalQueryExpander:
    """医疗查询扩展器
    
    扩展原始查询以提高检索效果
    """
    
    def __init__(self, llm):
        """初始化查询扩展器
        
        Args:
            llm: 大语言模型实例
        """
        self.llm = llm
    
    async def expand(self, query: str) -> str:
        """扩展用户查询
        
        Args:
            query: 原始用户查询
            
        Returns:
            扩展后的查询
        """
        # 在实际应用中，这里会使用LLM进行查询扩展
        # 这里使用简化的规则进行演示
        
        # 简单示例：添加相关术语
        if "糖尿病" in query:
            return f"{query} 血糖 胰岛素 2型糖尿病 糖尿病并发症"
            
        if "高血压" in query:
            return f"{query} 血压 收缩压 舒张压 心血管疾病"
            
        if "感冒" in query or "发烧" in query:
            return f"{query} 流感 病毒感染 发热 咳嗽 喉咙痛"
            
        # 如果没有特定匹配，返回原始查询
        return query 