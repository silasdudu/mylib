# 多Agent协作示例

这个示例展示了如何使用基础工具库构建一个多Agent协作系统。示例实现了一个智能客服团队，多个专门的Agent协同工作，处理用户查询。

## 系统架构

系统包含以下几个主要组件：

1. **查询分发Agent (QueryDispatcherAgent)**
   - 接收用户查询
   - 分析查询类型
   - 将查询分发给合适的Agent处理

2. **知识库Agent (KnowledgeBaseAgent)**
   - 负责文档检索
   - 从知识库中找到相关内容
   - 将检索结果传递给生成Agent

3. **回答生成Agent (AnswerGeneratorAgent)**
   - 基于检索结果生成答案
   - 确保答案的相关性和完整性
   - 将生成的答案发送给质检Agent

4. **质量检查Agent (QualityCheckerAgent)**
   - 检查答案质量
   - 评估答案的准确性
   - 提供质量反馈

## 工作流程

1. 用户发送查询到查询分发Agent
2. 查询分发Agent将查询转发给知识库Agent
3. 知识库Agent检索相关文档并交给回答生成Agent
4. 回答生成Agent生成答案并发送给质量检查Agent
5. 质量检查Agent评估答案质量并返回最终结果

## 运行要求

1. Python 3.9+
2. Redis服务器
3. 项目依赖：
```bash
pip install -r requirements.txt
```

## 配置说明

1. Redis配置（默认值）：
```python
host = "localhost"
port = 6379
prefix = "customer_service"
```

2. Agent配置：
```python
config = CollaborativeAgentConfig(
    name="agent_name",
    team_id="customer_service",
    role="role_name",
    capabilities={"capability1", "capability2"}
)
```

## 运行示例

1. 确保Redis服务器正在运行

2. 运行示例程序：
```bash
python customer_service.py
```

## 示例输出

```
INFO     Connected to Redis at redis://localhost:6379/0
INFO     Agent dispatcher registered with capabilities: {'query_dispatch'}
INFO     Agent kb_agent registered with capabilities: {'knowledge_base'}
INFO     Agent generator registered with capabilities: {'answer_generation'}
INFO     Agent qa_agent registered with capabilities: {'quality_check'}
INFO     Received query: 如何重置密码？
INFO     Query dispatched to knowledge base agent
INFO     Retrieved relevant documents
INFO     Generated answer
INFO     Quality check completed: 95% confidence
```

## 扩展建议

1. 添加更多专门的Agent：
   - 情感分析Agent
   - 多语言翻译Agent
   - 意图识别Agent
   - 对话管理Agent

2. 增强Agent能力：
   - 实现更复杂的任务分配策略
   - 添加学习和适应机制
   - 优化协作效率

3. 改进通信机制：
   - 添加消息优先级队列
   - 实现消息重试机制
   - 添加更多消息类型

4. 增加监控和管理：
   - 添加性能监控
   - 实现负载均衡
   - 添加日志分析

## 注意事项

1. 这是一个示例实现，主要用于演示多Agent协作的基本概念
2. 在生产环境中使用时需要考虑：
   - 错误处理和恢复机制
   - 性能优化
   - 安全性考虑
   - 可扩展性设计

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交变更
4. 推送到分支
5. 创建Pull Request

## 许可证

MIT License 