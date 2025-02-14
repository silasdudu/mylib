# 基础工具库

这是一个高度模块化、解耦合的工具库，提供统一的基础类、接口和工具，用于大模型相关应用开发。

## 项目结构

```
base/
├── core/                   # 核心模块
│   ├── config.py          # 配置管理
│   ├── logging.py         # 日志模块
│   ├── scheduler.py       # 任务调度器
│   └── exceptions.py      # 异常管理
├── model/                 # 模型相关模块
│   └── interface.py       # 大模型接口
├── data/                  # 数据处理模块
│   └── dataset.py         # 数据集接口
├── agent/                 # Agent模块
│   └── base.py           # Agent基类
└── workflow/              # 工作流模块
    └── engine.py         # 工作流引擎
```

## 主要功能

1. 核心模块
   - 配置管理：支持多种格式配置文件的加载和动态更新
   - 日志系统：异步日志记录，支持多种输出方式
   - 任务调度：支持协程、线程池的混合并发
   - 异常管理：统一的异常处理机制

2. 模型接口
   - 统一的大模型调用接口
   - 支持流式输出
   - 内置评测工具

3. 数据处理
   - 统一的数据集接口
   - 异步数据加载
   - 数据缓存机制

4. Agent系统
   - 灵活的Agent基类
   - 工具注册和管理
   - 状态管理

5. 工作流引擎
   - 任务编排
   - 并发执行
   - 依赖管理

## 安装

使用Poetry安装依赖：

```bash
poetry install
```

## 使用示例

1. 配置管理

```python
from base.core.config import ConfigManager, FileConfigSource

# 创建配置管理器
config_source = FileConfigSource("config.yaml")
config_manager = ConfigManager(config_source)

# 加载配置
config = await config_manager.load()
```

2. 日志记录

```python
from base.core.logging import AsyncLogger, FileLogHandler, ConsoleLogHandler

# 创建日志记录器
logger = AsyncLogger()
logger.add_handler(FileLogHandler("app.log"))
logger.add_handler(ConsoleLogHandler())

# 记录日志
await logger.log("INFO", "应用启动")
```

3. 任务调度

```python
from base.core.scheduler import Task, AsyncTaskExecutor

# 创建任务
class MyTask(Task):
    async def run(self):
        # 实现任务逻辑
        pass

# 创建执行器
executor = AsyncTaskExecutor()
await executor.submit(MyTask("task1"))
```

4. 大模型调用

```python
from base.model.interface import LargeModel, ModelConfig

# 实现模型接口
class MyModel(LargeModel):
    async def generate(self, prompt: str, config: ModelConfig = None):
        # 实现生成逻辑
        pass

# 使用模型
model = MyModel()
response = await model.generate("你好")
```

5. 工作流编排

```python
from base.workflow.engine import WorkflowEngine, WorkflowConfig, WorkflowStep

# 创建工作流配置
config = WorkflowConfig(
    name="数据处理流程",
    steps=[
        WorkflowStep(
            step_id="step1",
            name="数据加载"
        ),
        WorkflowStep(
            step_id="step2",
            name="数据处理",
            depends_on={"step1"}
        )
    ]
)

# 执行工作流
engine = WorkflowEngine(config, executor)
results = await engine.execute()
```

## 开发指南

1. 代码风格
   - 使用Black进行代码格式化
   - 使用isort进行导入排序
   - 使用mypy进行类型检查

2. 测试
   - 使用pytest编写单元测试
   - 使用pytest-asyncio测试异步代码

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交变更
4. 推送到分支
5. 创建Pull Request

## 许可证

MIT License