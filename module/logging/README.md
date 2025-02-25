# 异步日志模块使用指南

本模块提供了异步日志记录功能，适用于需要高性能日志记录的应用场景。

## 主要功能

- **异步日志记录**：支持异步记录日志，不会阻塞主线程
- **多处理器支持**：可以同时输出到控制台和文件
- **彩色输出**：支持在控制台中使用彩色输出
- **详细上下文**：自动记录模块、函数、行号等上下文信息
- **异常处理**：支持记录异常信息和堆栈跟踪

## 使用方法

### 基本用法

```python
import asyncio
import os
from module.logging import AsyncLogger
from module.logging.console_logger import ColoredConsoleHandler
from module.logging.file_logger import FileHandler

async def main():
    # 创建日志目录
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建异步日志记录器
    logger = AsyncLogger(module_name="my_module")
    
    try:
        # 添加彩色控制台处理器
        console_handler = ColoredConsoleHandler(use_color=True)
        logger.add_handler(console_handler)
        
        # 添加文件处理器
        log_file = os.path.join(log_dir, "app.log")
        file_handler = FileHandler(log_file)
        logger.add_handler(file_handler)
        
        # 启动日志记录器
        await logger.start()
        
        # 记录不同级别的日志
        logger.debug("这是一条调试信息")
        logger.info("这是一条普通信息")
        logger.warning("这是一条警告信息")
        logger.error("这是一条错误信息")
        logger.critical("这是一条严重错误信息")
        
        # 记录异常
        try:
            # 模拟异常
            1 / 0
        except Exception as e:
            logger.exception("捕获到异常", exc_info=e)
            
    except Exception as e:
        # 记录未捕获的异常
        logger.exception("应用发生未捕获的异常", exc_info=e)
    finally:
        # 确保日志记录器被正确关闭
        await logger.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### 在医疗应用中的使用示例

```python
import asyncio
import os
from module.logging import AsyncLogger
from module.logging.console_logger import ColoredConsoleHandler
from module.logging.file_logger import FileHandler

# 创建异步日志记录器
async_logger = AsyncLogger(module_name="medical_system")

async def setup_async_logger():
    """设置异步日志记录器"""
    # 创建日志目录
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 添加彩色控制台处理器
    console_handler = ColoredConsoleHandler(use_color=True)
    async_logger.add_handler(console_handler)
    
    # 添加文件处理器
    log_file = os.path.join(log_dir, "medical_system.log")
    file_handler = FileHandler(log_file)
    async_logger.add_handler(file_handler)
    
    # 启动日志记录器
    await async_logger.start()
    return async_logger

async def diagnose_patient(patient_id, symptoms):
    """诊断患者"""
    async_logger.info(f"开始诊断患者 {patient_id}")
    
    try:
        # 执行诊断逻辑
        if "胸痛" in symptoms:
            async_logger.warning(f"患者 {patient_id} 报告胸痛症状，需要紧急处理")
            
        # 记录诊断结果
        diagnosis = "感冒"
        async_logger.info(f"患者 {patient_id} 诊断结果: {diagnosis}")
        
    except Exception as e:
        error_msg = f"诊断患者 {patient_id} 时出错: {str(e)}"
        async_logger.exception(error_msg, exc_info=e)

async def main():
    # 初始化日志
    await setup_async_logger()
    
    try:
        async_logger.info("医疗系统启动")
        
        # 模拟诊断患者
        await diagnose_patient("P001", ["发热", "咳嗽"])
        await diagnose_patient("P002", ["胸痛", "呼吸困难"])
        
        async_logger.info("医疗系统正常关闭")
    except Exception as e:
        async_logger.exception("系统运行时出现未捕获的异常", exc_info=e)
    finally:
        # 确保日志记录器被正确关闭
        await async_logger.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## 最佳实践

1. **始终在应用启动时初始化日志记录器**：
   ```python
   logger = AsyncLogger(module_name="my_module")
   await logger.start()
   ```

2. **添加多个处理器以满足不同需求**：
   ```python
   # 控制台输出（开发调试）
   console_handler = ColoredConsoleHandler(use_color=True)
   logger.add_handler(console_handler)
   
   # 文件输出（持久化记录）
   file_handler = FileHandler("app.log")
   logger.add_handler(file_handler)
   ```

3. **使用适当的日志级别**：
   - `debug`：详细的调试信息
   - `info`：一般信息
   - `warning`：警告信息
   - `error`：错误信息
   - `critical`：严重错误信息
   - `exception`：异常信息（包含堆栈跟踪）

4. **在异常处理中使用日志**：
   ```python
   try:
       # 可能引发异常的代码
       result = process_data(data)
   except Exception as e:
       logger.exception("处理数据时出错", exc_info=e)
   ```

5. **在应用退出前关闭日志记录器**：
   ```python
   try:
       # 应用代码
   finally:
       await logger.stop()
   ```

6. **使用模块名称提供更好的上下文**：
   ```python
   logger = AsyncLogger(module_name="app.services.user")
   ```

## 完整示例

请参考 `examples/async_logger_demo.py` 和 `examples/workflow/medical_consultation.py` 获取完整的使用示例。 