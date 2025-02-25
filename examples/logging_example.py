#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志模块使用示例
展示如何使用各种日志类进行日志记录
"""

import os
import sys
import time
import asyncio
import threading
import random
from datetime import datetime
from typing import List, Dict, Any

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入日志模块
from module.logging import (
    AsyncLogger, 
    AsyncLogHandler,
    DailyRotatingLogger,
    SizeRotatingLogger,
    ConcurrentLogger,
    ThreadSafeHandler,
    ColoredConsoleLogger,
    SimpleFileLogger
)
from module.logging.async_logger import LogLevel
from module.logging.console_logger import ColoredConsoleHandler
from module.logging.file_logger import FileHandler


def test_colored_console_logger():
    """测试彩色控制台日志记录器"""
    print("\n=== 测试彩色控制台日志记录器 ===")
    
    # 创建彩色控制台日志记录器
    logger = ColoredConsoleLogger(
        name="彩色控制台日志",
        level="DEBUG",
        use_color=True
    )
    
    # 记录不同级别的日志
    logger.debug("这是一条调试信息")
    logger.info("这是一条普通信息")
    logger.warning("这是一条警告信息")
    logger.error("这是一条错误信息")
    logger.critical("这是一条严重错误信息")
    
    try:
        # 制造一个异常
        1 / 0
    except Exception as e:
        logger.exception("捕获到异常")
    
    # 关闭日志记录器
    logger.close()


def test_simple_file_logger():
    """测试简单文件日志记录器"""
    print("\n=== 测试简单文件日志记录器 ===")
    
    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建简单文件日志记录器
    log_file = os.path.join(log_dir, 'simple_file.log')
    logger = SimpleFileLogger(
        name="简单文件日志",
        log_file=log_file,
        level="INFO"
    )
    
    # 记录不同级别的日志
    logger.debug("这是一条调试信息 - 不会被记录")
    logger.info("这是一条普通信息")
    logger.warning("这是一条警告信息")
    logger.error("这是一条错误信息")
    logger.critical("这是一条严重错误信息")
    
    try:
        # 制造一个异常
        1 / 0
    except Exception as e:
        logger.exception("捕获到异常")
    
    # 关闭日志记录器
    logger.close()
    
    print(f"日志已写入文件: {log_file}")


def test_daily_rotating_logger():
    """测试每日轮转日志记录器"""
    print("\n=== 测试每日轮转日志记录器 ===")
    
    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建每日轮转日志记录器
    logger = DailyRotatingLogger(
        log_dir=log_dir,
        base_filename="daily_rotating",
        level="INFO",
        backup_count=7
    )
    
    # 记录不同级别的日志
    logger.debug("这是一条调试信息 - 不会被记录")
    logger.info("这是一条普通信息")
    logger.warning("这是一条警告信息")
    logger.error("这是一条错误信息")
    logger.critical("这是一条严重错误信息")
    
    try:
        # 制造一个异常
        1 / 0
    except Exception as e:
        logger.exception("捕获到异常")
    
    # 关闭日志记录器
    logger.close()
    
    print(f"日志已写入目录: {log_dir}")


def test_size_rotating_logger():
    """测试大小轮转日志记录器"""
    print("\n=== 测试大小轮转日志记录器 ===")
    
    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建大小轮转日志记录器 (最大1KB，便于测试)
    logger = SizeRotatingLogger(
        log_dir=log_dir,
        base_filename="size_rotating",
        level="INFO",
        max_bytes=1024,  # 1KB
        backup_count=5
    )
    
    # 记录多条日志以触发轮转
    for i in range(100):
        logger.info(f"这是第 {i+1} 条信息，用于测试大小轮转")
    
    # 记录不同级别的日志
    logger.debug("这是一条调试信息 - 不会被记录")
    logger.info("这是一条普通信息")
    logger.warning("这是一条警告信息")
    logger.error("这是一条错误信息")
    logger.critical("这是一条严重错误信息")
    
    # 关闭日志记录器
    logger.close()
    
    print(f"日志已写入目录: {log_dir}")


def test_concurrent_logger():
    """测试并发日志记录器"""
    print("\n=== 测试并发日志记录器 ===")
    
    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建并发日志记录器
    log_file = os.path.join(log_dir, 'concurrent.log')
    logger = ConcurrentLogger(
        name="并发日志",
        log_file=log_file,
        console=True,
        level="INFO"
    )
    
    # 创建多个线程记录日志
    def log_thread(thread_id):
        for i in range(10):
            logger.info(f"线程 {thread_id} - 消息 {i+1}")
            time.sleep(0.01)
    
    threads = []
    for i in range(5):
        thread = threading.Thread(target=log_thread, args=(i+1,))
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    # 记录不同级别的日志
    logger.debug("这是一条调试信息 - 不会被记录")
    logger.info("这是一条普通信息")
    logger.warning("这是一条警告信息")
    logger.error("这是一条错误信息")
    logger.critical("这是一条严重错误信息")
    
    # 关闭日志记录器
    logger.close()
    
    print(f"日志已写入文件: {log_file}")


class CustomConsoleHandler(AsyncLogHandler):
    """自定义控制台处理器"""
    
    async def emit(self, record):
        """输出日志记录"""
        timestamp = datetime.fromtimestamp(record.get('timestamp', datetime.now().timestamp()))
        time_str = timestamp.strftime('%H:%M:%S.%f')[:-3]
        level = record.get('level', 'INFO')
        message = f"[{time_str}] [{level}] {record['message']}"
        print(message)
    
    async def close(self):
        """关闭处理器"""
        pass


async def simulate_api_calls(logger: AsyncLogger, num_calls: int = 5) -> None:
    """模拟API调用并记录日志
    
    Args:
        logger: 异步日志记录器
        num_calls: 调用次数
    """
    api_endpoints = ["用户服务", "订单服务", "支付服务", "库存服务", "物流服务"]
    
    for i in range(num_calls):
        endpoint = random.choice(api_endpoints)
        logger.info(f"开始调用API: {endpoint}")
        
        # 模拟API调用
        delay = random.uniform(0.1, 1.0)
        start_time = time.time()
        
        try:
            # 模拟API调用
            await asyncio.sleep(delay)
            
            # 随机生成成功或失败
            success = random.random() > 0.3
            
            if success:
                logger.info(f"API调用成功: {endpoint}, 耗时: {time.time() - start_time:.2f}秒")
            else:
                error_code = random.randint(400, 500)
                logger.error(f"API调用失败: {endpoint}, 错误码: {error_code}")
                
                # 对于严重错误，记录更详细的信息
                if error_code >= 500:
                    logger.critical(f"服务器错误: {endpoint} 返回 {error_code}")
        except Exception as e:
            logger.exception(f"API调用异常: {endpoint}", exc_info=e)


async def process_data(logger: AsyncLogger, data_items: List[Dict[str, Any]]) -> None:
    """处理数据并记录日志
    
    Args:
        logger: 异步日志记录器
        data_items: 数据项列表
    """
    logger.info(f"开始处理 {len(data_items)} 条数据")
    
    for i, item in enumerate(data_items):
        try:
            # 记录处理进度
            if (i + 1) % 10 == 0 or i == 0 or i == len(data_items) - 1:
                logger.debug(f"处理进度: {i+1}/{len(data_items)}")
            
            # 模拟数据处理
            await asyncio.sleep(0.05)
            
            # 检查数据有效性
            if "id" not in item:
                logger.warning(f"数据缺少ID字段: {item}")
                continue
                
            if "value" not in item or not isinstance(item["value"], (int, float)):
                logger.warning(f"数据值无效: {item}")
                continue
            
            # 处理数据
            result = item["value"] * 2
            
            # 记录处理结果
            if result > 100:
                logger.info(f"数据 {item['id']} 处理结果超过阈值: {result}")
        except Exception as e:
            logger.exception(f"处理数据 {item.get('id', '未知')} 时出错", exc_info=e)


async def main():
    """主函数"""
    # 测试彩色控制台日志记录器
    test_colored_console_logger()
    
    # 测试简单文件日志记录器
    test_simple_file_logger()
    
    # 测试每日轮转日志记录器
    test_daily_rotating_logger()
    
    # 测试大小轮转日志记录器
    test_size_rotating_logger()
    
    # 测试并发日志记录器
    test_concurrent_logger()
    
    # 创建日志目录
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建异步日志记录器
    logger = AsyncLogger(module_name="examples.logging_example")
    
    try:
        # 添加彩色控制台处理器
        console_handler = ColoredConsoleHandler(use_color=True)
        logger.add_handler(console_handler)
        
        # 添加文件处理器
        log_file = os.path.join(log_dir, "async_log_example.log")
        file_handler = FileHandler(log_file)
        logger.add_handler(file_handler)
        
        # 启动日志记录器
        await logger.start()
        
        # 记录应用启动信息
        logger.info("应用启动")
        
        # 生成测试数据
        test_data = [
            {"id": i, "value": random.uniform(0, 200)} 
            for i in range(50)
        ]
        
        # 随机将一些数据设为无效
        for i in range(5):
            idx = random.randint(0, len(test_data) - 1)
            if random.random() > 0.5:
                del test_data[idx]["id"]
            else:
                test_data[idx]["value"] = "无效值"
        
        # 执行模拟任务
        await asyncio.gather(
            simulate_api_calls(logger, num_calls=10),
            process_data(logger, test_data)
        )
        
        # 记录应用结束信息
        logger.info("应用正常结束")
    except Exception as e:
        # 记录未捕获的异常
        logger.exception("应用发生未捕获的异常", exc_info=e)
    finally:
        # 确保日志记录器被正确关闭
        await logger.stop()
        print(f"日志已保存到: {log_file}")


if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main()) 