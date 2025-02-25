#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
医疗应用日志示例

展示如何在医疗应用中使用不同的日志类进行日志记录
"""

import os
import sys
import time
import asyncio
import threading
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# 导入日志模块
from module.logging import (
    AsyncLogger, 
    DailyRotatingLogger,
    SizeRotatingLogger,
    ConcurrentLogger,
    ColoredConsoleLogger,
    SimpleFileLogger
)
from module.logging.console_logger import ColoredConsoleHandler
from module.logging.file_logger import FileHandler


def setup_medical_loggers():
    """设置医疗应用的日志记录器"""
    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建彩色控制台日志记录器 - 用于开发调试
    console_logger = ColoredConsoleLogger(
        name="医疗应用",
        level="DEBUG",
        use_color=True
    )
    
    # 创建每日轮转日志记录器 - 用于生产环境
    daily_logger = DailyRotatingLogger(
        log_dir=log_dir,
        base_filename="medical_app",
        level="INFO",
        backup_count=30  # 保留30天的日志
    )
    
    # 创建大小轮转日志记录器 - 用于限制日志文件大小
    size_logger = SizeRotatingLogger(
        log_dir=log_dir,
        base_filename="medical_errors",
        level="ERROR",
        max_bytes=5 * 1024 * 1024,  # 5MB
        backup_count=10
    )
    
    # 创建并发日志记录器 - 用于多线程环境
    concurrent_logger = ConcurrentLogger(
        name="医疗并发",
        log_dir=log_dir,
        log_file=os.path.join(log_dir, "medical_concurrent.log"),
        level="INFO",
        console=True
    )
    
    # 创建简单文件日志记录器 - 用于特定模块
    search_logger = SimpleFileLogger(
        name="医疗搜索",
        log_file=os.path.join(log_dir, "medical_search.log"),
        level="INFO"
    )
    
    return {
        "console": console_logger,
        "daily": daily_logger,
        "size": size_logger,
        "concurrent": concurrent_logger,
        "search": search_logger
    }


async def setup_async_logger():
    """设置异步日志记录器"""
    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建异步日志记录器
    async_logger = AsyncLogger(module_name="examples.workflow.medical")
    
    # 添加彩色控制台处理器
    console_handler = ColoredConsoleHandler(use_color=True)
    async_logger.add_handler(console_handler)
    
    # 添加文件处理器
    log_file = os.path.join(log_dir, 'medical_async.log')
    file_handler = FileHandler(log_file)
    async_logger.add_handler(file_handler)
    
    # 启动日志记录器
    await async_logger.start()
    
    return async_logger


def simulate_medical_search(loggers):
    """模拟医疗搜索过程"""
    console_logger = loggers["console"]
    search_logger = loggers["search"]
    
    console_logger.info("开始医疗搜索模拟")
    search_logger.info("初始化医疗搜索引擎")
    
    # 模拟搜索过程
    search_terms = ["糖尿病症状", "高血压治疗", "感冒药物", "心脏病预防", "癌症筛查"]
    
    for term in search_terms:
        console_logger.info(f"搜索: {term}")
        search_logger.info(f"执行搜索: {term}")
        
        # 模拟搜索延迟
        time.sleep(0.5)
        
        # 模拟搜索结果
        results_count = hash(term) % 10 + 1  # 1-10个结果
        console_logger.info(f"找到 {results_count} 个结果")
        search_logger.info(f"搜索 '{term}' 找到 {results_count} 个结果")
        
        # 模拟偶尔的错误
        if results_count < 3:
            error_msg = f"搜索 '{term}' 结果过少，可能需要扩展查询"
            console_logger.warning(error_msg)
            search_logger.warning(error_msg)
    
    console_logger.info("医疗搜索模拟完成")
    search_logger.info("搜索引擎会话结束")


def simulate_medical_diagnosis(loggers):
    """模拟医疗诊断过程"""
    console_logger = loggers["console"]
    daily_logger = loggers["daily"]
    size_logger = loggers["size"]
    
    console_logger.info("开始医疗诊断模拟")
    daily_logger.info("初始化医疗诊断系统")
    
    # 模拟诊断过程
    symptoms = [
        "头痛、发热、咳嗽",
        "胸痛、呼吸困难",
        "腹痛、恶心、呕吐",
        "皮疹、瘙痒",
        "关节疼痛、肿胀"
    ]
    
    for symptom in symptoms:
        console_logger.info(f"分析症状: {symptom}")
        daily_logger.info(f"诊断症状组合: {symptom}")
        
        # 模拟诊断延迟
        time.sleep(0.7)
        
        # 模拟诊断结果
        if "胸痛" in symptom:
            error_msg = f"检测到严重症状: {symptom}，建议立即就医"
            console_logger.error(error_msg)
            daily_logger.error(error_msg)
            size_logger.error(error_msg)
        elif "发热" in symptom:
            warning_msg = f"检测到常见症状: {symptom}，可能是感染"
            console_logger.warning(warning_msg)
            daily_logger.warning(warning_msg)
        else:
            console_logger.info(f"分析完成: {symptom}")
            daily_logger.info(f"症状 '{symptom}' 分析完成")
    
    console_logger.info("医疗诊断模拟完成")
    daily_logger.info("诊断系统会话结束")


def simulate_concurrent_patient_records(loggers):
    """模拟并发患者记录处理"""
    concurrent_logger = loggers["concurrent"]
    
    concurrent_logger.info("开始并发患者记录处理模拟")
    
    # 模拟多线程处理患者记录
    def process_patient(patient_id):
        for i in range(5):
            concurrent_logger.info(f"处理患者 {patient_id} 的记录 {i+1}")
            time.sleep(0.1)
            
            # 模拟偶尔的错误
            if i == 2 and patient_id % 3 == 0:
                concurrent_logger.error(f"处理患者 {patient_id} 记录时出错: 数据不完整")
    
    # 创建多个线程
    threads = []
    for i in range(5):
        thread = threading.Thread(target=process_patient, args=(i+1,))
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    concurrent_logger.info("并发患者记录处理模拟完成")


async def simulate_async_medical_api(async_logger):
    """模拟异步医疗API调用"""
    await async_logger.info("开始异步医疗API调用模拟")
    
    # 模拟多个异步API调用
    async def call_api(api_name, delay):
        await async_logger.info(f"调用API: {api_name}")
        await asyncio.sleep(delay)
        await async_logger.info(f"API {api_name} 调用完成")
        
        # 模拟偶尔的错误
        if delay > 0.8:
            await async_logger.error(f"API {api_name} 响应时间过长: {delay}秒")
    
    # 创建多个异步任务
    tasks = [
        call_api("患者信息", 0.5),
        call_api("药物数据库", 0.7),
        call_api("医疗影像", 0.9),
        call_api("实验室结果", 0.3),
        call_api("医疗保险", 0.6)
    ]
    
    # 并发执行所有任务
    await asyncio.gather(*tasks)
    
    await async_logger.info("异步医疗API调用模拟完成")


async def main():
    """主函数"""
    print("\n=== 医疗应用日志示例 ===")
    
    # 设置日志记录器
    loggers = setup_medical_loggers()
    async_logger = await setup_async_logger()
    
    try:
        # 记录应用启动信息
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for name, logger in loggers.items():
            logger.info(f"医疗应用日志示例启动 - {start_time}")
        await async_logger.info(f"医疗应用日志示例启动 - {start_time}")
        
        # 模拟医疗搜索
        simulate_medical_search(loggers)
        
        # 模拟医疗诊断
        simulate_medical_diagnosis(loggers)
        
        # 模拟并发患者记录处理
        simulate_concurrent_patient_records(loggers)
        
        # 模拟异步医疗API调用
        await simulate_async_medical_api(async_logger)
        
        # 记录应用结束信息
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for name, logger in loggers.items():
            logger.info(f"医疗应用日志示例结束 - {end_time}")
        await async_logger.info(f"医疗应用日志示例结束 - {end_time}")
        
    except Exception as e:
        # 记录异常信息
        error_msg = f"应用程序运行时出错: {str(e)}"
        for name, logger in loggers.items():
            if hasattr(logger, 'exception'):
                logger.exception(error_msg)
            else:
                logger.error(error_msg)
        await async_logger.exception(error_msg)
        
    finally:
        # 关闭所有日志记录器
        for name, logger in loggers.items():
            logger.close()
        await async_logger.stop()
        
        print("\n日志文件已保存到 'logs' 目录")


if __name__ == "__main__":
    asyncio.run(main()) 