#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
异步日志示例，展示如何使用AsyncLogger类
"""
import os
import asyncio
import random
import time

# 导入异步日志类
from module.logging import AsyncLogger
from module.logging.console_logger import ColoredConsoleHandler
from module.logging.file_logger import FileHandler


async def simulate_medical_operations(logger: AsyncLogger) -> None:
    """模拟医疗操作并记录日志
    
    Args:
        logger: 异步日志记录器
    """
    operations = ["患者登记", "体温测量", "血压测量", "医生问诊", "开具处方", "药品发放"]
    patients = ["张三", "李四", "王五", "赵六", "钱七"]
    
    # 模拟多个医疗操作
    for _ in range(10):
        patient = random.choice(patients)
        operation = random.choice(operations)
        
        # 记录操作开始
        logger.info(f"开始为患者 {patient} 执行 {operation}")
        
        # 模拟操作耗时
        delay = random.uniform(0.1, 1.0)
        await asyncio.sleep(delay)
        
        # 随机模拟成功或异常情况
        if random.random() > 0.8:  # 20%概率出现问题
            if random.random() > 0.5:  # 一半是警告，一半是错误
                logger.warning(f"患者 {patient} 的 {operation} 操作出现异常情况，需要注意")
            else:
                try:
                    # 模拟异常
                    raise ValueError(f"患者 {patient} 的 {operation} 操作失败")
                except Exception as e:
                    logger.exception(f"患者 {patient} 的 {operation} 操作发生错误", exc_info=e)
        else:
            # 记录操作成功
            logger.info(f"患者 {patient} 的 {operation} 操作完成，耗时: {delay:.2f}秒")


async def main():
    """主函数"""
    # 创建日志目录
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建异步日志记录器
    logger = AsyncLogger(module_name="医疗系统")
    
    try:
        # 添加彩色控制台处理器
        console_handler = ColoredConsoleHandler(use_color=True)
        logger.add_handler(console_handler)
        
        # 添加文件处理器
        log_file = os.path.join(log_dir, "medical_system.log")
        file_handler = FileHandler(log_file)
        logger.add_handler(file_handler)
        
        # 启动日志记录器
        await logger.start()
        
        # 记录系统启动信息
        logger.info("医疗系统启动")
        
        # 执行模拟医疗操作
        await simulate_medical_operations(logger)
        
        # 记录系统关闭信息
        logger.info("医疗系统正常关闭")
    except Exception as e:
        # 记录未捕获的异常
        logger.exception("医疗系统发生未捕获的异常", exc_info=e)
    finally:
        # 确保日志记录器被正确关闭
        await logger.stop()
        print(f"日志已保存到: {log_file}")


if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main()) 