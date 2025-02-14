"""
任务调度器与异步执行器，支持协程、线程池及多进程混合并发
"""
import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union

T = TypeVar("T")


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    status: TaskStatus
    progress: float = 0.0
    result: Any = None
    error: Optional[Exception] = None


class Task(ABC):
    """任务抽象基类"""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.status = TaskStatus.PENDING
        self.progress = 0.0
        self.result: Any = None
        self.error: Optional[Exception] = None
        
    @abstractmethod
    async def run(self) -> Any:
        """运行任务"""
        pass
    
    @abstractmethod
    async def cancel(self) -> None:
        """取消任务"""
        pass
    
    def update_progress(self, progress: float) -> None:
        """更新进度"""
        self.progress = progress


class TaskExecutor(ABC):
    """任务执行器抽象基类"""
    
    @abstractmethod
    async def submit(self, task: Task) -> None:
        """提交任务"""
        pass
    
    @abstractmethod
    async def cancel(self, task_id: str) -> None:
        """取消任务"""
        pass
    
    @abstractmethod
    def get_task_info(self, task_id: str) -> Optional[TaskInfo]:
        """获取任务信息"""
        pass


class AsyncTaskExecutor(TaskExecutor):
    """异步任务执行器"""
    
    def __init__(self, max_workers: int = 10):
        self._tasks: Dict[str, Task] = {}
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._loop = asyncio.get_event_loop()
        
    async def submit(self, task: Task) -> None:
        """提交异步任务"""
        self._tasks[task.task_id] = task
        task.status = TaskStatus.RUNNING
        
        try:
            task.result = await task.run()
            task.status = TaskStatus.COMPLETED
        except Exception as e:
            task.error = e
            task.status = TaskStatus.FAILED
            raise
            
    async def cancel(self, task_id: str) -> None:
        """取消任务"""
        if task := self._tasks.get(task_id):
            await task.cancel()
            task.status = TaskStatus.CANCELLED
            
    def get_task_info(self, task_id: str) -> Optional[TaskInfo]:
        """获取任务信息"""
        if task := self._tasks.get(task_id):
            return TaskInfo(
                task_id=task.task_id,
                status=task.status,
                progress=task.progress,
                result=task.result,
                error=task.error
            )
        return None


class TaskScheduler:
    """任务调度器"""
    
    def __init__(self, executor: TaskExecutor):
        self._executor = executor
        self._scheduled_tasks: List[Task] = []
        
    async def schedule(self, task: Task, delay: Optional[float] = None) -> None:
        """调度任务"""
        if delay is not None:
            await asyncio.sleep(delay)
        await self._executor.submit(task)
        
    async def cancel_all(self) -> None:
        """取消所有任务"""
        for task in self._scheduled_tasks:
            await self._executor.cancel(task.task_id) 