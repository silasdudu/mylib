"""
工作流引擎模块，提供任务编排和流程控制
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, TypeVar

from pydantic import BaseModel

from ..core.exceptions import BaseError

T = TypeVar("T")


class WorkflowStatus(str, Enum):
    """工作流状态"""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class StepStatus(str, Enum):
    """步骤状态"""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class StepResult:
    """步骤执行结果"""
    step_id: str
    status: StepStatus
    output: Any = None
    error: Optional[Exception] = None


class WorkflowStep(BaseModel):
    """工作流步骤"""
    step_id: str
    name: str
    description: Optional[str] = None
    depends_on: Set[str] = set()
    timeout: Optional[float] = None
    retry_count: int = 0
    parameters: Dict[str, Any] = {}


class WorkflowConfig(BaseModel):
    """工作流配置"""
    name: str
    description: Optional[str] = None
    steps: List[WorkflowStep]
    max_concurrent_steps: int = 5
    timeout: Optional[float] = None


class WorkflowError(BaseError):
    """工作流异常"""
    pass


class StepExecutor(ABC):
    """步骤执行器抽象基类"""
    
    @abstractmethod
    async def execute(
        self,
        step: WorkflowStep,
        context: Dict[str, Any]
    ) -> Any:
        """执行步骤"""
        pass


class WorkflowEngine:
    """工作流引擎"""
    
    def __init__(
        self,
        config: WorkflowConfig,
        executor: StepExecutor
    ):
        self.config = config
        self.executor = executor
        self.status = WorkflowStatus.PENDING
        self.step_results: Dict[str, StepResult] = {}
        self.context: Dict[str, Any] = {}
        
    def _can_run_step(self, step: WorkflowStep) -> bool:
        """检查步骤是否可以执行"""
        if not step.depends_on:
            return True
            
        return all(
            self.step_results.get(dep_id, StepResult(dep_id, StepStatus.PENDING)).status == StepStatus.COMPLETED
            for dep_id in step.depends_on
        )
        
    async def execute(self) -> Dict[str, StepResult]:
        """执行工作流"""
        try:
            self.status = WorkflowStatus.RUNNING
            pending_steps = self.config.steps.copy()
            
            while pending_steps:
                # 找出所有可以执行的步骤
                runnable_steps = [
                    step for step in pending_steps
                    if self._can_run_step(step)
                ]
                
                if not runnable_steps:
                    # 如果没有可执行的步骤但还有待执行步骤，说明存在循环依赖
                    raise WorkflowError("检测到循环依赖")
                
                # 并发执行可运行的步骤
                for step in runnable_steps[:self.config.max_concurrent_steps]:
                    try:
                        output = await self.executor.execute(step, self.context)
                        self.step_results[step.step_id] = StepResult(
                            step_id=step.step_id,
                            status=StepStatus.COMPLETED,
                            output=output
                        )
                        # 更新上下文
                        self.context[step.step_id] = output
                    except Exception as e:
                        self.step_results[step.step_id] = StepResult(
                            step_id=step.step_id,
                            status=StepStatus.FAILED,
                            error=e
                        )
                        raise WorkflowError(f"步骤 {step.step_id} 执行失败: {str(e)}")
                    
                    pending_steps.remove(step)
                    
            self.status = WorkflowStatus.COMPLETED
            return self.step_results
        except Exception as e:
            self.status = WorkflowStatus.FAILED
            raise WorkflowError(f"工作流执行失败: {str(e)}")


class WorkflowRegistry:
    """工作流注册器"""
    
    def __init__(self):
        self._workflows: Dict[str, WorkflowConfig] = {}
        
    def register(self, workflow: WorkflowConfig) -> None:
        """注册工作流"""
        self._workflows[workflow.name] = workflow
        
    def get(self, name: str) -> Optional[WorkflowConfig]:
        """获取工作流"""
        return self._workflows.get(name)
    
    def list_workflows(self) -> List[str]:
        """列出所有工作流"""
        return list(self._workflows.keys()) 