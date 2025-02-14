"""
追踪模块，提供OpenTelemetry集成
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class TracingConfig(BaseModel):
    """追踪配置"""
    service_name: str
    environment: str = "production"
    sampler_rate: float = 1.0
    jaeger_host: str = "localhost"
    jaeger_port: int = 6831
    enable_console: bool = False
    enable_jaeger: bool = True
    tags: Dict[str, str] = {}


class Span:
    """追踪跨度"""
    
    def __init__(
        self,
        name: str,
        context: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.context = context or {}
        self._span = None
    
    async def __aenter__(self):
        """进入跨度上下文"""
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode
        
        tracer = trace.get_tracer(__name__)
        self._span = tracer.start_span(self.name)
        
        for key, value in self.context.items():
            self._span.set_attribute(key, str(value))
            
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出跨度上下文"""
        if exc_type is not None:
            self._span.set_status(
                Status(StatusCode.ERROR, str(exc_val))
            )
            self._span.record_exception(exc_val)
            
        self._span.end()
    
    def set_tag(self, key: str, value: Any) -> None:
        """设置标签"""
        if self._span:
            self._span.set_attribute(key, str(value))
    
    def log(self, message: str) -> None:
        """记录日志"""
        if self._span:
            self._span.add_event(message)


class Tracer(ABC):
    """追踪器抽象基类"""
    
    def __init__(self, config: TracingConfig):
        self.config = config
    
    @abstractmethod
    async def setup(self) -> None:
        """设置追踪器"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """关闭追踪器"""
        pass
    
    @abstractmethod
    def create_span(
        self,
        name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Span:
        """创建跨度"""
        pass


class OpenTelemetryTracer(Tracer):
    """OpenTelemetry追踪器"""
    
    async def setup(self) -> None:
        """设置OpenTelemetry追踪器"""
        from opentelemetry import trace
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter
        )
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
        
        # 创建资源
        resource = Resource.create({
            "service.name": self.config.service_name,
            "environment": self.config.environment,
            **self.config.tags
        })
        
        # 创建追踪提供者
        provider = TracerProvider(
            resource=resource,
            sampler=TraceIdRatioBased(self.config.sampler_rate)
        )
        
        # 配置导出器
        if self.config.enable_console:
            provider.add_span_processor(
                BatchSpanProcessor(ConsoleSpanExporter())
            )
            
        if self.config.enable_jaeger:
            jaeger_exporter = JaegerExporter(
                agent_host_name=self.config.jaeger_host,
                agent_port=self.config.jaeger_port,
            )
            provider.add_span_processor(
                BatchSpanProcessor(jaeger_exporter)
            )
            
        # 设置全局追踪提供者
        trace.set_tracer_provider(provider)
    
    async def shutdown(self) -> None:
        """关闭追踪器"""
        from opentelemetry import trace
        
        provider = trace.get_tracer_provider()
        await provider.shutdown()
    
    def create_span(
        self,
        name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Span:
        """创建跨度"""
        return Span(name, context)


class TracerRegistry:
    """追踪器注册表"""
    
    def __init__(self):
        self._tracers: Dict[str, Tracer] = {}
    
    def register(
        self,
        name: str,
        tracer: Tracer
    ) -> None:
        """注册追踪器"""
        self._tracers[name] = tracer
    
    def get(
        self,
        name: str
    ) -> Optional[Tracer]:
        """获取追踪器"""
        return self._tracers.get(name)
    
    def list_tracers(self) -> List[str]:
        """列出所有追踪器"""
        return list(self._tracers.keys()) 