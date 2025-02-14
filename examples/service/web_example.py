"""
服务基础设施组件使用示例
"""
import asyncio
from datetime import timedelta
from typing import Dict, List, Optional

from base.service.cache import (
    CacheConfig,
    RedisCache,
    CacheRegistry
)
from base.service.core import (
    ServiceConfig,
    ServiceStatus,
    UserRole,
    User,
    AuthConfig
)
from base.service.loadbalancer import (
    ServerConfig,
    Server,
    RoundRobinLoadBalancer,
    LoadBalancerRegistry
)
from base.service.resilience import (
    CircuitBreakerConfig,
    RetryConfig,
    CircuitBreaker,
    Retry,
    ResilienceRegistry
)
from base.service.tracing import (
    TracingConfig,
    OpenTelemetryTracer,
    TracerRegistry
)


class ExampleService:
    """示例服务"""
    
    def __init__(self):
        # 服务配置
        self.config = ServiceConfig(
            name="example_service",
            version="1.0.0",
            host="localhost",
            port=8000,
            auth=AuthConfig(
                auth_type="jwt",
                secret_key="your-secret-key"
            )
        )
        
        # 缓存
        self.cache_registry = CacheRegistry()
        self.setup_cache()
        
        # 追踪
        self.tracer_registry = TracerRegistry()
        self.setup_tracer()
        
        # 负载均衡
        self.lb_registry = LoadBalancerRegistry()
        self.setup_load_balancer()
        
        # 熔断和重试
        self.resilience_registry = ResilienceRegistry()
        self.setup_resilience()
    
    def setup_cache(self) -> None:
        """设置缓存"""
        cache_config = CacheConfig(
            host="localhost",
            port=6379,
            prefix="example:",
            enable_compression=True
        )
        
        cache = RedisCache(cache_config)
        self.cache_registry.register("default", cache)
    
    def setup_tracer(self) -> None:
        """设置追踪器"""
        tracing_config = TracingConfig(
            service_name=self.config.name,
            environment="development",
            enable_console=True,
            tags={
                "version": self.config.version
            }
        )
        
        tracer = OpenTelemetryTracer(tracing_config)
        self.tracer_registry.register("default", tracer)
    
    def setup_load_balancer(self) -> None:
        """设置负载均衡器"""
        lb = RoundRobinLoadBalancer()
        
        # 添加服务器
        servers = [
            ServerConfig(host="server1", port=8001),
            ServerConfig(host="server2", port=8002),
            ServerConfig(host="server3", port=8003)
        ]
        
        for config in servers:
            lb.add_server(Server(config))
            
        self.lb_registry.register("default", lb)
    
    def setup_resilience(self) -> None:
        """设置熔断和重试"""
        # 熔断器
        cb_config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout=30
        )
        
        circuit_breaker = CircuitBreaker(
            "example_breaker",
            cb_config
        )
        
        self.resilience_registry.register_circuit_breaker(
            "default",
            circuit_breaker
        )
        
        # 重试机制
        retry_config = RetryConfig(
            max_attempts=3,
            initial_delay=1.0,
            max_delay=5.0
        )
        
        retry = Retry(retry_config)
        self.resilience_registry.register_retry(
            "default",
            retry
        )
    
    async def example_operation(
        self,
        user_id: str,
        data: Dict
    ) -> Optional[Dict]:
        """示例操作"""
        # 获取组件
        cache = self.cache_registry.get("default")
        tracer = self.tracer_registry.get("default")
        lb = self.lb_registry.get("default")
        circuit_breaker = self.resilience_registry.get_circuit_breaker("default")
        retry = self.resilience_registry.get_retry("default")
        
        # 创建追踪跨度
        async with tracer.create_span(
            "example_operation",
            context={"user_id": user_id}
        ) as span:
            # 检查缓存
            cache_key = f"data:{user_id}"
            cached_data = await cache.get(cache_key)
            
            if cached_data:
                span.set_tag("cache_hit", True)
                return cached_data
                
            span.set_tag("cache_hit", False)
            
            # 获取服务器
            server = await lb.get_next_server(
                context={"client_ip": "127.0.0.1"}
            )
            
            if not server:
                span.set_tag("error", "no_server_available")
                return None
                
            # 定义远程调用
            async def remote_call():
                """模拟远程调用"""
                import aiohttp
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://{server.address}/api/data",
                        json=data
                    ) as response:
                        if response.status != 200:
                            raise Exception(f"Request failed: {response.status}")
                        return await response.json()
            
            try:
                # 使用熔断器和重试机制包装调用
                result = await circuit_breaker.call(
                    retry.call,
                    remote_call
                )
                
                # 缓存结果
                await cache.set(
                    cache_key,
                    result,
                    ttl=timedelta(minutes=5)
                )
                
                span.set_tag("success", True)
                return result
                
            except Exception as e:
                span.set_tag("error", str(e))
                return None


async def main():
    """主函数"""
    # 创建服务
    service = ExampleService()
    
    # 初始化追踪器
    tracer = service.tracer_registry.get("default")
    await tracer.setup()
    
    try:
        # 模拟请求
        user_id = "user123"
        data = {"key": "value"}
        
        result = await service.example_operation(user_id, data)
        print(f"Operation result: {result}")
        
    finally:
        # 关闭追踪器
        await tracer.shutdown()


if __name__ == "__main__":
    asyncio.run(main()) 