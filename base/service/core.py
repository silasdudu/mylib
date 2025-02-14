"""
服务核心模块，提供Web服务所需的基础设施组件
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class ServiceStatus(str, Enum):
    """服务状态"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class UserRole(str, Enum):
    """用户角色"""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"
    API = "api"


@dataclass
class ServiceMetrics:
    """服务指标"""
    total_requests: int = 0
    active_requests: int = 0
    error_count: int = 0
    avg_response_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    last_updated: datetime = None


class RateLimitConfig(BaseModel):
    """限流配置"""
    max_requests: int = 100  # 最大请求数
    time_window: int = 60    # 时间窗口(秒)
    burst_size: int = 10     # 突发大小
    per_user: bool = True    # 是否按用户限流
    per_ip: bool = True      # 是否按IP限流


class AuthConfig(BaseModel):
    """认证配置"""
    auth_type: str = "jwt"  # jwt, oauth2, api_key
    token_expire: int = 3600  # token过期时间(秒)
    refresh_token_expire: int = 86400  # 刷新token过期时间(秒)
    secret_key: str  # 密钥
    allowed_origins: List[str] = ["*"]  # CORS配置


class ServiceConfig(BaseModel):
    """服务配置"""
    name: str
    version: str
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 4
    rate_limit: RateLimitConfig = RateLimitConfig()
    auth: AuthConfig
    extra_params: Dict[str, Any] = {}


class User(BaseModel):
    """用户模型"""
    id: str
    username: str
    role: UserRole
    api_quota: Optional[int] = None
    rate_limit: Optional[RateLimitConfig] = None
    metadata: Dict[str, Any] = {}


class RateLimiter(ABC):
    """限流器抽象基类"""
    
    @abstractmethod
    async def acquire(
        self,
        key: str,
        cost: int = 1
    ) -> bool:
        """获取令牌"""
        pass
    
    @abstractmethod
    async def get_remaining(
        self,
        key: str
    ) -> int:
        """获取剩余令牌数"""
        pass


class TokenBucketLimiter(RateLimiter):
    """令牌桶限流器"""
    
    def __init__(
        self,
        config: RateLimitConfig
    ):
        self.config = config
        self._buckets: Dict[str, Dict[str, Any]] = {}
    
    async def acquire(
        self,
        key: str,
        cost: int = 1
    ) -> bool:
        """获取令牌"""
        now = datetime.now().timestamp()
        
        if key not in self._buckets:
            self._buckets[key] = {
                "tokens": self.config.burst_size,
                "last_update": now
            }
            
        bucket = self._buckets[key]
        
        # 计算新增令牌
        elapsed = now - bucket["last_update"]
        new_tokens = elapsed * (self.config.max_requests / self.config.time_window)
        bucket["tokens"] = min(
            bucket["tokens"] + new_tokens,
            self.config.burst_size
        )
        bucket["last_update"] = now
        
        # 尝试获取令牌
        if bucket["tokens"] >= cost:
            bucket["tokens"] -= cost
            return True
            
        return False
    
    async def get_remaining(
        self,
        key: str
    ) -> int:
        """获取剩余令牌数"""
        if key not in self._buckets:
            return self.config.burst_size
        return int(self._buckets[key]["tokens"])


class AuthProvider(ABC):
    """认证提供者抽象基类"""
    
    @abstractmethod
    async def authenticate(
        self,
        credentials: Dict[str, str]
    ) -> Optional[User]:
        """认证用户"""
        pass
    
    @abstractmethod
    async def create_token(
        self,
        user: User
    ) -> str:
        """创建token"""
        pass
    
    @abstractmethod
    async def verify_token(
        self,
        token: str
    ) -> Optional[User]:
        """验证token"""
        pass
    
    @abstractmethod
    async def refresh_token(
        self,
        refresh_token: str
    ) -> Optional[str]:
        """刷新token"""
        pass


class JWTAuthProvider(AuthProvider):
    """JWT认证提供者"""
    
    def __init__(
        self,
        config: AuthConfig
    ):
        self.config = config
    
    async def authenticate(
        self,
        credentials: Dict[str, str]
    ) -> Optional[User]:
        """认证用户"""
        # 实现具体的认证逻辑
        pass
    
    async def create_token(
        self,
        user: User
    ) -> str:
        """创建JWT token"""
        # 实现具体的token创建逻辑
        pass
    
    async def verify_token(
        self,
        token: str
    ) -> Optional[User]:
        """验证JWT token"""
        # 实现具体的token验证逻辑
        pass
    
    async def refresh_token(
        self,
        refresh_token: str
    ) -> Optional[str]:
        """刷新JWT token"""
        # 实现具体的token刷新逻辑
        pass


class MetricsCollector(ABC):
    """指标收集器抽象基类"""
    
    @abstractmethod
    async def collect(self) -> ServiceMetrics:
        """收集指标"""
        pass
    
    @abstractmethod
    async def record_request(
        self,
        path: str,
        method: str,
        status_code: int,
        response_time: float
    ) -> None:
        """记录请求"""
        pass
    
    @abstractmethod
    async def export(
        self,
        format: str = "prometheus"
    ) -> str:
        """导出指标"""
        pass


class PrometheusCollector(MetricsCollector):
    """Prometheus指标收集器"""
    
    def __init__(self):
        self._metrics = ServiceMetrics()
        self._requests_total = 0
        self._requests_active = 0
        self._errors_total = 0
        self._response_times: List[float] = []
    
    async def collect(self) -> ServiceMetrics:
        """收集指标"""
        import psutil
        
        self._metrics.total_requests = self._requests_total
        self._metrics.active_requests = self._requests_active
        self._metrics.error_count = self._errors_total
        self._metrics.avg_response_time = (
            sum(self._response_times) / len(self._response_times)
            if self._response_times else 0.0
        )
        self._metrics.memory_usage = psutil.Process().memory_percent()
        self._metrics.cpu_usage = psutil.Process().cpu_percent()
        self._metrics.last_updated = datetime.now()
        
        return self._metrics
    
    async def record_request(
        self,
        path: str,
        method: str,
        status_code: int,
        response_time: float
    ) -> None:
        """记录请求指标"""
        self._requests_total += 1
        self._response_times.append(response_time)
        
        if len(self._response_times) > 1000:
            self._response_times = self._response_times[-1000:]
            
        if status_code >= 400:
            self._errors_total += 1
    
    async def export(
        self,
        format: str = "prometheus"
    ) -> str:
        """导出Prometheus格式指标"""
        metrics = await self.collect()
        
        if format == "prometheus":
            return f"""
# HELP api_requests_total Total number of API requests
# TYPE api_requests_total counter
api_requests_total {metrics.total_requests}

# HELP api_requests_active Current number of active requests
# TYPE api_requests_active gauge
api_requests_active {metrics.active_requests}

# HELP api_errors_total Total number of API errors
# TYPE api_errors_total counter
api_errors_total {metrics.error_count}

# HELP api_response_time_seconds Average response time in seconds
# TYPE api_response_time_seconds gauge
api_response_time_seconds {metrics.avg_response_time}

# HELP process_memory_usage Memory usage percentage
# TYPE process_memory_usage gauge
process_memory_usage {metrics.memory_usage}

# HELP process_cpu_usage CPU usage percentage
# TYPE process_cpu_usage gauge
process_cpu_usage {metrics.cpu_usage}
"""
        return str(metrics)


class HealthCheck(ABC):
    """健康检查抽象基类"""
    
    @abstractmethod
    async def check(self) -> bool:
        """执行健康检查"""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """获取状态详情"""
        pass


class ServiceHealthCheck(HealthCheck):
    """服务健康检查"""
    
    def __init__(
        self,
        service_name: str,
        dependencies: List[str] = None
    ):
        self.service_name = service_name
        self.dependencies = dependencies or []
        self._status = ServiceStatus.STARTING
        self._last_check = None
        self._error = None
    
    async def check(self) -> bool:
        """检查服务健康状态"""
        try:
            # 检查依赖服务
            for dep in self.dependencies:
                # 实现依赖服务检查逻辑
                pass
                
            self._status = ServiceStatus.RUNNING
            self._last_check = datetime.now()
            self._error = None
            return True
            
        except Exception as e:
            self._status = ServiceStatus.ERROR
            self._last_check = datetime.now()
            self._error = str(e)
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """获取健康状态详情"""
        return {
            "service": self.service_name,
            "status": self._status,
            "last_check": self._last_check,
            "error": self._error,
            "dependencies": self.dependencies
        }


class ServiceRegistry:
    """服务注册表"""
    
    def __init__(self):
        self._rate_limiters: Dict[str, RateLimiter] = {}
        self._auth_providers: Dict[str, AuthProvider] = {}
        self._metrics_collectors: Dict[str, MetricsCollector] = {}
        self._health_checks: Dict[str, HealthCheck] = {}
    
    def register_rate_limiter(
        self,
        name: str,
        limiter: RateLimiter
    ) -> None:
        """注册限流器"""
        self._rate_limiters[name] = limiter
    
    def register_auth_provider(
        self,
        name: str,
        provider: AuthProvider
    ) -> None:
        """注册认证提供者"""
        self._auth_providers[name] = provider
    
    def register_metrics_collector(
        self,
        name: str,
        collector: MetricsCollector
    ) -> None:
        """注册指标收集器"""
        self._metrics_collectors[name] = collector
    
    def register_health_check(
        self,
        name: str,
        check: HealthCheck
    ) -> None:
        """注册健康检查"""
        self._health_checks[name] = check
    
    def get_rate_limiter(
        self,
        name: str
    ) -> Optional[RateLimiter]:
        """获取限流器"""
        return self._rate_limiters.get(name)
    
    def get_auth_provider(
        self,
        name: str
    ) -> Optional[AuthProvider]:
        """获取认证提供者"""
        return self._auth_providers.get(name)
    
    def get_metrics_collector(
        self,
        name: str
    ) -> Optional[MetricsCollector]:
        """获取指标收集器"""
        return self._metrics_collectors.get(name)
    
    def get_health_check(
        self,
        name: str
    ) -> Optional[HealthCheck]:
        """获取健康检查"""
        return self._health_checks.get(name) 