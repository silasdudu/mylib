"""
负载均衡模块，提供多种负载均衡策略
"""
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import random

from pydantic import BaseModel


class LoadBalancerType(str, Enum):
    """负载均衡类型"""
    ROUND_ROBIN = "round_robin"  # 轮询
    RANDOM = "random"  # 随机
    LEAST_CONN = "least_conn"  # 最少连接
    WEIGHTED = "weighted"  # 加权
    IP_HASH = "ip_hash"  # IP哈希


class ServerStatus(str, Enum):
    """服务器状态"""
    ONLINE = "online"  # 在线
    OFFLINE = "offline"  # 离线
    DRAINING = "draining"  # 正在下线


class ServerConfig(BaseModel):
    """服务器配置"""
    host: str
    port: int
    weight: int = 1
    max_connections: int = 100
    tags: Dict[str, str] = {}


class Server:
    """服务器"""
    
    def __init__(
        self,
        config: ServerConfig
    ):
        self.config = config
        self.status = ServerStatus.ONLINE
        self.current_connections = 0
        self.total_connections = 0
        self.last_check_time = None
        self.last_response_time = None
    
    @property
    def address(self) -> str:
        """获取服务器地址"""
        return f"{self.config.host}:{self.config.port}"
    
    def is_available(self) -> bool:
        """检查服务器是否可用"""
        return (
            self.status == ServerStatus.ONLINE and
            self.current_connections < self.config.max_connections
        )
    
    async def check_health(self) -> bool:
        """检查服务器健康状态"""
        import aiohttp
        import asyncio
        
        self.last_check_time = datetime.now()
        
        try:
            async with aiohttp.ClientSession() as session:
                start_time = datetime.now()
                async with session.get(
                    f"http://{self.address}/health",
                    timeout=5
                ) as response:
                    self.last_response_time = (datetime.now() - start_time).total_seconds()
                    return response.status == 200
        except:
            return False


class LoadBalancer(ABC):
    """负载均衡器抽象基类"""
    
    def __init__(self):
        self._servers: List[Server] = []
    
    def add_server(
        self,
        server: Server
    ) -> None:
        """添加服务器"""
        self._servers.append(server)
    
    def remove_server(
        self,
        address: str
    ) -> None:
        """移除服务器"""
        self._servers = [
            s for s in self._servers
            if s.address != address
        ]
    
    def get_server(
        self,
        address: str
    ) -> Optional[Server]:
        """获取服务器"""
        for server in self._servers:
            if server.address == address:
                return server
        return None
    
    def list_servers(self) -> List[Server]:
        """列出所有服务器"""
        return self._servers
    
    @abstractmethod
    async def get_next_server(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Server]:
        """获取下一个服务器"""
        pass


class RoundRobinLoadBalancer(LoadBalancer):
    """轮询负载均衡器"""
    
    def __init__(self):
        super().__init__()
        self._current_index = 0
    
    async def get_next_server(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Server]:
        """获取下一个服务器"""
        if not self._servers:
            return None
            
        # 查找下一个可用的服务器
        start_index = self._current_index
        while True:
            self._current_index = (self._current_index + 1) % len(self._servers)
            server = self._servers[self._current_index]
            
            if server.is_available():
                return server
                
            if self._current_index == start_index:
                return None


class RandomLoadBalancer(LoadBalancer):
    """随机负载均衡器"""
    
    async def get_next_server(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Server]:
        """获取下一个服务器"""
        available_servers = [
            s for s in self._servers
            if s.is_available()
        ]
        
        if not available_servers:
            return None
            
        return random.choice(available_servers)


class LeastConnectionLoadBalancer(LoadBalancer):
    """最少连接负载均衡器"""
    
    async def get_next_server(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Server]:
        """获取下一个服务器"""
        available_servers = [
            s for s in self._servers
            if s.is_available()
        ]
        
        if not available_servers:
            return None
            
        return min(
            available_servers,
            key=lambda s: s.current_connections
        )


class WeightedLoadBalancer(LoadBalancer):
    """加权负载均衡器"""
    
    async def get_next_server(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Server]:
        """获取下一个服务器"""
        available_servers = [
            s for s in self._servers
            if s.is_available()
        ]
        
        if not available_servers:
            return None
            
        total_weight = sum(s.config.weight for s in available_servers)
        if total_weight == 0:
            return random.choice(available_servers)
            
        r = random.uniform(0, total_weight)
        upto = 0
        
        for server in available_servers:
            upto += server.config.weight
            if upto > r:
                return server
                
        return available_servers[-1]


class IPHashLoadBalancer(LoadBalancer):
    """IP哈希负载均衡器"""
    
    async def get_next_server(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Server]:
        """获取下一个服务器"""
        available_servers = [
            s for s in self._servers
            if s.is_available()
        ]
        
        if not available_servers:
            return None
            
        if not context or "client_ip" not in context:
            return random.choice(available_servers)
            
        client_ip = context["client_ip"]
        hash_value = sum(ord(c) for c in client_ip)
        return available_servers[hash_value % len(available_servers)]


class LoadBalancerRegistry:
    """负载均衡器注册表"""
    
    def __init__(self):
        self._load_balancers: Dict[str, LoadBalancer] = {}
    
    def register(
        self,
        name: str,
        load_balancer: LoadBalancer
    ) -> None:
        """注册负载均衡器"""
        self._load_balancers[name] = load_balancer
    
    def get(
        self,
        name: str
    ) -> Optional[LoadBalancer]:
        """获取负载均衡器"""
        return self._load_balancers.get(name)
    
    def list_load_balancers(self) -> List[str]:
        """列出所有负载均衡器"""
        return list(self._load_balancers.keys()) 