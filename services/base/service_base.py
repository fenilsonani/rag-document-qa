"""
Base Service Architecture for Enterprise RAG Platform
Provides common service patterns, health checks, and observability.
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Union, Callable
import traceback

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
import uvicorn

try:
    from opentelemetry import trace, metrics, baggage
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logging.warning("OpenTelemetry not available. Observability features will be limited.")

try:
    import redis
    from redis.connection import ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class ServiceStatus(str, Enum):
    """Service health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"


@dataclass
class ServiceMetrics:
    """Service performance and health metrics."""
    request_count: int = 0
    error_count: int = 0
    avg_response_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    uptime_seconds: int = 0
    last_health_check: datetime = None
    custom_metrics: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        data = asdict(self)
        if self.last_health_check:
            data['last_health_check'] = self.last_health_check.isoformat()
        return data


class ServiceRequest(BaseModel):
    """Base service request model with tracing and metadata."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ServiceResponse(BaseModel):
    """Base service response model with status and metrics."""
    success: bool = True
    data: Any = None
    error: Optional[str] = None
    request_id: str
    processing_time_ms: float
    service_name: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit breaker pattern implementation for service resilience."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        async def wrapper(*args, **kwargs):
            if self.state == "open":
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = "half_open"
                else:
                    raise CircuitBreakerError("Circuit breaker is OPEN")
            
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                if self.state == "half_open":
                    self.reset()
                return result
            
            except self.expected_exception as e:
                self.record_failure()
                raise e
        
        return wrapper
    
    def record_failure(self):
        """Record a service failure."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
    
    def reset(self):
        """Reset circuit breaker to closed state."""
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_time = None


class BaseService(ABC):
    """
    Abstract base class for all enterprise RAG platform services.
    Provides common functionality for health checks, metrics, and observability.
    """
    
    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        description: str = "",
        port: int = 8000,
        enable_observability: bool = True,
        redis_url: Optional[str] = None
    ):
        self.name = name
        self.version = version
        self.description = description
        self.port = port
        self.start_time = datetime.utcnow()
        self.metrics = ServiceMetrics()
        self.status = ServiceStatus.STARTING
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title=f"Enterprise RAG - {name}",
            description=description,
            version=version,
            docs_url="/api/docs",
            redoc_url="/api/redoc",
            openapi_url="/api/openapi.json"
        )
        
        # Add middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Setup observability
        if enable_observability and OTEL_AVAILABLE:
            self._setup_observability()
        
        # Setup Redis connection
        self.redis_client = None
        if redis_url and REDIS_AVAILABLE:
            self._setup_redis(redis_url)
        
        # Setup routes
        self._setup_routes()
        
        # Circuit breakers for external services
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        self.logger.info(f"Service {name} v{version} initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for the service."""
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        
        # Create structured formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s - '
            '[service=%(name)s] [version=%(version)s]',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_observability(self):
        """Setup OpenTelemetry tracing and metrics."""
        if not OTEL_AVAILABLE:
            return
        
        # Setup tracing
        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer(__name__)
        
        # Setup Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=6831,
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        # Instrument FastAPI
        FastAPIInstrumentor.instrument_app(self.app)
        LoggingInstrumentor().instrument()
        
        self.tracer = tracer
        self.logger.info("OpenTelemetry observability configured")
    
    def _setup_redis(self, redis_url: str):
        """Setup Redis connection for caching and pub/sub."""
        try:
            self.redis_client = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            self.logger.info("Redis connection established")
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def _setup_routes(self):
        """Setup common API routes for all services."""
        
        @self.app.middleware("http")
        async def add_process_time_header(request: Request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            
            # Update metrics
            self.metrics.request_count += 1
            self.metrics.avg_response_time = (
                (self.metrics.avg_response_time * (self.metrics.request_count - 1) + process_time * 1000) 
                / self.metrics.request_count
            )
            
            return response
        
        @self.app.get("/health")
        async def health_check():
            """Service health check endpoint."""
            health_status = await self.health_check()
            return {
                "service": self.name,
                "version": self.version,
                "status": health_status,
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds()
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics endpoint."""
            self.metrics.uptime_seconds = int((datetime.utcnow() - self.start_time).total_seconds())
            self.metrics.last_health_check = datetime.utcnow()
            return self.metrics.to_dict()
        
        @self.app.get("/info")
        async def service_info():
            """Get service information endpoint."""
            return {
                "name": self.name,
                "version": self.version,
                "description": self.description,
                "status": self.status,
                "start_time": self.start_time.isoformat(),
                "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds()
            }
    
    @abstractmethod
    async def initialize(self):
        """Initialize service-specific resources."""
        pass
    
    @abstractmethod
    async def shutdown(self):
        """Cleanup service resources."""
        pass
    
    @abstractmethod
    async def health_check(self) -> ServiceStatus:
        """Perform service health check."""
        pass
    
    def add_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0
    ) -> CircuitBreaker:
        """Add a circuit breaker for external service calls."""
        circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    async def cache_get(self, key: str, default=None) -> Any:
        """Get value from Redis cache."""
        if not self.redis_client:
            return default
        
        try:
            value = self.redis_client.get(f"{self.name}:{key}")
            if value:
                return json.loads(value)
            return default
        except Exception as e:
            self.logger.warning(f"Cache get error: {e}")
            return default
    
    async def cache_set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in Redis cache with TTL."""
        if not self.redis_client:
            return False
        
        try:
            serialized = json.dumps(value, default=str)
            return self.redis_client.setex(f"{self.name}:{key}", ttl, serialized)
        except Exception as e:
            self.logger.warning(f"Cache set error: {e}")
            return False
    
    async def publish_event(self, event_type: str, data: Dict[str, Any]):
        """Publish event to Redis pub/sub."""
        if not self.redis_client:
            return
        
        event = {
            "service": self.name,
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            self.redis_client.publish(f"events:{event_type}", json.dumps(event, default=str))
        except Exception as e:
            self.logger.warning(f"Event publish error: {e}")
    
    @asynccontextmanager
    async def trace_operation(self, operation_name: str):
        """Context manager for tracing operations."""
        start_time = time.time()
        trace_id = str(uuid.uuid4())
        
        if hasattr(self, 'tracer'):
            with self.tracer.start_as_current_span(operation_name) as span:
                span.set_attribute("service.name", self.name)
                span.set_attribute("service.version", self.version)
                span.set_attribute("trace.id", trace_id)
                try:
                    yield trace_id
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise
                finally:
                    end_time = time.time()
                    span.set_attribute("duration.ms", (end_time - start_time) * 1000)
        else:
            yield trace_id
    
    def create_response(
        self,
        request_id: str,
        data: Any = None,
        error: str = None,
        processing_time_ms: float = 0.0,
        metadata: Dict[str, Any] = None
    ) -> ServiceResponse:
        """Create standardized service response."""
        return ServiceResponse(
            success=error is None,
            data=data,
            error=error,
            request_id=request_id,
            processing_time_ms=processing_time_ms,
            service_name=self.name,
            version=self.version,
            metadata=metadata or {}
        )
    
    async def start(self):
        """Start the service."""
        try:
            await self.initialize()
            self.status = ServiceStatus.HEALTHY
            self.logger.info(f"Service {self.name} started successfully on port {self.port}")
            
            # Publish service start event
            await self.publish_event("service_started", {
                "service": self.name,
                "version": self.version,
                "port": self.port
            })
            
        except Exception as e:
            self.status = ServiceStatus.UNHEALTHY
            self.logger.error(f"Failed to start service {self.name}: {e}")
            raise
    
    async def stop(self):
        """Stop the service gracefully."""
        try:
            self.status = ServiceStatus.STOPPING
            await self.shutdown()
            
            # Publish service stop event
            await self.publish_event("service_stopped", {
                "service": self.name,
                "version": self.version
            })
            
            if self.redis_client:
                self.redis_client.close()
            
            self.logger.info(f"Service {self.name} stopped gracefully")
            
        except Exception as e:
            self.logger.error(f"Error during service shutdown: {e}")
    
    def run(self, host: str = "0.0.0.0", debug: bool = False):
        """Run the service using uvicorn."""
        uvicorn_config = {
            "host": host,
            "port": self.port,
            "log_level": "info" if not debug else "debug",
            "access_log": True,
            "use_colors": True
        }
        
        if debug:
            uvicorn_config.update({
                "reload": True,
                "reload_dirs": ["./services"]
            })
        
        uvicorn.run(self.app, **uvicorn_config)


class ServiceRegistry:
    """Service discovery and registration."""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.services: Dict[str, Dict[str, Any]] = {}
    
    async def register_service(
        self,
        name: str,
        host: str,
        port: int,
        health_check_url: str,
        metadata: Dict[str, Any] = None
    ):
        """Register a service in the registry."""
        service_info = {
            "name": name,
            "host": host,
            "port": port,
            "health_check_url": health_check_url,
            "registered_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        self.services[name] = service_info
        
        if self.redis_client:
            try:
                await self.redis_client.hset(
                    "service_registry",
                    name,
                    json.dumps(service_info, default=str)
                )
            except Exception as e:
                logging.warning(f"Failed to register service in Redis: {e}")
    
    async def discover_service(self, name: str) -> Optional[Dict[str, Any]]:
        """Discover a service by name."""
        if name in self.services:
            return self.services[name]
        
        if self.redis_client:
            try:
                service_data = await self.redis_client.hget("service_registry", name)
                if service_data:
                    return json.loads(service_data)
            except Exception as e:
                logging.warning(f"Failed to discover service from Redis: {e}")
        
        return None
    
    async def list_services(self) -> Dict[str, Dict[str, Any]]:
        """List all registered services."""
        if self.redis_client:
            try:
                services_data = await self.redis_client.hgetall("service_registry")
                return {
                    name: json.loads(data)
                    for name, data in services_data.items()
                }
            except Exception as e:
                logging.warning(f"Failed to list services from Redis: {e}")
        
        return self.services.copy()