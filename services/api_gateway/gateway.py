"""
Enterprise API Gateway - RAG Platform
Orchestrates all microservices with advanced routing, authentication, and monitoring.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
import uvicorn

from ..base.service_base import BaseService, ServiceStatus, ServiceResponse, ServiceRegistry

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from opentelemetry import trace
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


class APIVersion(str, Enum):
    """API version enumeration."""
    V1 = "v1"
    V2 = "v2"


class ServiceRoute(BaseModel):
    """Service route configuration."""
    service_name: str
    service_url: str
    health_check_url: str
    timeout_seconds: int = 30
    retry_count: int = 3
    circuit_breaker_threshold: int = 5
    load_balancer_weight: int = 100


class APIGatewayRequest(BaseModel):
    """Standardized gateway request."""
    service: str
    endpoint: str
    method: str = "POST"
    data: Dict[str, Any] = Field(default_factory=dict)
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout: Optional[int] = None
    retry_count: Optional[int] = None


class APIGatewayResponse(BaseModel):
    """Standardized gateway response."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    service: str
    endpoint: str
    request_id: str
    processing_time_ms: float
    gateway_version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    trace_id: Optional[str] = None


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    requests_per_minute: int = 100
    burst_limit: int = 10
    window_size_seconds: int = 60


class AuthenticationConfig(BaseModel):
    """Authentication configuration."""
    enabled: bool = True
    jwt_secret: str = "your-secret-key"
    token_expiry_hours: int = 24
    require_tenant: bool = True


class EnterpriseAPIGateway(BaseService):
    """
    Enterprise API Gateway for RAG Platform.
    Provides unified access to all microservices with advanced features.
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="api-gateway",
            version="2.0.0",
            description="Enterprise API Gateway for RAG Platform",
            port=8000,
            **kwargs
        )
        
        # Service registry and discovery
        self.service_registry = ServiceRegistry(self.redis_client)
        self.service_routes: Dict[str, ServiceRoute] = {}
        self.service_health: Dict[str, bool] = {}
        
        # HTTP client for service communication
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=200)
        )
        
        # Authentication and authorization
        self.auth_config = AuthenticationConfig()
        self.security = HTTPBearer(auto_error=False)
        
        # Rate limiting
        self.rate_limits: Dict[str, RateLimitConfig] = {
            "default": RateLimitConfig(requests_per_minute=100),
            "premium": RateLimitConfig(requests_per_minute=1000),
            "enterprise": RateLimitConfig(requests_per_minute=10000)
        }
        
        # Gateway statistics
        self.gateway_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time_ms": 0.0,
            "service_requests": {},
            "rate_limit_violations": 0,
            "authentication_failures": 0,
            "circuit_breaker_trips": 0
        }
        
        # Setup gateway routes and middleware
        self._setup_middleware()
        self._setup_gateway_routes()
        
        # Load balancing and health checking
        self.load_balancer = LoadBalancer()
        self.health_checker = HealthChecker(self)
        
        # Start background tasks
        asyncio.create_task(self._start_health_checking())
        asyncio.create_task(self._start_service_discovery())
    
    async def initialize(self):
        """Initialize API Gateway components."""
        try:
            self.logger.info("Initializing API Gateway...")
            
            # Register default services
            await self._register_default_services()
            
            # Initialize rate limiting
            await self._initialize_rate_limiting()
            
            # Start health checking
            self.health_checker.start()
            
            self.logger.info("API Gateway initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize API Gateway: {e}")
            raise
    
    async def shutdown(self):
        """Cleanup API Gateway resources."""
        self.logger.info("Shutting down API Gateway...")
        
        await self.http_client.aclose()
        if self.health_checker:
            await self.health_checker.stop()
    
    async def health_check(self) -> ServiceStatus:
        """Perform gateway health check."""
        try:
            # Check service connectivity
            healthy_services = sum(1 for status in self.service_health.values() if status)
            total_services = len(self.service_health)
            
            if total_services == 0:
                return ServiceStatus.STARTING
            
            health_ratio = healthy_services / total_services
            
            if health_ratio >= 0.8:
                return ServiceStatus.HEALTHY
            elif health_ratio >= 0.5:
                return ServiceStatus.DEGRADED
            else:
                return ServiceStatus.UNHEALTHY
                
        except Exception as e:
            self.logger.error(f"Gateway health check failed: {e}")
            return ServiceStatus.UNHEALTHY
    
    def _setup_middleware(self):
        """Setup gateway middleware."""
        
        @self.app.middleware("http")
        async def add_request_tracking(request: Request, call_next):
            """Track requests and add correlation IDs."""
            start_time = time.time()
            request_id = str(uuid4())
            
            # Add request ID to headers
            request.state.request_id = request_id
            request.state.start_time = start_time
            
            # Add trace ID if available
            trace_id = None
            if OTEL_AVAILABLE:
                span = trace.get_current_span()
                if span:
                    trace_id = format(span.get_span_context().trace_id, '032x')
            
            response = await call_next(request)
            
            # Add response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = str((time.time() - start_time) * 1000)
            if trace_id:
                response.headers["X-Trace-ID"] = trace_id
            
            # Update statistics
            self._update_gateway_stats(request, response, time.time() - start_time)
            
            return response
        
        @self.app.middleware("http")
        async def rate_limiting_middleware(request: Request, call_next):
            """Apply rate limiting based on user tier."""
            if not await self._check_rate_limit(request):
                self.gateway_stats["rate_limit_violations"] += 1
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
            
            return await call_next(request)
    
    def _setup_gateway_routes(self):
        """Setup API Gateway routes."""
        
        @self.app.get("/")
        async def gateway_info():
            """Gateway information and status."""
            return {
                "service": "Enterprise RAG API Gateway",
                "version": self.version,
                "status": self.status,
                "services": list(self.service_routes.keys()),
                "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds()
            }
        
        @self.app.post("/api/v1/gateway/request", response_model=APIGatewayResponse)
        async def gateway_request(
            request: APIGatewayRequest,
            auth_credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            """Generic gateway request handler."""
            start_time = time.time()
            request_id = str(uuid4())
            
            try:
                # Authenticate request
                user_info = await self._authenticate_request(auth_credentials)
                
                # Route request to appropriate service
                response = await self._route_request(request, user_info, request_id)
                
                processing_time = (time.time() - start_time) * 1000
                
                return APIGatewayResponse(
                    success=True,
                    data=response,
                    service=request.service,
                    endpoint=request.endpoint,
                    request_id=request_id,
                    processing_time_ms=processing_time,
                    gateway_version=self.version
                )
                
            except Exception as e:
                self.logger.error(f"Gateway request failed: {e}")
                processing_time = (time.time() - start_time) * 1000
                
                return APIGatewayResponse(
                    success=False,
                    error=str(e),
                    service=request.service,
                    endpoint=request.endpoint,
                    request_id=request_id,
                    processing_time_ms=processing_time,
                    gateway_version=self.version
                )
        
        # Document Processing Service Routes
        @self.app.post("/api/v1/documents/process")
        async def process_document(
            request: Dict[str, Any],
            auth: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            """Process document through document processing service."""
            return await self._proxy_to_service("document-processor", "/api/v1/process/upload", request, auth)
        
        @self.app.get("/api/v1/documents/{document_id}")
        async def get_document(
            document_id: str,
            auth: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            """Get document processing status."""
            return await self._proxy_to_service("document-processor", f"/api/v1/document/{document_id}", {}, auth)
        
        # Query Intelligence Service Routes
        @self.app.post("/api/v1/query/analyze")
        async def analyze_query(
            request: Dict[str, Any],
            auth: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            """Analyze query through query intelligence service."""
            return await self._proxy_to_service("query-intelligence", "/api/v1/analyze", request, auth)
        
        @self.app.post("/api/v1/query/enhance")
        async def enhance_query(
            request: Dict[str, Any],
            auth: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            """Enhance query through query intelligence service."""
            return await self._proxy_to_service("query-intelligence", "/api/v1/enhance", request, auth)
        
        # Vector Search Service Routes
        @self.app.post("/api/v1/search")
        async def search_documents(
            request: Dict[str, Any],
            auth: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            """Search documents through vector search service."""
            return await self._proxy_to_service("vector-search", "/api/v1/search", request, auth)
        
        @self.app.post("/api/v1/search/batch")
        async def batch_search(
            request: Dict[str, Any],
            auth: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            """Perform batch search through vector search service."""
            return await self._proxy_to_service("vector-search", "/api/v1/search/batch", request, auth)
        
        # Advanced Orchestration Routes
        @self.app.post("/api/v1/rag/complete")
        async def complete_rag_pipeline(
            request: Dict[str, Any],
            auth: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            """Complete RAG pipeline: query analysis + search + generation."""
            return await self._orchestrate_rag_pipeline(request, auth)
        
        # Gateway Management Routes
        @self.app.get("/api/v1/gateway/services")
        async def list_services(
            auth: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            """List all registered services."""
            await self._authenticate_request(auth, require_admin=True)
            
            services_info = []
            for name, route in self.service_routes.items():
                services_info.append({
                    "name": name,
                    "url": route.service_url,
                    "healthy": self.service_health.get(name, False),
                    "timeout": route.timeout_seconds,
                    "weight": route.load_balancer_weight
                })
            
            return {"services": services_info}
        
        @self.app.get("/api/v1/gateway/stats")
        async def get_gateway_stats(
            auth: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            """Get gateway statistics."""
            await self._authenticate_request(auth, require_admin=True)
            
            stats = self.gateway_stats.copy()
            stats.update({
                "service_health": self.service_health,
                "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds()
            })
            
            return stats
        
        # OpenAPI customization
        def custom_openapi():
            if self.app.openapi_schema:
                return self.app.openapi_schema
            
            openapi_schema = get_openapi(
                title="Enterprise RAG Platform API",
                version=self.version,
                description="Comprehensive API for enterprise document intelligence and RAG capabilities",
                routes=self.app.routes,
            )
            
            # Add security schemes
            openapi_schema["components"]["securitySchemes"] = {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT"
                }
            }
            
            self.app.openapi_schema = openapi_schema
            return openapi_schema
        
        self.app.openapi = custom_openapi
    
    async def _authenticate_request(
        self,
        credentials: Optional[HTTPAuthorizationCredentials],
        require_admin: bool = False
    ) -> Dict[str, Any]:
        """Authenticate and authorize request."""
        if not self.auth_config.enabled:
            return {"user_id": "anonymous", "tenant_id": "default"}
        
        if not credentials:
            self.gateway_stats["authentication_failures"] += 1
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        try:
            # Validate JWT token (simplified)
            # In production, this would use proper JWT validation
            token = credentials.credentials
            
            # Mock user info extraction
            user_info = {
                "user_id": "user123",
                "tenant_id": "tenant456",
                "role": "user",
                "tier": "premium"
            }
            
            if require_admin and user_info.get("role") != "admin":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin access required"
                )
            
            return user_info
            
        except Exception as e:
            self.gateway_stats["authentication_failures"] += 1
            self.logger.warning(f"Authentication failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
    
    async def _check_rate_limit(self, request: Request) -> bool:
        """Check if request is within rate limits."""
        if not REDIS_AVAILABLE or not self.redis_client:
            return True  # Skip rate limiting if Redis unavailable
        
        try:
            # Extract user/IP for rate limiting
            user_id = getattr(request.state, 'user_id', 'anonymous')
            client_ip = request.client.host if request.client else 'unknown'
            
            # Use user-specific or IP-based rate limiting
            rate_limit_key = f"rate_limit:{user_id}:{client_ip}"
            
            # Get user tier for rate limit configuration
            tier = getattr(request.state, 'tier', 'default')
            rate_config = self.rate_limits.get(tier, self.rate_limits["default"])
            
            # Check current request count
            current_count = self.redis_client.get(rate_limit_key)
            if current_count is None:
                current_count = 0
            else:
                current_count = int(current_count)
            
            if current_count >= rate_config.requests_per_minute:
                return False
            
            # Increment counter
            pipe = self.redis_client.pipeline()
            pipe.incr(rate_limit_key)
            pipe.expire(rate_limit_key, rate_config.window_size_seconds)
            pipe.execute()
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Rate limiting check failed: {e}")
            return True  # Allow request if rate limiting fails
    
    async def _route_request(
        self,
        request: APIGatewayRequest,
        user_info: Dict[str, Any],
        request_id: str
    ) -> Any:
        """Route request to appropriate service."""
        service_name = request.service
        
        if service_name not in self.service_routes:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Service {service_name} not found"
            )
        
        service_route = self.service_routes[service_name]
        
        # Check service health
        if not self.service_health.get(service_name, False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Service {service_name} is unavailable"
            )
        
        # Prepare request
        url = f"{service_route.service_url}{request.endpoint}"
        timeout = request.timeout or service_route.timeout_seconds
        max_retries = request.retry_count or service_route.retry_count
        
        # Add authentication and tenant info to request
        request_data = request.data.copy()
        request_data.update({
            "user_id": user_info["user_id"],
            "tenant_id": user_info["tenant_id"],
            "request_id": request_id
        })
        
        # Perform request with retries
        for attempt in range(max_retries):
            try:
                if request.method.upper() == "GET":
                    response = await self.http_client.get(
                        url,
                        params=request_data,
                        headers=request.headers,
                        timeout=timeout
                    )
                else:
                    response = await self.http_client.post(
                        url,
                        json=request_data,
                        headers=request.headers,
                        timeout=timeout
                    )
                
                response.raise_for_status()
                return response.json()
                
            except httpx.RequestError as e:
                if attempt == max_retries - 1:  # Last attempt
                    self.logger.error(f"Service request failed after {max_retries} attempts: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"Service {service_name} request failed"
                    )
                
                # Wait before retry
                await asyncio.sleep(0.5 * (attempt + 1))
            
            except httpx.HTTPStatusError as e:
                # Don't retry client errors
                if e.response.status_code < 500:
                    raise HTTPException(
                        status_code=e.response.status_code,
                        detail=f"Service error: {e.response.text}"
                    )
                
                if attempt == max_retries - 1:
                    raise HTTPException(
                        status_code=e.response.status_code,
                        detail=f"Service {service_name} returned error"
                    )
                
                await asyncio.sleep(0.5 * (attempt + 1))
    
    async def _proxy_to_service(
        self,
        service_name: str,
        endpoint: str,
        request_data: Dict[str, Any],
        auth: Optional[HTTPAuthorizationCredentials]
    ) -> Dict[str, Any]:
        """Proxy request to specific service."""
        user_info = await self._authenticate_request(auth)
        request_id = str(uuid4())
        
        gateway_request = APIGatewayRequest(
            service=service_name,
            endpoint=endpoint,
            data=request_data
        )
        
        return await self._route_request(gateway_request, user_info, request_id)
    
    async def _orchestrate_rag_pipeline(
        self,
        request: Dict[str, Any],
        auth: Optional[HTTPAuthorizationCredentials]
    ) -> Dict[str, Any]:
        """Orchestrate complete RAG pipeline across multiple services."""
        user_info = await self._authenticate_request(auth)
        pipeline_id = str(uuid4())
        
        try:
            # Step 1: Query Analysis
            query_analysis = await self._route_request(
                APIGatewayRequest(
                    service="query-intelligence",
                    endpoint="/api/v1/analyze",
                    data={"query_text": request.get("query", "")}
                ),
                user_info,
                f"{pipeline_id}_analysis"
            )
            
            # Step 2: Vector Search based on analysis
            search_request = {
                "query": request.get("query", ""),
                "strategy": query_analysis.get("data", {}).get("routing_decision", {}).get("recommended_strategy", "hybrid"),
                "top_k": request.get("top_k", 10)
            }
            
            search_results = await self._route_request(
                APIGatewayRequest(
                    service="vector-search",
                    endpoint="/api/v1/search",
                    data=search_request
                ),
                user_info,
                f"{pipeline_id}_search"
            )
            
            # Step 3: Generate response (would be implemented when generation service exists)
            # For now, return analysis + search results
            
            return {
                "pipeline_id": pipeline_id,
                "query_analysis": query_analysis,
                "search_results": search_results,
                "status": "completed"
            }
            
        except Exception as e:
            self.logger.error(f"RAG pipeline orchestration failed: {e}")
            return {
                "pipeline_id": pipeline_id,
                "error": str(e),
                "status": "failed"
            }
    
    async def _register_default_services(self):
        """Register default RAG platform services."""
        default_services = [
            ServiceRoute(
                service_name="document-processor",
                service_url="http://localhost:8001",
                health_check_url="http://localhost:8001/health",
                timeout_seconds=60  # Document processing can take longer
            ),
            ServiceRoute(
                service_name="query-intelligence",
                service_url="http://localhost:8002",
                health_check_url="http://localhost:8002/health"
            ),
            ServiceRoute(
                service_name="vector-search",
                service_url="http://localhost:8003",
                health_check_url="http://localhost:8003/health"
            )
        ]
        
        for service in default_services:
            self.service_routes[service.service_name] = service
            # Initialize health status
            self.service_health[service.service_name] = False
    
    async def _initialize_rate_limiting(self):
        """Initialize rate limiting system."""
        try:
            if REDIS_AVAILABLE and self.redis_client:
                # Test Redis connection for rate limiting
                self.redis_client.ping()
                self.logger.info("Rate limiting initialized with Redis")
            else:
                self.logger.warning("Redis unavailable, rate limiting disabled")
        except Exception as e:
            self.logger.warning(f"Rate limiting initialization failed: {e}")
    
    def _update_gateway_stats(self, request: Request, response: Response, processing_time: float):
        """Update gateway statistics."""
        self.gateway_stats["total_requests"] += 1
        
        if response.status_code < 400:
            self.gateway_stats["successful_requests"] += 1
        else:
            self.gateway_stats["failed_requests"] += 1
        
        # Update average response time
        total_requests = self.gateway_stats["total_requests"]
        prev_avg = self.gateway_stats["avg_response_time_ms"]
        self.gateway_stats["avg_response_time_ms"] = (
            (prev_avg * (total_requests - 1) + processing_time * 1000) / total_requests
        )
    
    async def _start_health_checking(self):
        """Start background health checking of services."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                for service_name, service_route in self.service_routes.items():
                    try:
                        response = await self.http_client.get(
                            service_route.health_check_url,
                            timeout=5.0
                        )
                        self.service_health[service_name] = response.status_code == 200
                    except Exception:
                        self.service_health[service_name] = False
                        
            except Exception as e:
                self.logger.error(f"Health checking error: {e}")
    
    async def _start_service_discovery(self):
        """Start background service discovery."""
        while True:
            try:
                await asyncio.sleep(60)  # Discover every minute
                
                # This would implement service discovery logic
                # For now, we use static configuration
                
            except Exception as e:
                self.logger.error(f"Service discovery error: {e}")


class LoadBalancer:
    """Simple load balancer for service instances."""
    
    def __init__(self):
        self.service_instances: Dict[str, List[str]] = {}
        self.current_instance: Dict[str, int] = {}
    
    def add_instance(self, service_name: str, instance_url: str):
        """Add service instance."""
        if service_name not in self.service_instances:
            self.service_instances[service_name] = []
            self.current_instance[service_name] = 0
        
        if instance_url not in self.service_instances[service_name]:
            self.service_instances[service_name].append(instance_url)
    
    def get_instance(self, service_name: str) -> Optional[str]:
        """Get next instance using round-robin."""
        if service_name not in self.service_instances:
            return None
        
        instances = self.service_instances[service_name]
        if not instances:
            return None
        
        # Round-robin selection
        current = self.current_instance[service_name]
        instance = instances[current]
        self.current_instance[service_name] = (current + 1) % len(instances)
        
        return instance


class HealthChecker:
    """Health checker for services."""
    
    def __init__(self, gateway: EnterpriseAPIGateway):
        self.gateway = gateway
        self.running = False
    
    def start(self):
        """Start health checking."""
        self.running = True
    
    async def stop(self):
        """Stop health checking."""
        self.running = False


# Entry point for running the gateway
if __name__ == "__main__":
    gateway = EnterpriseAPIGateway()
    gateway.run(debug=True)