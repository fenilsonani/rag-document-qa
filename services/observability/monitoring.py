"""
Enterprise Observability Service - RAG Platform
Comprehensive monitoring, logging, and alerting system.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable
from uuid import uuid4
from dataclasses import dataclass, asdict

from fastapi import BackgroundTasks
from pydantic import BaseModel, Field
import psutil

from ..base.service_base import BaseService, ServiceStatus, ServiceResponse

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import opentelemetry.instrumentation.logging
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class MetricType(str, Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    service: str
    metric: str
    value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    tags: Dict[str, str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricDefinition:
    """Metric definition."""
    name: str
    type: MetricType
    description: str
    labels: List[str] = None
    unit: str = ""


class ObservabilityConfig(BaseModel):
    """Observability configuration."""
    enable_metrics: bool = True
    enable_tracing: bool = True
    enable_logging: bool = True
    metrics_retention_days: int = 30
    trace_sampling_rate: float = 0.1
    log_level: str = "INFO"
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    prometheus_port: int = 9090


class SystemMetrics(BaseModel):
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    network_io_bytes: Dict[str, int]
    open_file_descriptors: int
    load_average: List[float]
    uptime_seconds: float


class ServiceMetrics(BaseModel):
    """Service-specific metrics."""
    service_name: str
    request_count: int
    error_count: int
    avg_response_time_ms: float
    active_connections: int
    cache_hit_rate: float
    queue_size: int
    custom_metrics: Dict[str, float] = Field(default_factory=dict)


class ObservabilityService(BaseService):
    """
    Enterprise observability service providing comprehensive monitoring,
    logging, alerting, and performance analytics.
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="observability",
            version="2.0.0",
            description="Enterprise Observability and Monitoring Service",
            port=8004,
            **kwargs
        )
        
        self.config = ObservabilityConfig()
        
        # Metrics storage and management
        self.metrics_registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        self.custom_metrics: Dict[str, Any] = {}
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        
        # Alert management
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Dict[str, Any]] = []
        self.alert_handlers: Dict[str, Callable] = {}
        
        # Service monitoring
        self.monitored_services: Dict[str, ServiceMetrics] = {}
        self.system_metrics_history: List[SystemMetrics] = []
        
        # Tracing and logging
        self.tracer = None
        self.meter = None
        
        # Background tasks
        self.monitoring_tasks = []
        
        # Initialize observability components
        self._initialize_metrics()
        self._initialize_tracing()
        self._setup_default_alerts()
        self._setup_monitoring_routes()
    
    async def initialize(self):
        """Initialize observability service."""
        try:
            self.logger.info("Initializing observability service...")
            
            # Start background monitoring
            await self._start_background_monitoring()
            
            # Initialize metric collection
            await self._initialize_metric_collection()
            
            self.logger.info("Observability service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize observability service: {e}")
            raise
    
    async def shutdown(self):
        """Cleanup observability resources."""
        self.logger.info("Shutting down observability service...")
        
        # Cancel background tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
    
    async def health_check(self) -> ServiceStatus:
        """Perform observability service health check."""
        try:
            # Check metric collection
            if not self._check_metrics_health():
                return ServiceStatus.DEGRADED
            
            # Check alert system
            if not self._check_alerts_health():
                return ServiceStatus.DEGRADED
            
            return ServiceStatus.HEALTHY
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return ServiceStatus.UNHEALTHY
    
    def _initialize_metrics(self):
        """Initialize metrics collection system."""
        if not PROMETHEUS_AVAILABLE:
            self.logger.warning("Prometheus not available, metrics will be limited")
            return
        
        # Core system metrics
        self.system_cpu_gauge = Gauge(
            'system_cpu_percent',
            'System CPU usage percentage',
            registry=self.metrics_registry
        )
        
        self.system_memory_gauge = Gauge(
            'system_memory_percent',
            'System memory usage percentage',
            registry=self.metrics_registry
        )
        
        self.service_request_counter = Counter(
            'service_requests_total',
            'Total service requests',
            ['service', 'method', 'status'],
            registry=self.metrics_registry
        )
        
        self.service_response_time = Histogram(
            'service_response_time_seconds',
            'Service response time',
            ['service', 'endpoint'],
            registry=self.metrics_registry
        )
        
        self.service_error_counter = Counter(
            'service_errors_total',
            'Total service errors',
            ['service', 'error_type'],
            registry=self.metrics_registry
        )
        
        # Register default metric definitions
        self._register_default_metrics()
    
    def _initialize_tracing(self):
        """Initialize distributed tracing."""
        if not OTEL_AVAILABLE:
            self.logger.warning("OpenTelemetry not available, tracing will be limited")
            return
        
        # Setup tracing
        trace.set_tracer_provider(TracerProvider())
        self.tracer = trace.get_tracer(__name__)
        
        # Setup Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=6831,
        )
        
        # Setup metrics
        metrics.set_meter_provider(MeterProvider())
        self.meter = metrics.get_meter(__name__)
    
    def _setup_monitoring_routes(self):
        """Setup monitoring API routes."""
        
        @self.app.get("/metrics")
        async def get_prometheus_metrics():
            """Get Prometheus formatted metrics."""
            if PROMETHEUS_AVAILABLE and self.metrics_registry:
                return generate_latest(self.metrics_registry).decode('utf-8')
            return "Metrics not available"
        
        @self.app.get("/api/v1/metrics/system", response_model=ServiceResponse)
        async def get_system_metrics():
            """Get current system metrics."""
            try:
                metrics = await self._collect_system_metrics()
                return self.create_response(
                    request_id=str(uuid4()),
                    data=metrics.dict()
                )
            except Exception as e:
                return self.create_response(
                    request_id=str(uuid4()),
                    error=f"Failed to collect system metrics: {str(e)}"
                )
        
        @self.app.get("/api/v1/metrics/services", response_model=ServiceResponse)
        async def get_service_metrics():
            """Get metrics for all monitored services."""
            try:
                return self.create_response(
                    request_id=str(uuid4()),
                    data={
                        "services": {name: metrics.dict() for name, metrics in self.monitored_services.items()},
                        "total_services": len(self.monitored_services)
                    }
                )
            except Exception as e:
                return self.create_response(
                    request_id=str(uuid4()),
                    error=f"Failed to get service metrics: {str(e)}"
                )
        
        @self.app.get("/api/v1/alerts", response_model=ServiceResponse)
        async def get_alerts():
            """Get all active alerts."""
            try:
                active_alerts = [alert.to_dict() for alert in self.alerts.values() if not alert.resolved]
                return self.create_response(
                    request_id=str(uuid4()),
                    data={
                        "active_alerts": active_alerts,
                        "total_alerts": len(active_alerts)
                    }
                )
            except Exception as e:
                return self.create_response(
                    request_id=str(uuid4()),
                    error=f"Failed to get alerts: {str(e)}"
                )
        
        @self.app.post("/api/v1/alerts/resolve/{alert_id}", response_model=ServiceResponse)
        async def resolve_alert(alert_id: str):
            """Resolve an active alert."""
            try:
                if alert_id in self.alerts:
                    alert = self.alerts[alert_id]
                    alert.resolved = True
                    alert.resolved_at = datetime.utcnow()
                    
                    await self.publish_event("alert_resolved", {
                        "alert_id": alert_id,
                        "title": alert.title,
                        "service": alert.service
                    })
                    
                    return self.create_response(
                        request_id=str(uuid4()),
                        data={"resolved": True, "alert_id": alert_id}
                    )
                else:
                    return self.create_response(
                        request_id=str(uuid4()),
                        error="Alert not found"
                    )
            except Exception as e:
                return self.create_response(
                    request_id=str(uuid4()),
                    error=f"Failed to resolve alert: {str(e)}"
                )
        
        @self.app.post("/api/v1/metrics/custom", response_model=ServiceResponse)
        async def record_custom_metric(request: Dict[str, Any]):
            """Record custom metric from services."""
            try:
                metric_name = request.get("name")
                metric_value = request.get("value")
                metric_labels = request.get("labels", {})
                service_name = request.get("service", "unknown")
                
                if not metric_name or metric_value is None:
                    return self.create_response(
                        request_id=str(uuid4()),
                        error="Metric name and value are required"
                    )
                
                # Record the metric
                await self._record_custom_metric(metric_name, metric_value, metric_labels, service_name)
                
                return self.create_response(
                    request_id=str(uuid4()),
                    data={"recorded": True, "metric": metric_name, "value": metric_value}
                )
                
            except Exception as e:
                return self.create_response(
                    request_id=str(uuid4()),
                    error=f"Failed to record metric: {str(e)}"
                )
        
        @self.app.get("/api/v1/dashboard/overview", response_model=ServiceResponse)
        async def get_dashboard_overview():
            """Get comprehensive dashboard overview."""
            try:
                system_metrics = await self._collect_system_metrics()
                
                # Service health summary
                healthy_services = sum(1 for m in self.monitored_services.values() 
                                     if m.error_count == 0 or (m.request_count > 0 and m.error_count / m.request_count < 0.05))
                
                total_services = len(self.monitored_services)
                
                # Alert summary
                critical_alerts = len([a for a in self.alerts.values() if not a.resolved and a.severity == AlertSeverity.CRITICAL])
                total_active_alerts = len([a for a in self.alerts.values() if not a.resolved])
                
                # Performance summary
                avg_response_times = [m.avg_response_time_ms for m in self.monitored_services.values() if m.avg_response_time_ms > 0]
                overall_avg_response = sum(avg_response_times) / len(avg_response_times) if avg_response_times else 0
                
                overview = {
                    "system_health": {
                        "cpu_percent": system_metrics.cpu_percent,
                        "memory_percent": system_metrics.memory_percent,
                        "disk_usage_percent": system_metrics.disk_usage_percent,
                        "uptime_seconds": system_metrics.uptime_seconds
                    },
                    "service_health": {
                        "healthy_services": healthy_services,
                        "total_services": total_services,
                        "health_percentage": (healthy_services / total_services * 100) if total_services > 0 else 0
                    },
                    "alerts": {
                        "critical_alerts": critical_alerts,
                        "total_active_alerts": total_active_alerts
                    },
                    "performance": {
                        "avg_response_time_ms": overall_avg_response,
                        "total_requests": sum(m.request_count for m in self.monitored_services.values()),
                        "total_errors": sum(m.error_count for m in self.monitored_services.values())
                    }
                }
                
                return self.create_response(
                    request_id=str(uuid4()),
                    data=overview
                )
                
            except Exception as e:
                return self.create_response(
                    request_id=str(uuid4()),
                    error=f"Failed to get dashboard overview: {str(e)}"
                )
    
    async def _start_background_monitoring(self):
        """Start background monitoring tasks."""
        # System metrics collection
        self.monitoring_tasks.append(
            asyncio.create_task(self._system_metrics_collector())
        )
        
        # Service health monitoring
        self.monitoring_tasks.append(
            asyncio.create_task(self._service_health_monitor())
        )
        
        # Alert evaluation
        self.monitoring_tasks.append(
            asyncio.create_task(self._alert_evaluator())
        )
        
        # Cleanup old data
        self.monitoring_tasks.append(
            asyncio.create_task(self._data_cleanup_task())
        )
    
    async def _system_metrics_collector(self):
        """Collect system metrics periodically."""
        while True:
            try:
                await asyncio.sleep(10)  # Collect every 10 seconds
                
                metrics = await self._collect_system_metrics()
                
                # Update Prometheus metrics if available
                if PROMETHEUS_AVAILABLE:
                    self.system_cpu_gauge.set(metrics.cpu_percent)
                    self.system_memory_gauge.set(metrics.memory_percent)
                
                # Store in history (keep last 1000 entries)
                self.system_metrics_history.append(metrics)
                if len(self.system_metrics_history) > 1000:
                    self.system_metrics_history.pop(0)
                
                # Check for system alerts
                await self._check_system_alerts(metrics)
                
            except Exception as e:
                self.logger.error(f"System metrics collection failed: {e}")
    
    async def _service_health_monitor(self):
        """Monitor service health and update metrics."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # This would collect metrics from registered services
                # For now, simulate some services
                await self._update_service_metrics()
                
            except Exception as e:
                self.logger.error(f"Service health monitoring failed: {e}")
    
    async def _alert_evaluator(self):
        """Evaluate alert rules and trigger alerts."""
        while True:
            try:
                await asyncio.sleep(5)  # Evaluate every 5 seconds
                
                for rule in self.alert_rules:
                    await self._evaluate_alert_rule(rule)
                
            except Exception as e:
                self.logger.error(f"Alert evaluation failed: {e}")
    
    async def _data_cleanup_task(self):
        """Clean up old metrics and resolved alerts."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up old resolved alerts (older than 7 days)
                cutoff = datetime.utcnow() - timedelta(days=7)
                alerts_to_remove = [
                    alert_id for alert_id, alert in self.alerts.items()
                    if alert.resolved and alert.resolved_at and alert.resolved_at < cutoff
                ]
                
                for alert_id in alerts_to_remove:
                    del self.alerts[alert_id]
                
                self.logger.info(f"Cleaned up {len(alerts_to_remove)} old alerts")
                
            except Exception as e:
                self.logger.error(f"Data cleanup failed: {e}")
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_mb = memory.available / (1024 * 1024)
            
            # Disk usage (root partition)
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_io_bytes = {
                "bytes_sent": network_io.bytes_sent,
                "bytes_recv": network_io.bytes_recv
            }
            
            # Open file descriptors (Unix-like systems)
            try:
                open_fds = len(psutil.Process().open_files())
            except:
                open_fds = 0
            
            # Load average (Unix-like systems)
            try:
                load_average = list(psutil.getloadavg())
            except:
                load_average = [0.0, 0.0, 0.0]
            
            # Uptime
            uptime_seconds = time.time() - psutil.boot_time()
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                network_io_bytes=network_io_bytes,
                open_file_descriptors=open_fds,
                load_average=load_average,
                uptime_seconds=uptime_seconds
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            # Return default metrics
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                network_io_bytes={"bytes_sent": 0, "bytes_recv": 0},
                open_file_descriptors=0,
                load_average=[0.0, 0.0, 0.0],
                uptime_seconds=0.0
            )
    
    async def _check_system_alerts(self, metrics: SystemMetrics):
        """Check system metrics against alert thresholds."""
        # CPU usage alert
        if metrics.cpu_percent > 90:
            await self._trigger_alert(
                title="High CPU Usage",
                description=f"CPU usage is {metrics.cpu_percent}%",
                severity=AlertSeverity.HIGH,
                service="system",
                metric="cpu_percent",
                value=metrics.cpu_percent,
                threshold=90
            )
        
        # Memory usage alert
        if metrics.memory_percent > 85:
            await self._trigger_alert(
                title="High Memory Usage",
                description=f"Memory usage is {metrics.memory_percent}%",
                severity=AlertSeverity.HIGH,
                service="system",
                metric="memory_percent",
                value=metrics.memory_percent,
                threshold=85
            )
        
        # Disk usage alert
        if metrics.disk_usage_percent > 90:
            await self._trigger_alert(
                title="High Disk Usage",
                description=f"Disk usage is {metrics.disk_usage_percent}%",
                severity=AlertSeverity.CRITICAL,
                service="system",
                metric="disk_usage_percent",
                value=metrics.disk_usage_percent,
                threshold=90
            )
    
    async def _trigger_alert(
        self,
        title: str,
        description: str,
        severity: AlertSeverity,
        service: str,
        metric: str,
        value: float,
        threshold: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """Trigger a new alert."""
        alert_id = str(uuid4())
        
        alert = Alert(
            id=alert_id,
            title=title,
            description=description,
            severity=severity,
            service=service,
            metric=metric,
            value=value,
            threshold=threshold,
            timestamp=datetime.utcnow(),
            tags=tags or {}
        )
        
        # Check if similar alert already exists
        existing_alert = None
        for existing in self.alerts.values():
            if (not existing.resolved and 
                existing.service == service and 
                existing.metric == metric and
                existing.severity == severity):
                existing_alert = existing
                break
        
        if not existing_alert:
            self.alerts[alert_id] = alert
            
            # Publish alert event
            await self.publish_event("alert_triggered", {
                "alert_id": alert_id,
                "title": title,
                "severity": severity,
                "service": service,
                "value": value,
                "threshold": threshold
            })
            
            self.logger.warning(f"Alert triggered: {title} (Severity: {severity}, Service: {service})")
    
    async def _record_custom_metric(
        self,
        name: str,
        value: float,
        labels: Dict[str, str],
        service: str
    ):
        """Record a custom metric."""
        metric_key = f"{service}_{name}"
        
        if metric_key not in self.custom_metrics:
            self.custom_metrics[metric_key] = []
        
        metric_entry = {
            "timestamp": datetime.utcnow(),
            "value": value,
            "labels": labels,
            "service": service
        }
        
        self.custom_metrics[metric_key].append(metric_entry)
        
        # Keep only last 1000 entries per metric
        if len(self.custom_metrics[metric_key]) > 1000:
            self.custom_metrics[metric_key].pop(0)
        
        # Update Prometheus metric if available
        if PROMETHEUS_AVAILABLE:
            # This would update the appropriate Prometheus metric
            pass
    
    def _register_default_metrics(self):
        """Register default metric definitions."""
        default_metrics = [
            MetricDefinition(
                name="request_duration",
                type=MetricType.HISTOGRAM,
                description="Request duration in seconds",
                labels=["service", "endpoint", "method"]
            ),
            MetricDefinition(
                name="request_count",
                type=MetricType.COUNTER,
                description="Total number of requests",
                labels=["service", "status_code"]
            ),
            MetricDefinition(
                name="error_count",
                type=MetricType.COUNTER,
                description="Total number of errors",
                labels=["service", "error_type"]
            ),
            MetricDefinition(
                name="active_connections",
                type=MetricType.GAUGE,
                description="Number of active connections",
                labels=["service"]
            )
        ]
        
        for metric in default_metrics:
            self.metric_definitions[metric.name] = metric
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        self.alert_rules = [
            {
                "name": "high_error_rate",
                "condition": "error_rate > 0.05",  # 5% error rate
                "severity": AlertSeverity.HIGH,
                "duration": "5m"
            },
            {
                "name": "slow_response_time",
                "condition": "avg_response_time > 1000",  # 1 second
                "severity": AlertSeverity.MEDIUM,
                "duration": "2m"
            },
            {
                "name": "service_down",
                "condition": "up == 0",
                "severity": AlertSeverity.CRITICAL,
                "duration": "1m"
            }
        ]
    
    async def _initialize_metric_collection(self):
        """Initialize metric collection from external sources."""
        # This would set up collection from external monitoring systems
        pass
    
    async def _update_service_metrics(self):
        """Update metrics for monitored services."""
        # This would collect real metrics from services
        # For demo, create sample data
        sample_services = ["document-processor", "query-intelligence", "vector-search"]
        
        for service in sample_services:
            if service not in self.monitored_services:
                self.monitored_services[service] = ServiceMetrics(
                    service_name=service,
                    request_count=0,
                    error_count=0,
                    avg_response_time_ms=0.0,
                    active_connections=0,
                    cache_hit_rate=0.0,
                    queue_size=0
                )
            
            # Simulate metric updates
            metrics = self.monitored_services[service]
            metrics.request_count += 10
            metrics.error_count += 1 if metrics.request_count % 50 == 0 else 0
            metrics.avg_response_time_ms = 150 + (metrics.request_count % 100)
            metrics.active_connections = 5 + (metrics.request_count % 10)
            metrics.cache_hit_rate = 0.85 + (0.1 * (metrics.request_count % 10) / 10)
    
    async def _evaluate_alert_rule(self, rule: Dict[str, Any]):
        """Evaluate a specific alert rule."""
        # This would implement alert rule evaluation logic
        # For demo, skip implementation
        pass
    
    def _check_metrics_health(self) -> bool:
        """Check if metrics collection is healthy."""
        return len(self.system_metrics_history) > 0
    
    def _check_alerts_health(self) -> bool:
        """Check if alert system is healthy."""
        return len(self.alert_rules) > 0


# Entry point for running the service
if __name__ == "__main__":
    service = ObservabilityService()
    service.run(debug=True)