#!/bin/bash

# Enterprise RAG Platform - Service Orchestration Script
# Starts all microservices in the correct order

set -e

echo "🚀 Starting Enterprise RAG Platform Services..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker and Docker Compose are installed
check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Dependencies check passed"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p uploads
    mkdir -p vector_store
    mkdir -p models
    mkdir -p logs
    mkdir -p config/grafana/dashboards
    mkdir -p config/grafana/datasources
    mkdir -p config/nginx/ssl
    
    print_success "Directories created"
}

# Generate configuration files
generate_configs() {
    print_status "Generating configuration files..."
    
    # Create Grafana datasource configuration
    cat > config/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

    # Create environment file if it doesn't exist
    if [ ! -f .env ]; then
        cat > .env << EOF
# Enterprise RAG Platform Configuration

# API Keys (add your actual keys)
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Service Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TEMPERATURE=0.7
MAX_TOKENS=1000

# Redis Configuration
REDIS_URL=redis://redis:6379

# Observability
JAEGER_ENDPOINT=http://jaeger:14268/api/traces
PROMETHEUS_PORT=9090

# Security
JWT_SECRET=your-jwt-secret-key-here

# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO
EOF
        print_warning "Created .env file with default values. Please update with your actual API keys."
    fi
    
    print_success "Configuration files generated"
}

# Start infrastructure services first
start_infrastructure() {
    print_status "Starting infrastructure services..."
    
    docker-compose -f docker-compose.microservices.yml up -d \
        redis \
        jaeger \
        prometheus \
        grafana \
        chromadb
    
    print_status "Waiting for infrastructure services to be ready..."
    sleep 10
    
    print_success "Infrastructure services started"
}

# Start application services
start_application_services() {
    print_status "Starting application services..."
    
    # Start services in dependency order
    docker-compose -f docker-compose.microservices.yml up -d \
        document-processor \
        query-intelligence \
        vector-search \
        observability
    
    print_status "Waiting for application services to be ready..."
    sleep 15
    
    # Start API gateway
    docker-compose -f docker-compose.microservices.yml up -d \
        api-gateway
    
    sleep 10
    
    # Start load balancer
    docker-compose -f docker-compose.microservices.yml up -d \
        nginx
    
    print_success "Application services started"
}

# Health check
health_check() {
    print_status "Performing health checks..."
    
    services=(
        "http://localhost:8000/health"  # API Gateway
        "http://localhost:8001/health"  # Document Processor
        "http://localhost:8002/health"  # Query Intelligence
        "http://localhost:8003/health"  # Vector Search
        "http://localhost:8004/health"  # Observability
    )
    
    for service in "${services[@]}"; do
        service_name=$(echo $service | sed 's|.*://localhost:[0-9]*/.*|\1|' | sed 's|http://localhost:||' | sed 's|/.*||')
        
        for i in {1..30}; do
            if curl -f -s $service > /dev/null 2>&1; then
                print_success "Service on port $service_name is healthy"
                break
            fi
            
            if [ $i -eq 30 ]; then
                print_warning "Service on port $service_name health check failed"
            fi
            
            sleep 2
        done
    done
}

# Display service URLs
display_urls() {
    echo ""
    echo "🎉 Enterprise RAG Platform is ready!"
    echo ""
    echo "📋 Service URLs:"
    echo "   API Gateway:        http://localhost:8000"
    echo "   API Documentation:  http://localhost:8000/docs"
    echo "   Grafana Dashboard:  http://localhost:3000 (admin/admin)"
    echo "   Jaeger Tracing:     http://localhost:16686"
    echo "   Prometheus:         http://localhost:9090"
    echo ""
    echo "🔧 Individual Services:"
    echo "   Document Processor: http://localhost:8001"
    echo "   Query Intelligence: http://localhost:8002"
    echo "   Vector Search:      http://localhost:8003"
    echo "   Observability:      http://localhost:8004"
    echo ""
    echo "💡 Try the API:"
    echo "   curl -X POST http://localhost:8000/api/v1/query/analyze \\"
    echo "        -H \"Content-Type: application/json\" \\"
    echo "        -d '{\"query_text\": \"What is artificial intelligence?\"}'"
    echo ""
}

# Cleanup function
cleanup() {
    if [ "$1" == "down" ]; then
        print_status "Stopping all services..."
        docker-compose -f docker-compose.microservices.yml down
        print_success "All services stopped"
    elif [ "$1" == "clean" ]; then
        print_status "Stopping and removing all services, networks, and volumes..."
        docker-compose -f docker-compose.microservices.yml down -v --remove-orphans
        docker system prune -f
        print_success "Cleanup completed"
    fi
}

# Show logs
show_logs() {
    if [ -n "$2" ]; then
        docker-compose -f docker-compose.microservices.yml logs -f "$2"
    else
        docker-compose -f docker-compose.microservices.yml logs -f
    fi
}

# Main execution
case "${1:-start}" in
    "start")
        check_dependencies
        create_directories
        generate_configs
        start_infrastructure
        start_application_services
        health_check
        display_urls
        ;;
    "stop")
        cleanup "down"
        ;;
    "clean")
        cleanup "clean"
        ;;
    "logs")
        show_logs "$@"
        ;;
    "status")
        docker-compose -f docker-compose.microservices.yml ps
        ;;
    "restart")
        cleanup "down"
        sleep 5
        $0 start
        ;;
    *)
        echo "Usage: $0 {start|stop|clean|logs [service]|status|restart}"
        echo ""
        echo "Commands:"
        echo "  start   - Start all services"
        echo "  stop    - Stop all services"
        echo "  clean   - Stop and remove all services, networks, and volumes"
        echo "  logs    - Show logs (optionally for specific service)"
        echo "  status  - Show service status"
        echo "  restart - Restart all services"
        exit 1
        ;;
esac