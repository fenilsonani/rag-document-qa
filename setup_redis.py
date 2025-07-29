"""
Redis Setup Script for Development
Quick setup script to configure Redis for development environment.
"""

import subprocess
import sys
import time
import logging
from pathlib import Path

def check_redis_installation():
    """Check if Redis is installed."""
    try:
        result = subprocess.run(['redis-server', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Redis is installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ Redis is not installed")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Redis is not installed")
        return False

def install_redis_mac():
    """Install Redis on macOS using Homebrew."""
    print("🍺 Installing Redis using Homebrew...")
    try:
        # Check if Homebrew is installed
        subprocess.run(['brew', '--version'], check=True, capture_output=True)
        
        # Install Redis
        result = subprocess.run(['brew', 'install', 'redis'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Redis installed successfully via Homebrew")
            return True
        else:
            print(f"❌ Failed to install Redis: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError:
        print("❌ Homebrew is not installed. Please install Homebrew first:")
        print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Redis installation timed out")
        return False

def install_redis_linux():
    """Install Redis on Linux."""
    print("🐧 Installing Redis on Linux...")
    try:
        # Try apt-get first (Ubuntu/Debian)
        result = subprocess.run(['sudo', 'apt-get', 'update'], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            result = subprocess.run(['sudo', 'apt-get', 'install', '-y', 'redis-server'], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Redis installed successfully via apt-get")
                return True
        
        # Try yum (CentOS/RHEL)
        result = subprocess.run(['sudo', 'yum', 'install', '-y', 'redis'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Redis installed successfully via yum")
            return True
        
        print("❌ Failed to install Redis. Please install manually.")
        return False
        
    except subprocess.TimeoutExpired:
        print("❌ Redis installation timed out")
        return False

def start_redis_server():
    """Start Redis server."""
    print("🚀 Starting Redis server...")
    
    try:
        # Check if Redis is already running
        result = subprocess.run(['redis-cli', 'ping'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0 and result.stdout.strip() == 'PONG':
            print("✅ Redis server is already running")
            return True
        
        # Start Redis server in background
        if sys.platform == "darwin":  # macOS
            subprocess.run(['brew', 'services', 'start', 'redis'], 
                         capture_output=True, text=True, timeout=30)
        else:  # Linux
            subprocess.run(['sudo', 'systemctl', 'start', 'redis'], 
                         capture_output=True, text=True, timeout=30)
        
        # Wait for Redis to start
        for _ in range(10):
            time.sleep(1)
            result = subprocess.run(['redis-cli', 'ping'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and result.stdout.strip() == 'PONG':
                print("✅ Redis server started successfully")
                return True
        
        print("❌ Failed to start Redis server")
        return False
        
    except subprocess.TimeoutExpired:
        print("❌ Redis startup timed out")
        return False

def test_redis_connection():
    """Test Redis connection and basic operations."""
    print("🔍 Testing Redis connection...")
    
    try:
        import redis
        
        # Connect to Redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        
        # Test basic operations
        r.set('test_key', 'test_value')
        value = r.get('test_key')
        
        if value and value.decode() == 'test_value':
            r.delete('test_key')
            print("✅ Redis connection test successful")
            return True
        else:
            print("❌ Redis connection test failed")
            return False
            
    except ImportError:
        print("❌ Redis Python client not installed. Installing...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'redis>=5.0.0'], 
                         check=True, timeout=120)
            print("✅ Redis Python client installed")
            return test_redis_connection()  # Retry
        except subprocess.CalledProcessError:
            print("❌ Failed to install Redis Python client")
            return False
    
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

def create_redis_config():
    """Create basic Redis configuration for development."""
    config_content = """
# Redis Configuration for RAG Development

# Network settings
bind 127.0.0.1
port 6379

# Memory settings
maxmemory 1gb
maxmemory-policy allkeys-lru

# Persistence settings (optional for development)
save 900 1
save 300 10
save 60 10000

# Log settings
loglevel notice
logfile ""

# Performance settings
tcp-keepalive 300
timeout 0

# Security (no password for development)
# requirepass your_password_here
"""
    
    config_path = Path("redis.conf")
    
    try:
        with open(config_path, 'w') as f:
            f.write(config_content.strip())
        
        print(f"✅ Created Redis config file: {config_path.absolute()}")
        print("   To use this config: redis-server redis.conf")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create Redis config: {e}")
        return False

def create_env_template():
    """Create .env template with Redis settings."""
    env_content = """
# Redis Cache Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
# REDIS_PASSWORD=your_password_here

# Cache Settings
CACHE_ENABLED=true
CACHE_DEFAULT_TTL=3600
CACHE_MAX_MEMORY_MB=1024
"""
    
    env_path = Path(".env.example")
    
    try:
        with open(env_path, 'a') as f:
            f.write(env_content)
        
        print(f"✅ Updated .env.example with Redis settings")
        print("   Copy to .env and adjust settings as needed")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update .env.example: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Redis Setup for RAG System")
    print("=" * 40)
    
    # Step 1: Check if Redis is installed
    if not check_redis_installation():
        print("\n📦 Installing Redis...")
        
        if sys.platform == "darwin":
            success = install_redis_mac()
        elif sys.platform.startswith("linux"):
            success = install_redis_linux()
        else:
            print("❌ Unsupported platform. Please install Redis manually.")
            print("   Visit: https://redis.io/download")
            return False
        
        if not success:
            return False
    
    # Step 2: Start Redis server
    print("\n🚀 Starting Redis server...")
    if not start_redis_server():
        print("\n💡 Manual start instructions:")
        print("   macOS: brew services start redis")
        print("   Linux: sudo systemctl start redis")
        print("   Manual: redis-server")
        return False
    
    # Step 3: Test connection
    print("\n🔍 Testing Redis connection...")
    if not test_redis_connection():
        return False
    
    # Step 4: Create configuration files
    print("\n📝 Creating configuration files...")
    create_redis_config()
    create_env_template()
    
    print("\n🎉 Redis setup completed successfully!")
    print("\n📋 Next steps:")
    print("   1. The Redis server is running on localhost:6379")
    print("   2. Your RAG system will automatically use Redis for caching")
    print("   3. Check the Performance Dashboard in the app for cache metrics")
    print("   4. For production, configure Redis password and persistence")
    
    print("\n🔧 Useful Redis commands:")
    print("   redis-cli ping          # Test connection")
    print("   redis-cli info          # Server info")
    print("   redis-cli flushall      # Clear all data")
    print("   redis-cli monitor       # Monitor commands")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)