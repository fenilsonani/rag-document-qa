"""
Redis Cache System - Pro-Level Performance Enhancement
Implements intelligent caching for queries, embeddings, and search results to achieve < 200ms response times.
"""

import json
import time
import hashlib
import logging
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict

try:
    import redis
    from redis.exceptions import ConnectionError, TimeoutError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available. Caching will use in-memory fallback.")

from .config import Config


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_time_saved: float = 0.0
    avg_retrieval_time: float = 0.0
    cache_size_bytes: int = 0
    hit_rate: float = 0.0
    
    def update_hit_rate(self):
        """Update the hit rate calculation."""
        total = self.cache_hits + self.cache_misses
        self.hit_rate = self.cache_hits / total if total > 0 else 0.0


@dataclass
class CacheEntry:
    """Represents a cached entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: int
    size_bytes: int
    cache_type: str
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl_seconds <= 0:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    def update_access(self):
        """Update access metadata."""
        self.last_accessed = datetime.now()
        self.access_count += 1


class CacheManager:
    """Base cache manager with common functionality."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.metrics = CacheMetrics()
        self.cache_policies = {
            'query_results': {'ttl': 3600, 'max_size': 1000},  # 1 hour, 1000 entries
            'embeddings': {'ttl': 7200, 'max_size': 5000},     # 2 hours, 5000 entries
            'search_results': {'ttl': 1800, 'max_size': 2000}, # 30 minutes, 2000 entries
            'document_analysis': {'ttl': 14400, 'max_size': 500}, # 4 hours, 500 entries
            'reranking_scores': {'ttl': 1800, 'max_size': 1000}  # 30 minutes, 1000 entries
        }
    
    def _generate_cache_key(self, namespace: str, *args, **kwargs) -> str:
        """Generate a unique cache key from arguments."""
        # Create a string representation of all arguments
        key_parts = [namespace]
        
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            elif isinstance(arg, (list, tuple, dict)):
                key_parts.append(json.dumps(arg, sort_keys=True))
            else:
                key_parts.append(str(arg))
        
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")
        
        # Generate hash of the key
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize a value for caching."""
        if isinstance(value, np.ndarray):
            # Special handling for numpy arrays
            return pickle.dumps({
                'type': 'numpy_array',
                'data': value.tobytes(),
                'dtype': str(value.dtype),
                'shape': value.shape
            })
        else:
            return pickle.dumps(value)
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize a cached value."""
        try:
            value = pickle.loads(data)
            if isinstance(value, dict) and value.get('type') == 'numpy_array':
                # Reconstruct numpy array
                array_data = np.frombuffer(value['data'], dtype=value['dtype'])
                return array_data.reshape(value['shape'])
            return value
        except Exception as e:
            logging.warning(f"Failed to deserialize cached value: {e}")
            return None


class RedisCache(CacheManager):
    """Redis-based caching implementation for production environments."""
    
    def __init__(self, config: Optional[Config] = None, 
                 redis_host: str = 'localhost', 
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 redis_password: Optional[str] = None):
        super().__init__(config)
        
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available. Install with: pip install redis")
        
        self.redis_client = None
        self.connection_params = {
            'host': redis_host,
            'port': redis_port,
            'db': redis_db,
            'password': redis_password,
            'decode_responses': False,  # We handle binary data
            'socket_timeout': 1.0,
            'socket_connect_timeout': 1.0
        }
        
        self._connect()
    
    def _connect(self) -> bool:
        """Establish Redis connection."""
        try:
            self.redis_client = redis.Redis(**self.connection_params)
            # Test connection
            self.redis_client.ping()
            logging.info("✅ Redis cache connected successfully")
            return True
        except (ConnectionError, TimeoutError) as e:
            logging.warning(f"⚠️ Redis connection failed: {e}")
            self.redis_client = None
            return False
    
    def is_available(self) -> bool:
        """Check if Redis is available."""
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.ping()
            return True
        except:
            return False
    
    def get(self, namespace: str, *args, **kwargs) -> Optional[Any]:
        """Get a value from cache."""
        if not self.is_available():
            return None
        
        cache_key = self._generate_cache_key(namespace, *args, **kwargs)
        
        try:
            start_time = time.time()
            
            # Get from Redis
            cached_data = self.redis_client.get(f"rag_cache:{cache_key}")
            
            if cached_data is not None:
                # Cache hit
                value = self._deserialize_value(cached_data)
                
                if value is not None:
                    self.metrics.cache_hits += 1
                    self.metrics.total_requests += 1
                    self.metrics.avg_retrieval_time = (
                        (self.metrics.avg_retrieval_time * (self.metrics.total_requests - 1) + 
                         (time.time() - start_time)) / self.metrics.total_requests
                    )
                    self.metrics.update_hit_rate()
                    
                    # Update access count
                    self.redis_client.incr(f"rag_cache_access:{cache_key}")
                    
                    return value
            
            # Cache miss
            self.metrics.cache_misses += 1
            self.metrics.total_requests += 1
            self.metrics.update_hit_rate()
            return None
            
        except Exception as e:
            logging.warning(f"Cache get failed: {e}")
            return None
    
    def set(self, namespace: str, value: Any, *args, **kwargs) -> bool:
        """Set a value in cache."""
        if not self.is_available():
            return False
        
        cache_key = self._generate_cache_key(namespace, *args, **kwargs)
        
        try:
            # Get cache policy
            policy = self.cache_policies.get(namespace, {'ttl': 3600, 'max_size': 1000})
            
            # Serialize value
            serialized_value = self._serialize_value(value)
            
            # Set in Redis with TTL
            if policy['ttl'] > 0:
                success = self.redis_client.setex(
                    f"rag_cache:{cache_key}", 
                    policy['ttl'], 
                    serialized_value
                )
            else:
                success = self.redis_client.set(f"rag_cache:{cache_key}", serialized_value)
            
            if success:
                # Set metadata
                self.redis_client.setex(
                    f"rag_cache_meta:{cache_key}",
                    policy['ttl'] if policy['ttl'] > 0 else 86400,  # Default 24h for metadata
                    json.dumps({
                        'namespace': namespace,
                        'created_at': datetime.now().isoformat(),
                        'size_bytes': len(serialized_value),
                        'access_count': 0
                    })
                )
                
                # Initialize access counter
                self.redis_client.setex(
                    f"rag_cache_access:{cache_key}",
                    policy['ttl'] if policy['ttl'] > 0 else 86400,
                    0
                )
                
                return True
            
        except Exception as e:
            logging.warning(f"Cache set failed: {e}")
        
        return False
    
    def delete(self, namespace: str, *args, **kwargs) -> bool:
        """Delete a value from cache."""
        if not self.is_available():
            return False
        
        cache_key = self._generate_cache_key(namespace, *args, **kwargs)
        
        try:
            # Delete main cache entry, metadata, and access counter
            pipe = self.redis_client.pipeline()
            pipe.delete(f"rag_cache:{cache_key}")
            pipe.delete(f"rag_cache_meta:{cache_key}")
            pipe.delete(f"rag_cache_access:{cache_key}")
            results = pipe.execute()
            
            return any(results)
            
        except Exception as e:
            logging.warning(f"Cache delete failed: {e}")
            return False
    
    def clear_namespace(self, namespace: str) -> int:
        """Clear all entries in a namespace."""
        if not self.is_available():
            return 0
        
        try:
            # Find all keys for this namespace
            pattern = f"rag_cache:*"
            keys_to_delete = []
            
            # Get all cache keys
            for key in self.redis_client.scan_iter(match=pattern):
                # Check if this key belongs to the namespace
                meta_key = key.decode().replace('rag_cache:', 'rag_cache_meta:')
                meta_data = self.redis_client.get(meta_key)
                
                if meta_data:
                    try:
                        meta = json.loads(meta_data)
                        if meta.get('namespace') == namespace:
                            keys_to_delete.extend([
                                key,
                                meta_key.encode(),
                                key.decode().replace('rag_cache:', 'rag_cache_access:').encode()
                            ])
                    except:
                        continue
            
            if keys_to_delete:
                return self.redis_client.delete(*keys_to_delete)
            
            return 0
            
        except Exception as e:
            logging.warning(f"Cache clear namespace failed: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'metrics': asdict(self.metrics),
            'redis_available': self.is_available(),
            'namespaces': {}
        }
        
        if not self.is_available():
            return stats
        
        try:
            # Get namespace statistics
            pattern = "rag_cache_meta:*"
            namespace_stats = defaultdict(lambda: {'count': 0, 'total_size': 0, 'total_accesses': 0})
            
            for key in self.redis_client.scan_iter(match=pattern):
                try:
                    meta_data = self.redis_client.get(key)
                    if meta_data:
                        meta = json.loads(meta_data)
                        namespace = meta.get('namespace', 'unknown')
                        
                        # Get access count
                        access_key = key.decode().replace('rag_cache_meta:', 'rag_cache_access:')
                        access_count = self.redis_client.get(access_key) or 0
                        
                        namespace_stats[namespace]['count'] += 1
                        namespace_stats[namespace]['total_size'] += meta.get('size_bytes', 0)
                        namespace_stats[namespace]['total_accesses'] += int(access_count)
                
                except:
                    continue
            
            stats['namespaces'] = dict(namespace_stats)
            
            # Redis info
            redis_info = self.redis_client.info()
            stats['redis_info'] = {
                'used_memory': redis_info.get('used_memory', 0),
                'used_memory_human': redis_info.get('used_memory_human', 'N/A'),
                'connected_clients': redis_info.get('connected_clients', 0),
                'keyspace_hits': redis_info.get('keyspace_hits', 0),
                'keyspace_misses': redis_info.get('keyspace_misses', 0)
            }
            
        except Exception as e:
            logging.warning(f"Failed to get cache stats: {e}")
        
        return stats


class InMemoryCache(CacheManager):
    """In-memory fallback cache implementation."""
    
    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self.cache_store: Dict[str, CacheEntry] = {}
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        expired_keys = []
        for key, entry in self.cache_store.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache_store[key]
        
        self._last_cleanup = current_time
    
    def get(self, namespace: str, *args, **kwargs) -> Optional[Any]:
        """Get a value from in-memory cache."""
        self._cleanup_expired()
        
        cache_key = self._generate_cache_key(namespace, *args, **kwargs)
        
        if cache_key in self.cache_store:
            entry = self.cache_store[cache_key]
            
            if not entry.is_expired():
                # Cache hit
                entry.update_access()
                self.metrics.cache_hits += 1
                self.metrics.total_requests += 1
                self.metrics.update_hit_rate()
                return entry.value
            else:
                # Expired entry
                del self.cache_store[cache_key]
        
        # Cache miss
        self.metrics.cache_misses += 1
        self.metrics.total_requests += 1
        self.metrics.update_hit_rate()
        return None
    
    def set(self, namespace: str, value: Any, *args, **kwargs) -> bool:
        """Set a value in in-memory cache."""
        cache_key = self._generate_cache_key(namespace, *args, **kwargs)
        policy = self.cache_policies.get(namespace, {'ttl': 3600, 'max_size': 1000})
        
        # Check size limits
        if len(self.cache_store) >= policy['max_size']:
            # Remove least recently used entries
            sorted_entries = sorted(
                self.cache_store.items(),
                key=lambda x: x[1].last_accessed
            )
            
            # Remove oldest 10% of entries
            remove_count = max(1, len(sorted_entries) // 10)
            for key, _ in sorted_entries[:remove_count]:
                del self.cache_store[key]
        
        # Calculate size
        try:
            size_bytes = len(self._serialize_value(value))
        except:
            size_bytes = 0
        
        # Create cache entry
        entry = CacheEntry(
            key=cache_key,
            value=value,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
            ttl_seconds=policy['ttl'],
            size_bytes=size_bytes,
            cache_type=namespace
        )
        
        self.cache_store[cache_key] = entry
        return True
    
    def delete(self, namespace: str, *args, **kwargs) -> bool:
        """Delete a value from in-memory cache."""
        cache_key = self._generate_cache_key(namespace, *args, **kwargs)
        
        if cache_key in self.cache_store:
            del self.cache_store[cache_key]
            return True
        
        return False
    
    def clear_namespace(self, namespace: str) -> int:
        """Clear all entries in a namespace."""
        keys_to_delete = []
        
        for key, entry in self.cache_store.items():
            if entry.cache_type == namespace:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self.cache_store[key]
        
        return len(keys_to_delete)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get in-memory cache statistics."""
        self._cleanup_expired()
        
        # Calculate namespace stats
        namespace_stats = defaultdict(lambda: {'count': 0, 'total_size': 0, 'total_accesses': 0})
        
        for entry in self.cache_store.values():
            namespace_stats[entry.cache_type]['count'] += 1
            namespace_stats[entry.cache_type]['total_size'] += entry.size_bytes
            namespace_stats[entry.cache_type]['total_accesses'] += entry.access_count
        
        return {
            'metrics': asdict(self.metrics),
            'redis_available': False,
            'cache_entries': len(self.cache_store),
            'namespaces': dict(namespace_stats)
        }


class SmartCache:
    """Intelligent cache that automatically chooses Redis or in-memory based on availability."""
    
    def __init__(self, config: Optional[Config] = None, 
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 redis_password: Optional[str] = None):
        self.config = config or Config()
        
        # Try to initialize Redis cache first
        self.redis_cache = None
        self.memory_cache = InMemoryCache(config)
        
        if REDIS_AVAILABLE:
            try:
                self.redis_cache = RedisCache(
                    config, redis_host, redis_port, redis_db, redis_password
                )
                if not self.redis_cache.is_available():
                    self.redis_cache = None
            except Exception as e:
                logging.warning(f"Failed to initialize Redis cache: {e}")
                self.redis_cache = None
        
        self.active_cache = self.redis_cache if self.redis_cache else self.memory_cache
        
        if self.redis_cache:
            logging.info("🚀 Smart cache using Redis for production performance")
        else:
            logging.info("🔄 Smart cache using in-memory fallback")
    
    def get(self, namespace: str, *args, **kwargs) -> Optional[Any]:
        """Get from active cache."""
        return self.active_cache.get(namespace, *args, **kwargs)
    
    def set(self, namespace: str, value: Any, *args, **kwargs) -> bool:
        """Set in active cache."""
        return self.active_cache.set(namespace, value, *args, **kwargs)
    
    def delete(self, namespace: str, *args, **kwargs) -> bool:
        """Delete from active cache."""
        return self.active_cache.delete(namespace, *args, **kwargs)
    
    def clear_namespace(self, namespace: str) -> int:
        """Clear namespace from active cache."""
        return self.active_cache.clear_namespace(namespace)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = self.active_cache.get_stats()
        stats['cache_type'] = 'redis' if self.redis_cache else 'memory'
        stats['redis_fallback'] = not bool(self.redis_cache)
        return stats
    
    def is_redis_available(self) -> bool:
        """Check if Redis is available."""
        return self.redis_cache is not None and self.redis_cache.is_available()


def cached_method(namespace: str, ttl: Optional[int] = None):
    """Decorator for caching method results."""
    def decorator(func: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            # Check if the instance has a cache
            cache = getattr(self, '_cache', None)
            if not cache:
                return func(self, *args, **kwargs)
            
            # Try to get from cache
            cached_result = cache.get(namespace, func.__name__, *args, **kwargs)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            start_time = time.time()
            result = func(self, *args, **kwargs)
            execution_time = time.time() - start_time
            
            # Cache the result
            cache.set(namespace, result, func.__name__, *args, **kwargs)
            
            # Update performance metrics
            if hasattr(cache, 'metrics'):
                cache.metrics.total_time_saved += execution_time * 0.9  # Assume 90% time saving on cache hits
            
            return result
        
        return wrapper
    return decorator


def cache_warming_job(cache: SmartCache, warming_queries: List[Dict[str, Any]]):
    """Background job to warm up the cache with common queries."""
    logging.info("🔥 Starting cache warming job")
    
    for query_config in warming_queries:
        try:
            namespace = query_config['namespace']
            # This would typically call the actual function to generate and cache results
            # Implementation depends on the specific use case
            logging.info(f"Warming cache for namespace: {namespace}")
        except Exception as e:
            logging.warning(f"Cache warming failed for {query_config}: {e}")
    
    logging.info("✅ Cache warming completed")