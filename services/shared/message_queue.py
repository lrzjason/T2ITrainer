"""
Message Queue module for inter-service communication.
Supports SQLite (primary) with Redis (production) and in-memory queue (development) as fallbacks.
"""
import json
import time
import threading
import asyncio
from queue import Queue, Empty
from typing import Optional, Callable, Any, Dict, List
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# Try to import redis, use in-memory fallback if not available
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, using SQLite message queue (primary) with in-memory fallback")

# Import SQLite queue
from .sqlite_queue import get_sqlite_queue


class InMemoryMessageQueue:
    """
    In-memory message queue for development/testing.
    Mimics Redis Pub/Sub behavior using threading primitives.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._channels: Dict[str, List[Queue]] = defaultdict(list)
        self._queues: Dict[str, Queue] = {}
        self._channel_lock = threading.Lock()
        self._initialized = True
    
    def publish(self, channel: str, message: str) -> int:
        """Publish a message to a channel"""
        with self._channel_lock:
            subscribers = self._channels.get(channel, [])
            for queue in subscribers:
                try:
                    queue.put_nowait({
                        'type': 'message',
                        'channel': channel,
                        'data': message.encode() if isinstance(message, str) else message
                    })
                except:
                    pass
            return len(subscribers)
    
    def subscribe(self, channel: str) -> 'InMemoryPubSub':
        """Subscribe to a channel"""
        return InMemoryPubSub(self, channel)
    
    def rpush(self, key: str, value: str) -> int:
        """Push to a list (right)"""
        if key not in self._queues:
            self._queues[key] = Queue()
        self._queues[key].put(value)
        return self._queues[key].qsize()
    
    def blpop(self, key: str, timeout: int = 0) -> Optional[tuple]:
        """Blocking pop from a list (left)"""
        if key not in self._queues:
            self._queues[key] = Queue()
        try:
            value = self._queues[key].get(timeout=timeout if timeout > 0 else None)
            return (key.encode(), value.encode() if isinstance(value, str) else value)
        except Empty:
            return None
    
    def set(self, key: str, value: str, ex: int = None):
        """Set a value (expiry not implemented for in-memory)"""
        self._queues[key] = value
    
    def get(self, key: str) -> Optional[str]:
        """Get a value"""
        return self._queues.get(key)
    
    def delete(self, key: str):
        """Delete a key"""
        if key in self._queues:
            del self._queues[key]
    
    def _add_subscriber(self, channel: str, queue: Queue):
        """Internal: add subscriber to channel"""
        with self._channel_lock:
            self._channels[channel].append(queue)
    
    def _remove_subscriber(self, channel: str, queue: Queue):
        """Internal: remove subscriber from channel"""
        with self._channel_lock:
            if channel in self._channels:
                try:
                    self._channels[channel].remove(queue)
                except ValueError:
                    pass


class InMemoryPubSub:
    """Mimics redis PubSub for in-memory queue"""
    
    def __init__(self, mq: InMemoryMessageQueue, channel: str = None):
        self._mq = mq
        self._queue = Queue()
        self._subscribed_channels = []
        if channel:
            self.subscribe(channel)
    
    def subscribe(self, *channels):
        """Subscribe to channels"""
        for channel in channels:
            self._mq._add_subscriber(channel, self._queue)
            self._subscribed_channels.append(channel)
    
    def unsubscribe(self, *channels):
        """Unsubscribe from channels"""
        for channel in (channels or self._subscribed_channels):
            self._mq._remove_subscriber(channel, self._queue)
            if channel in self._subscribed_channels:
                self._subscribed_channels.remove(channel)
    
    def get_message(self, timeout: float = 0.1) -> Optional[dict]:
        """Get a message from subscribed channels"""
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None
    
    def close(self):
        """Close the pubsub and unsubscribe from all channels"""
        self.unsubscribe()


class MessageQueue:
    """
    Unified message queue interface.
    Uses SQLite as primary, Redis if available, falls back to in-memory queue.
    """
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.db = db
        self._client = None
        self._use_sqlite = True  # Always use SQLite as primary
        self._use_redis = REDIS_AVAILABLE
    
    @property
    def client(self):
        """Get or create the client"""
        if self._client is None:
            # Use SQLite as primary queue
            try:
                from .sqlite_queue import get_sqlite_queue
                self._client = get_sqlite_queue()
                logger.info("Using SQLite message queue (primary)")
            except Exception as e:
                logger.warning(f"Failed to initialize SQLite queue: {e}")
                # Fallback to Redis if available
                if self._use_redis:
                    try:
                        self._client = redis.Redis(host=self.host, port=self.port, db=self.db)
                        # Test connection
                        self._client.ping()
                        logger.info(f"Connected to Redis at {self.host}:{self.port}")
                    except Exception as e2:
                        logger.warning(f"Failed to connect to Redis: {e2}. Using in-memory queue.")
                        self._use_redis = False
                        self._client = InMemoryMessageQueue()
                else:
                    self._client = InMemoryMessageQueue()
        return self._client
    
    def publish(self, channel: str, message: Any) -> int:
        """Publish a message to a channel"""
        if isinstance(message, dict):
            message = json.dumps(message)
        # For SQLite, we use the output publishing mechanism
        if hasattr(self.client, 'publish_output'):
            # Extract job_id from channel name (JOB_OUTPUT_PREFIX + job_id)
            from .config import JOB_OUTPUT_PREFIX
            if channel.startswith(JOB_OUTPUT_PREFIX):
                job_id = channel[len(JOB_OUTPUT_PREFIX):]
                message_type = message.get('type', 'output') if isinstance(message, dict) else 'output'
                self.client.publish_output(job_id, message_type, message)
                return 1
        return self.client.publish(channel, message) if hasattr(self.client, 'publish') else 0
    
    def pubsub(self):
        """Create a pubsub object - not used with SQLite, but kept for compatibility"""
        if self._use_redis:
            return self.client.pubsub()
        return InMemoryPubSub(self.client)
    
    def rpush(self, key: str, value: Any) -> int:
        """Push to a list (right)"""
        if isinstance(value, dict):
            value = json.dumps(value)
        # For SQLite, we use the job queue mechanism
        if hasattr(self.client, 'push_job'):
            from .config import JOB_QUEUE_KEY
            if key == JOB_QUEUE_KEY:
                # Extract job_id and action from value
                if isinstance(value, str):
                    import json as json_lib
                    try:
                        parsed_value = json_lib.loads(value)
                        job_id = parsed_value.get('job_id', 'unknown')
                        action = parsed_value.get('action', 'unknown')
                        config = parsed_value.get('config', {})
                        return self.client.push_job(job_id, action, config)
                    except:
                        pass
        return self.client.rpush(key, value) if hasattr(self.client, 'rpush') else 0
    
    def blpop(self, key: str, timeout: int = 0) -> Optional[tuple]:
        """Blocking pop from a list (left)"""
        # For SQLite, we use the job queue mechanism
        if hasattr(self.client, 'pop_job'):
            from .config import JOB_QUEUE_KEY
            if key == JOB_QUEUE_KEY:
                job = self.client.pop_job(timeout=timeout)
                if job:
                    # Convert to Redis-compatible format
                    import json as json_lib
                    value = json_lib.dumps({
                        'job_id': job['job_id'],
                        'action': job['action'],
                        'config': job['config']
                    })
                    return (key.encode(), value.encode())
        return self.client.blpop(key, timeout) if hasattr(self.client, 'blpop') else None
    
    def set(self, key: str, value: str, ex: int = None):
        """Set a value with optional expiry"""
        if self._use_redis:
            return self.client.set(key, value, ex=ex)
        return self.client.set(key, value, ex)
    
    def get(self, key: str) -> Optional[str]:
        """Get a value"""
        value = self.client.get(key)
        if value and isinstance(value, bytes):
            return value.decode()
        return value
    
    def delete(self, key: str):
        """Delete a key"""
        return self.client.delete(key)


# Global message queue instance
_mq_instance = None


def get_message_queue(host: str = None, port: int = None, db: int = None) -> MessageQueue:
    """Get the global message queue instance"""
    global _mq_instance
    
    if _mq_instance is None:
        from .config import REDIS_HOST, REDIS_PORT, REDIS_DB
        _mq_instance = MessageQueue(
            host=host or REDIS_HOST,
            port=port or REDIS_PORT,
            db=db or REDIS_DB
        )
    
    return _mq_instance
