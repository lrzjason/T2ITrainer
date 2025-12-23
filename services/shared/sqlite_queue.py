"""
SQLite-based Message Queue for inter-service communication.
Replaces Redis with a lightweight, Windows-friendly solution.

This provides:
- Job queue (tasks to be processed)
- Job outputs (real-time output lines)
- Job states (status tracking)
- Pub/Sub emulation via polling
"""
import sqlite3
import json
import time
import threading
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from queue import Queue, Empty
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Database path - in the project root
DB_PATH = Path(__file__).parent.parent.parent / "tasks.db"


def get_db_connection() -> sqlite3.Connection:
    """Get a database connection with WAL mode for better concurrency"""
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")  # Better for concurrent access
    conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the SQLite database with required tables"""
    conn = get_db_connection()
    c = conn.cursor()
    
    # Job queue table
    c.execute('''CREATE TABLE IF NOT EXISTS job_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT UNIQUE,
                action TEXT,
                config TEXT,
                created_at REAL,
                processed INTEGER DEFAULT 0)''')
    
    # Job outputs table (real-time output lines)
    c.execute('''CREATE TABLE IF NOT EXISTS job_outputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT,
                timestamp REAL,
                message_type TEXT,
                data TEXT,
                is_complete INTEGER DEFAULT 0)''')
    
    # Job states table
    c.execute('''CREATE TABLE IF NOT EXISTS job_states (
                job_id TEXT PRIMARY KEY,
                status TEXT,
                start_time REAL,
                end_time REAL,
                exit_code INTEGER,
                error TEXT)''')
    
    # Create indexes for faster queries
    c.execute('CREATE INDEX IF NOT EXISTS idx_job_outputs_job_id ON job_outputs(job_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_job_queue_processed ON job_queue(processed)')
    
    conn.commit()
    conn.close()
    
    logger.info(f"SQLite database initialized at {DB_PATH}")


class SQLiteMessageQueue:
    """
    SQLite-based message queue that provides Redis-like functionality.
    Uses polling for pub/sub emulation.
    """
    
    def __init__(self):
        init_db()
        self._subscribers: Dict[str, List[Queue]] = {}
        self._lock = threading.Lock()
    
    def push_job(self, job_id: str, action: str, config: Dict[str, Any]) -> int:
        """Push a job to the queue"""
        conn = get_db_connection()
        c = conn.cursor()
        
        try:
            c.execute("""INSERT OR REPLACE INTO job_queue 
                        (job_id, action, config, created_at, processed)
                        VALUES (?, ?, ?, ?, 0)""",
                     (job_id, action, json.dumps(config), time.time()))
            conn.commit()
            row_id = c.lastrowid
            logger.info(f"Job {job_id} pushed to queue")
            return row_id
        finally:
            conn.close()
    
    def pop_job(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Pop a job from the queue (blocking with timeout)"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            conn = get_db_connection()
            c = conn.cursor()
            
            try:
                # Get the oldest unprocessed job
                c.execute("""SELECT id, job_id, action, config 
                            FROM job_queue 
                            WHERE processed = 0 
                            ORDER BY id ASC 
                            LIMIT 1""")
                row = c.fetchone()
                
                if row:
                    # Mark as processed
                    c.execute("UPDATE job_queue SET processed = 1 WHERE id = ?", (row['id'],))
                    conn.commit()
                    
                    return {
                        'job_id': row['job_id'],
                        'action': row['action'],
                        'config': json.loads(row['config']) if row['config'] else {}
                    }
            finally:
                conn.close()
            
            time.sleep(0.1)  # Short sleep before retry
        
        return None
    
    def publish_output(self, job_id: str, message_type: str, data: Any):
        """Publish output for a job"""
        conn = get_db_connection()
        c = conn.cursor()
        
        try:
            is_complete = 1 if message_type in ['complete', 'error', 'training_end'] else 0
            
            message_data = json.dumps(data) if isinstance(data, dict) else str(data)
            
            c.execute("""INSERT INTO job_outputs 
                        (job_id, timestamp, message_type, data, is_complete)
                        VALUES (?, ?, ?, ?, ?)""",
                     (job_id, time.time(), message_type, message_data, is_complete))
            conn.commit()
            
        finally:
            conn.close()
    
    def get_new_outputs(self, job_id: str, last_id: int = 0) -> List[Dict[str, Any]]:
        """Get new outputs for a job since last_id"""
        conn = get_db_connection()
        c = conn.cursor()
        
        try:
            c.execute("""SELECT id, timestamp, message_type, data, is_complete 
                        FROM job_outputs 
                        WHERE job_id = ? AND id > ?
                        ORDER BY id ASC""",
                     (job_id, last_id))
            
            rows = c.fetchall()
            
            results = []
            for row in rows:
                try:
                    data = json.loads(row['data'])
                except (json.JSONDecodeError, TypeError):
                    data = row['data']
                
                results.append({
                    'id': row['id'],
                    'timestamp': row['timestamp'],
                    'type': row['message_type'],
                    'data': data,
                    'is_complete': bool(row['is_complete'])
                })
            
            return results
            
        finally:
            conn.close()
    
    def update_job_state(self, job_id: str, status: str, 
                         exit_code: int = None, error: str = None):
        """Update job state"""
        conn = get_db_connection()
        c = conn.cursor()
        
        try:
            if status == "running":
                c.execute("""INSERT OR REPLACE INTO job_states 
                            (job_id, status, start_time) 
                            VALUES (?, ?, ?)""",
                         (job_id, status, time.time()))
            else:
                c.execute("""UPDATE job_states SET 
                            status = ?, 
                            end_time = ?,
                            exit_code = ?,
                            error = ?
                            WHERE job_id = ?""",
                         (status, time.time(), exit_code, error, job_id))
                
                # If no rows updated, insert
                if c.rowcount == 0:
                    c.execute("""INSERT INTO job_states 
                                (job_id, status, end_time, exit_code, error)
                                VALUES (?, ?, ?, ?, ?)""",
                             (job_id, status, time.time(), exit_code, error))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def get_job_state(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job state"""
        conn = get_db_connection()
        c = conn.cursor()
        
        try:
            c.execute("SELECT * FROM job_states WHERE job_id = ?", (job_id,))
            row = c.fetchone()
            
            if row:
                return dict(row)
            return None
            
        finally:
            conn.close()
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old jobs and outputs"""
        conn = get_db_connection()
        c = conn.cursor()
        
        try:
            cutoff = time.time() - (max_age_hours * 3600)
            
            c.execute("DELETE FROM job_queue WHERE created_at < ? AND processed = 1", (cutoff,))
            c.execute("DELETE FROM job_outputs WHERE timestamp < ?", (cutoff,))
            c.execute("DELETE FROM job_states WHERE end_time IS NOT NULL AND end_time < ?", (cutoff,))
            
            conn.commit()
            logger.info(f"Cleaned up jobs older than {max_age_hours} hours")
            
        finally:
            conn.close()


class OutputSubscriber:
    """
    Async subscriber for job outputs.
    Uses polling to check for new outputs.
    """
    
    def __init__(self, job_id: str, mq: SQLiteMessageQueue):
        self.job_id = job_id
        self.mq = mq
        self.last_id = 0
        self.is_complete = False
    
    async def get_messages(self, timeout: float = 0.05) -> List[Dict[str, Any]]:
        """Get new messages (async-friendly)"""
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        messages = await loop.run_in_executor(
            None,
            self.mq.get_new_outputs,
            self.job_id,
            self.last_id
        )
        
        if messages:
            self.last_id = messages[-1]['id']
            
            # Check if job is complete
            for msg in messages:
                if msg.get('is_complete'):
                    self.is_complete = True
        
        return messages


# Global instance
_sqlite_mq: Optional[SQLiteMessageQueue] = None


def get_sqlite_queue() -> SQLiteMessageQueue:
    """Get the global SQLite message queue instance"""
    global _sqlite_mq
    
    if _sqlite_mq is None:
        _sqlite_mq = SQLiteMessageQueue()
    
    return _sqlite_mq
