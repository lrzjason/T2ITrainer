"""
Worker Service - Executes training subprocesses and publishes output.

This service:
- Listens to the job queue for new training requests
- Executes training scripts as subprocesses
- Captures and publishes subprocess output to message queue (Redis Pub/Sub)
- Handles job lifecycle (start, stop, status)
- Manages multiple concurrent jobs (with limits)

This is a synchronous service that runs independently from the async API/WebSocket services.
"""
import sys
import os
import json
import time
import threading
import subprocess
import signal
from pathlib import Path
from typing import Dict, Optional, Any
from queue import Queue, Empty
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.shared.config import (
    PROJECT_ROOT, JOB_QUEUE_KEY, JOB_OUTPUT_PREFIX, JOB_STATUS_PREFIX,
    LOG_DIR
)
from services.shared.message_queue import get_message_queue
from services.shared.sqlite_queue import get_sqlite_queue
from services.shared.utils import (
    update_training_status, init_log_file, append_to_log, get_current_log_filename
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Maximum concurrent jobs
MAX_CONCURRENT_JOBS = 1  # Training is resource-intensive, usually only 1 at a time


class JobRunner:
    """Manages a single training job"""
    
    def __init__(self, job_id: str, script: str, config_path: str, sqlite_mq):
        self.job_id = job_id
        self.script = script
        self.config_path = config_path
        self.sqlite_mq = sqlite_mq
        
        self.process: Optional[subprocess.Popen] = None
        self.output_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.started_at = time.time()
        self.completed_at: Optional[float] = None
        self.return_code: Optional[int] = None
        
        # Output channel for this job
        self.output_channel = f"{JOB_OUTPUT_PREFIX}{job_id}"
        
        # Log file for this session
        self.log_file = get_current_log_filename()
        init_log_file(self.log_file)
    
    def start(self) -> bool:
        """Start the training subprocess"""
        try:
            # Validate script exists
            script_path = PROJECT_ROOT / self.script
            if not script_path.exists():
                self._publish_error(f"Training script not found: {self.script}")
                return False
            
            # Resolve config path
            if os.path.isabs(self.config_path):
                config_path = self.config_path
            else:
                config_path = str(PROJECT_ROOT / self.config_path)
            
            # Build command
            cmd = [sys.executable, str(script_path), "--config_path", config_path]
            
            # Set up environment for unbuffered output
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            
            logger.info(f"[{self.job_id}] Starting: {' '.join(cmd)}")
            
            # Start subprocess
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                cwd=str(PROJECT_ROOT)
            )
            
            # Start output reader thread
            self.output_thread = threading.Thread(
                target=self._read_output,
                daemon=True
            )
            self.output_thread.start()
            
            # Publish start message
            self._publish_message({
                "type": "status",
                "status": "running",
                "message": f"Training started with PID {self.process.pid}",
                "pid": self.process.pid
            })
            
            update_training_status("running")
            
            return True
            
        except Exception as e:
            logger.error(f"[{self.job_id}] Failed to start: {e}")
            self._publish_error(str(e))
            return False
    
    def stop(self):
        """Stop the training subprocess"""
        logger.info(f"[{self.job_id}] Stop requested")
        
        self.stop_event.set()
        
        if self.process and self.process.poll() is None:
            logger.info(f"[{self.job_id}] Terminating process {self.process.pid}")
            
            # Try graceful termination first
            self.process.terminate()
            
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"[{self.job_id}] Force killing process")
                self.process.kill()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            
            self._publish_message({
                "type": "complete",
                "status": "stopped",
                "return_code": -1,
                "message": "Training stopped by user"
            })
            
            update_training_status("stopped")
    
    def wait(self):
        """Wait for the job to complete"""
        if self.output_thread:
            self.output_thread.join()
    
    def is_running(self) -> bool:
        """Check if the job is still running"""
        return self.process is not None and self.process.poll() is None
    
    def _read_output(self):
        """Read subprocess output and publish to message queue"""
        try:
            while self.process.poll() is None and not self.stop_event.is_set():
                # Read a line with a timeout to allow checking stop event
                try:
                    line = self.process.stdout.readline()
                    if not line:  # EOF reached
                        break
                        
                    if self.stop_event.is_set():
                        break
                        
                    line = line.rstrip('\n\r')
                        
                    if not line.strip():
                        continue
                        
                    # Check for training end marker
                    if "=========== End Training ==========" in line:
                        self._publish_message({
                            "type": "training_end",
                            "data": "Training completed successfully"
                        })
                        
                    # Publish output line
                    self._publish_message({
                        "type": "output",
                        "data": line.replace("\r", "\n")
                    })
                        
                    # Log the output
                    self._log_entry({
                        "type": "output",
                        "data": line.replace("\r", "\n")
                    })
                except:
                    # If readline fails, check if we should stop
                    if self.stop_event.is_set():
                        break
                    time.sleep(0.01)  # Small delay to prevent busy loop
                
            # At this point, either process finished or stop was requested
            if not self.stop_event.is_set():
                # Process completed normally
                self.return_code = self.process.wait()
            else:
                # Stop was requested, terminate the process
                if self.process.poll() is None:  # Process still running
                    logger.info(f"[{self.job_id}] Terminating process due to stop request")
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"[{self.job_id}] Force killing process after stop request")
                        self.process.kill()
                        try:
                            self.process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            pass
                self.return_code = -1  # Indicate stopped by user
                
            self.completed_at = time.time()
                
            # Publish completion message
            if self.stop_event.is_set():
                status = "stopped"
                message = "Training stopped by user"
            else:
                status = "success" if self.return_code == 0 else "error"
                message = "Training completed successfully" if self.return_code == 0 \
                          else f"Training failed with return code {self.return_code}"
                
            self._publish_message({
                "type": "complete",
                "status": status,
                "return_code": self.return_code,
                "message": message
            })
                
            # Log completion
            self._log_entry({
                "type": "completion",
                "status": status,
                "return_code": self.return_code
            })
                
            update_training_status("completed" if self.return_code == 0 else ("stopped" if self.stop_event.is_set() else "failed"))
                
            # Update job state in SQLite
            final_status = "completed" if self.return_code == 0 else ("stopped" if self.stop_event.is_set() else "failed")
            self.sqlite_mq.update_job_state(
                self.job_id, 
                final_status,
                exit_code=self.return_code
            )
                
            logger.info(f"[{self.job_id}] Completed with return code {self.return_code}")
                
        except Exception as e:
            logger.error(f"[{self.job_id}] Error reading output: {e}")
            self._publish_error(str(e))
            update_training_status("error")
    
    def _publish_message(self, data: Dict[str, Any]):
        """Publish a message to the job's output channel using SQLite"""
        message = {
            "job_id": self.job_id,
            "timestamp": time.time(),
            **data
        }
        message_type = data.get("type", "output")
        self.sqlite_mq.publish_output(self.job_id, message_type, message)
    
    def _publish_error(self, error: str):
        """Publish an error message"""
        self._publish_message({
            "type": "error",
            "status": "error",
            "message": error
        })
        # Also update job state in SQLite
        self.sqlite_mq.update_job_state(self.job_id, "failed", error=error)
    
    def _log_entry(self, data: Dict[str, Any]):
        """Append entry to log file"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "job_id": self.job_id,
            **data
        }
        append_to_log(self.log_file, entry)


class WorkerService:
    """Main worker service that processes job queue"""
    
    def __init__(self):
        self.mq = get_message_queue()  # Keep for fallback
        self.sqlite_mq = get_sqlite_queue()
        self.running_jobs: Dict[str, JobRunner] = {}
        self.stop_event = threading.Event()
        self.cleanup_thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start the worker service"""
        logger.info("Worker Service starting...")
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_completed_jobs,
            daemon=True
        )
        self.cleanup_thread.start()
        
        # Main job processing loop
        self._process_job_queue()
    
    def stop(self):
        """Stop the worker service"""
        logger.info("Worker Service stopping...")
        
        self.stop_event.set()
        
        # Stop all running jobs
        for job_id, runner in list(self.running_jobs.items()):
            runner.stop()
            runner.wait()
    
    def _process_job_queue(self):
        """Main loop: listen for job messages and process them using SQLite"""
        logger.info("Listening for jobs via SQLite queue")
        
        while not self.stop_event.is_set():
            try:
                # Pop job from SQLite queue
                job_message = self.sqlite_mq.pop_job(timeout=1.0)
                
                if job_message is None:
                    continue
                
                job_id = job_message.get("job_id")
                action = job_message.get("action")
                config = job_message.get("config", {})
                
                logger.info(f"Received job: {job_id}, action: {action}")
                
                if action == "start":
                    self._handle_start(job_id, config)
                elif action == "stop":
                    self._handle_stop(job_id)
                elif action == "status":
                    self._handle_status(job_id)
                else:
                    logger.warning(f"Unknown action: {action}")
                    
            except Exception as e:
                logger.error(f"Error processing job queue: {e}")
                time.sleep(1)  # Avoid tight loop on error
    
    def _handle_start(self, job_id: str, config: Dict[str, Any]):
        """Handle job start request"""
        # Check concurrent job limit
        active_count = sum(1 for r in self.running_jobs.values() if r.is_running())
        
        if active_count >= MAX_CONCURRENT_JOBS:
            logger.warning(f"Cannot start job {job_id}: max concurrent jobs reached ({MAX_CONCURRENT_JOBS})")
            
            # Publish error to the job's channel
            mq = get_message_queue()
            channel = f"{JOB_OUTPUT_PREFIX}{job_id}"
            mq.publish(channel, {
                "job_id": job_id,
                "type": "error",
                "status": "rejected",
                "message": f"Maximum concurrent jobs ({MAX_CONCURRENT_JOBS}) reached. Please wait.",
                "timestamp": time.time()
            })
            return
        
        # Create job runner
        script = config.get("script", "train_flux_lora_ui_kontext_new.py")
        config_path = config.get("config_path", "config.json")
        
        runner = JobRunner(job_id, script, config_path, self.sqlite_mq)
        
        # Update job state in SQLite
        self.sqlite_mq.update_job_state(job_id, "running")
        
        if runner.start():
            self.running_jobs[job_id] = runner
        else:
            logger.error(f"Failed to start job {job_id}")
    
    def _handle_stop(self, job_id: str):
        """Handle job stop request"""
        if job_id in self.running_jobs:
            self.running_jobs[job_id].stop()
            # Update job state in SQLite
            self.sqlite_mq.update_job_state(job_id, "stopped")
        else:
            logger.warning(f"Job {job_id} not found for stop request")
            # Still update state in case job was queued but not started
            self.sqlite_mq.update_job_state(job_id, "stopped")
    
    def _handle_status(self, job_id: str):
        """Handle job status request"""
        if job_id in self.running_jobs:
            runner = self.running_jobs[job_id]
            status = "running" if runner.is_running() else "completed"
            
            # Publish status via SQLite
            self.sqlite_mq.publish_output(job_id, "status", {
                "job_id": job_id,
                "type": "status",
                "status": status,
                "started_at": runner.started_at,
                "completed_at": runner.completed_at,
                "return_code": runner.return_code,
                "timestamp": time.time()
            })
        else:
            # Get status from SQLite if not in memory
            job_state = self.sqlite_mq.get_job_state(job_id)
            status = job_state.get("status", "not_found") if job_state else "not_found"
            
            # Publish status via SQLite
            self.sqlite_mq.publish_output(job_id, "status", {
                "job_id": job_id,
                "type": "status",
                "status": status,
                "timestamp": time.time()
            })
    
    def _cleanup_completed_jobs(self):
        """Periodically clean up completed jobs from memory"""
        while not self.stop_event.is_set():
            time.sleep(60)  # Check every minute
            
            # Find completed jobs older than 5 minutes
            cutoff = time.time() - 300
            
            for job_id in list(self.running_jobs.keys()):
                runner = self.running_jobs[job_id]
                
                if not runner.is_running() and runner.completed_at:
                    if runner.completed_at < cutoff:
                        logger.info(f"Cleaning up completed job: {job_id}")
                        del self.running_jobs[job_id]


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    if hasattr(signal_handler, 'worker'):
        signal_handler.worker.stop()
    sys.exit(0)


def run_direct_training(script_path: str, config_path: str, job_id: str):
    """
    Run training directly and return a subprocess.Popen object.
    This is used for the legacy /ws/training endpoint.
    """
    import subprocess
    import sys
    from services.shared.config import PROJECT_ROOT
    
    # Validate script exists
    script_full_path = PROJECT_ROOT / script_path
    if not script_full_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")
    
    # Resolve config path
    import os
    if os.path.isabs(config_path):
        full_config_path = config_path
    else:
        full_config_path = str(PROJECT_ROOT / config_path)
    
    # Build command
    cmd = [sys.executable, str(script_full_path), "--config_path", full_config_path]
    
    # Set up environment for unbuffered output
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    # Start subprocess
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
        cwd=str(PROJECT_ROOT)
    )
    
    return process


def main():
    """Run the Worker service"""
    logger.info("=" * 50)
    logger.info("T2I Trainer Worker Service")
    logger.info("=" * 50)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start worker
    worker = WorkerService()
    signal_handler.worker = worker
    
    try:
        worker.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        worker.stop()
    
    logger.info("Worker Service stopped")


if __name__ == "__main__":
    main()
