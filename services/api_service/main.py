"""
API Service - FastAPI server for handling frontend requests.

This service handles:
- Configuration management (save/load configs)
- Job scheduling (start/stop training jobs)
- Template management
- Health checks

It does NOT directly run training processes - instead it sends job requests
to the Worker Service via message queue.
"""
import sys
import os
import json
import time
import glob
from pathlib import Path
from typing import Dict, Any, Optional, List
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.shared.config import (
    PROJECT_ROOT, API_SERVICE_PORT, STREAMER_SERVICE_PORT,
    DEFAULT_CONFIG, LOG_DIR, CUSTOM_TEMPLATES_DIR
)
from services.shared.models import TrainingConfig, TemplateData, JobAction, BaseModel
from services.shared.message_queue import get_message_queue
from services.shared.sqlite_queue import get_sqlite_queue
from services.shared.utils import (
    save_config_to_file, load_config_from_file,
    get_script_specific_config, list_templates, load_template,
    list_custom_templates, get_training_status, update_training_status,
    generate_job_id, get_current_log_filename
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="T2I Trainer API Service",
    version="2.0.0",
    description="API service for T2I Trainer - handles configuration and job scheduling"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Message queue for inter-service communication
mq = None
sqlite_mq = None

# Track active jobs
active_jobs: Dict[str, Dict[str, Any]] = {}

# Thread pool for database operations to avoid blocking async event loop
thread_pool: Optional[ThreadPoolExecutor] = None


@app.on_event("startup")
async def startup_event():
    """Initialize message queue on startup"""
    global mq, sqlite_mq, thread_pool
    mq = get_message_queue()
    sqlite_mq = get_sqlite_queue()
    thread_pool = ThreadPoolExecutor(max_workers=4)
    logger.info(f"API Service started on port {API_SERVICE_PORT}")
    logger.info(f"WebSocket Streamer available at ws://localhost:{STREAMER_SERVICE_PORT}/ws/{{job_id}}")
    logger.info(f"Using SQLite queue for inter-service communication")
    
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up resources on shutdown"""
        global thread_pool
        if thread_pool:
            thread_pool.shutdown(wait=True)
        logger.info("API Service shutdown complete")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "api_service",
        "message": "T2I Trainer API Service is running",
        "streamer_url": f"ws://localhost:{STREAMER_SERVICE_PORT}/ws"
    }


@app.get("/api/default_config")
async def get_default_config():
    """Get the default configuration"""
    return {
        "status": "success",
        "config": DEFAULT_CONFIG
    }


@app.post("/api/load_config")
async def load_config_endpoint(config_path: str):
    """Load an existing configuration from file"""
    try:
        config_data = load_config_from_file(config_path)
        return {
            "status": "success",
            "config": config_data
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in config file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/save_config")
async def save_config(config: TrainingConfig):
    """Save configuration to a file without running training"""
    try:
        config_dict = config.model_dump()
        config_path = save_config_to_file(config_dict)
        return {
            "status": "success",
            "message": f"Configuration saved to {config_path}",
            "config_path": config_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/start_training")
async def start_training(config: TrainingConfig):
    """
    Start a training job.
    
    This endpoint:
    1. Saves the configuration
    2. Generates a job ID
    3. Sends a job request to the Worker Service via SQLite queue
    4. Returns the job ID for WebSocket connection
    """
    global sqlite_mq
    
    try:
        # Save configuration
        config_dict = config.model_dump()
        config_path = save_config_to_file(config_dict)
        
        # Generate job ID
        job_id = generate_job_id()
        
        # Run the database operation in a thread pool to avoid blocking the async event loop
        loop = asyncio.get_event_loop()
        
        # Push job to SQLite queue
        await loop.run_in_executor(thread_pool, sqlite_mq.push_job,
            job_id,
            JobAction.START.value,
            {
                "script": config.script,
                "config_path": config_path
            })
        
        # Track job
        active_jobs[job_id] = {
            "status": "queued",
            "config": config_dict,
            "started_at": time.time()
        }
        
        logger.info(f"Training job {job_id} queued successfully (SQLite)")
        
        return {
            "status": "success",
            "job_id": job_id,
            "message": f"Training job queued. Connect to ws://localhost:{STREAMER_SERVICE_PORT}/ws/{job_id} for output.",
            "websocket_url": f"ws://localhost:{STREAMER_SERVICE_PORT}/ws/{job_id}"
        }
        
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stop_training")
async def stop_training(job_id: Optional[str] = None):
    """
    Stop a training job.
    
    If job_id is not provided, stops the most recent job.
    """
    global sqlite_mq
    
    try:
        # If no job_id provided, get the most recent one
        if not job_id and active_jobs:
            job_id = max(active_jobs.keys(), key=lambda k: active_jobs[k].get("started_at", 0))
        
        if not job_id:
            return {"status": "error", "message": "No active training job found"}
        
        # Run the database operation in a thread pool to avoid blocking the async event loop
        loop = asyncio.get_event_loop()
        
        # Push stop request to SQLite queue
        await loop.run_in_executor(thread_pool, sqlite_mq.push_job,
            job_id,
            JobAction.STOP.value,
            {})
        
        # Update local tracking
        if job_id in active_jobs:
            active_jobs[job_id]["status"] = "stopping"
        
        update_training_status("stopped")
        
        logger.info(f"Stop request sent for job {job_id} (SQLite)")
        
        return {
            "status": "success",
            "job_id": job_id,
            "message": f"Stop request sent for job {job_id}"
        }
        
    except Exception as e:
        logger.error(f"Failed to stop training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/job_status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a specific job"""
    if job_id in active_jobs:
        return {
            "status": "success",
            "job": active_jobs[job_id]
        }
    return {
        "status": "error",
        "message": f"Job {job_id} not found"
    }


@app.get("/api/active_jobs")
async def get_active_jobs():
    """Get all active jobs"""
    return {
        "status": "success",
        "jobs": active_jobs
    }


@app.get("/api/get_training_status")
async def get_training_status_endpoint():
    """Get the current training status"""
    try:
        status = get_training_status()
        return {"status": "success", "training_status": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reset_training_status")
async def reset_training_status():
    """Reset the training status to idle"""
    update_training_status("idle")
    return {"status": "success", "message": "Training status reset to idle"}


@app.post("/api/change_script")
async def change_script_endpoint(script_data: Dict[str, Any]):
    """Handle script change - inject required config sections for specific scripts"""
    selected_script = script_data.get("script", "")
    config_path = script_data.get("config_path", "config.json")
    
    script_specific_config = get_script_specific_config(selected_script, config_path)
    
    return {
        "status": "success",
        "config": script_specific_config
    }


@app.get("/api/list_templates")
async def list_templates_endpoint(config_path: str = "config.json"):
    """List available templates"""
    templates = list_templates(config_path)
    return {
        "status": "success",
        "templates": templates
    }


@app.post("/api/load_template")
async def load_template_endpoint(template_data: Dict[str, Any]):
    """Load a template configuration"""
    template_name = template_data.get("template_name", "")
    config_path = template_data.get("config_path", "config.json")
    
    template_config = load_template(template_name, config_path)
    
    return {
        "status": "success",
        "config": template_config
    }


@app.get("/api/available_scripts")
async def get_available_scripts():
    """Get list of available training scripts"""
    script_patterns = ["train_*.py"]
    available_scripts = []
    
    for pattern in script_patterns:
        scripts = glob.glob(str(PROJECT_ROOT / pattern))
        available_scripts.extend([os.path.basename(s) for s in scripts])
    
    return {
        "status": "success",
        "scripts": list(set(available_scripts))
    }


# Custom template endpoints
@app.post("/api/templates/custom")
async def save_custom_template(template_data: TemplateData):
    """Save a custom template"""
    try:
        filename = template_data.name
        if not filename.endswith('.json'):
            filename = f"{filename}.json"
        
        template_path = CUSTOM_TEMPLATES_DIR / filename
        
        workflow_data = {
            **template_data.workflow,
            "timestamp": time.time(),
            "version": "1.0"
        }
        
        with open(template_path, 'w', encoding='utf-8') as f:
            json.dump(workflow_data, f, indent=2, ensure_ascii=False)
        
        return {"status": "success", "message": f"Template '{filename}' saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workflow/current")
async def save_current_workflow(workflow_data: Dict[str, Any] = Body(...)):
    """Save the current workflow to a specific location"""
    try:
        # Define the current workflow path
        current_workflow_path = PROJECT_ROOT / "frontend" / "public" / "current" / "workflow.json"
        
        # Ensure the directory exists
        current_workflow_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        workflow_with_metadata = {
            **workflow_data,
            "timestamp": time.time(),
            "version": "1.0"
        }
        
        # Write the workflow data to the file
        with open(current_workflow_path, 'w', encoding='utf-8') as f:
            json.dump(workflow_with_metadata, f, indent=2, ensure_ascii=False)
        
        return {"status": "success", "message": "Current workflow saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workflow/current")
async def load_current_workflow():
    """Load the current workflow from the specific location"""
    try:
        # Define the current workflow path
        current_workflow_path = PROJECT_ROOT / "frontend" / "public" / "current" / "workflow.json"
        
        if not current_workflow_path.exists():
            raise HTTPException(status_code=404, detail="Current workflow file not found")
        
        with open(current_workflow_path, 'r', encoding='utf-8') as f:
            workflow_data = json.load(f)
        
        return {"status": "success", "workflow": workflow_data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/templates/custom")
async def get_custom_templates_list():
    """Get list of custom templates"""
    try:
        templates = list_custom_templates()
        return {"templates": templates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/templates/custom/{template_name}")
async def delete_custom_template(template_name: str):
    """Delete a custom template"""
    try:
        template_path = CUSTOM_TEMPLATES_DIR / template_name
        
        if template_path.exists():
            os.remove(template_path)
            return {"status": "success", "message": f"Template '{template_name}' deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Template not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# New endpoint to save template to any path
class TemplateSaveData(BaseModel):
    path: str
    workflow: Dict[str, Any]

@app.post("/api/templates/save_to_path")
async def save_template_to_path(template_save_data: TemplateSaveData):
    """Save a template to any specified path"""
    try:
        # Validate that the path is within allowed directories to prevent security issues
        template_path = Path(template_save_data.path)
        
        # Join the relative path with the project root to get the full path
        abs_path = (PROJECT_ROOT / template_path).resolve()
        
        # Ensure the path is within allowed directories (PROJECT_ROOT)
        project_root = PROJECT_ROOT.resolve()
        if not str(abs_path).startswith(str(project_root)):
            raise HTTPException(status_code=400, detail="Path is outside allowed directories")
        
        # Ensure the directory exists
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        workflow_data = {
            **template_save_data.workflow,
            "timestamp": time.time(),
            "version": "1.0"
        }
        
        # Write the template to the specified path
        with open(abs_path, 'w', encoding='utf-8') as f:
            json.dump(workflow_data, f, indent=2, ensure_ascii=False)
        
        return {"status": "success", "message": f"Template saved to '{template_save_data.path}' successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/get_log")
async def get_log(date: str = None):
    """Retrieve training log for a specific date or today's log"""
    try:
        from datetime import datetime
        
        if date:
            try:
                datetime.strptime(date, "%Y-%m-%d")
                log_file = LOG_DIR / f"training_log_{date}.json"
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        else:
            log_file = get_current_log_filename()
        
        if log_file.exists():
            with open(log_file, 'r') as f:
                try:
                    logs = json.load(f)
                    return {"status": "success", "logs": logs, "log_file": str(log_file)}
                except json.JSONDecodeError:
                    return {"status": "success", "logs": [], "log_file": str(log_file), "message": "Log file is empty"}
        else:
            return {"status": "success", "logs": [], "log_file": str(log_file), "message": "Log file does not exist"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Backward compatibility endpoints
@app.post("/api/run")
async def run_training(config: TrainingConfig):
    """
    Legacy endpoint for backward compatibility.
    Redirects to start_training.
    """
    return await start_training(config)


def main():
    """Run the API service"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=API_SERVICE_PORT)


if __name__ == "__main__":
    main()
