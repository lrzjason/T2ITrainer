"""
Shared Pydantic models for API requests/responses
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
from enum import Enum


class JobStatus(str, Enum):
    """Job status enumeration"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class JobAction(str, Enum):
    """Job action enumeration"""
    START = "start"
    STOP = "stop"
    STATUS = "status"


class TrainingConfig(BaseModel):
    """Pydantic model for training configuration"""
    script: str = "train_flux_lora_ui_kontext.py"
    config_path: str = "config.json"
    seed: Optional[int] = 4321
    mixed_precision: Optional[str] = "bf16"
    report_to: Optional[str] = "all"
    lr_warmup_steps: Optional[int] = 0
    output_dir: Optional[str] = None
    save_name: Optional[str] = "flux-lora"
    train_data_dir: Optional[str] = None
    optimizer: Optional[str] = "adamw"
    lr_scheduler: Optional[str] = "constant"
    learning_rate: Optional[float] = 1e-4
    train_batch_size: Optional[int] = 1
    repeats: Optional[int] = 1
    gradient_accumulation_steps: Optional[int] = 1
    num_train_epochs: Optional[int] = 5
    save_model_epochs: Optional[int] = 1
    validation_epochs: Optional[int] = 1
    rank: Optional[int] = 16
    skip_epoch: Optional[int] = 0
    skip_step: Optional[int] = 0
    gradient_checkpointing: Optional[bool] = True
    validation_ratio: Optional[float] = 0.1
    pretrained_model_name_or_path: Optional[str] = None
    model_path: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None
    recreate_cache: Optional[bool] = False
    resolution: Optional[str] = "512"
    caption_dropout: Optional[float] = 0.1
    cosine_restarts: Optional[int] = 1
    max_time_steps: Optional[int] = 0
    blocks_to_swap: Optional[int] = 0
    mask_dropout: Optional[float] = 0
    reg_ratio: Optional[float] = 0.0
    reg_timestep: Optional[int] = 0
    use_two_captions: Optional[bool] = False
    slider_positive_scale: Optional[float] = 1.0
    slider_negative_scale: Optional[float] = -1.0
    use_lokr: Optional[bool] = False
    rank_alpha: Optional[float] = 1.0
    lokr_factor: Optional[int] = 2
    image_configs: Optional[Dict[str, Any]] = {}
    caption_configs: Optional[Dict[str, Any]] = {}
    training_set: Any = None
    dataset_configs: Optional[List[Dict[str, Any]]] = []


class JobRequest(BaseModel):
    """Request to start a training job"""
    job_id: Optional[str] = None
    config: TrainingConfig
    

class JobMessage(BaseModel):
    """Message for job queue"""
    job_id: str
    action: JobAction
    config: Optional[Dict[str, Any]] = None
    timestamp: float = 0


class OutputMessage(BaseModel):
    """Output message from worker to streamer"""
    job_id: str
    type: str  # "output", "error", "complete", "status"
    data: Optional[str] = None
    status: Optional[str] = None
    return_code: Optional[int] = None
    message: Optional[str] = None
    timestamp: float = 0


class TemplateData(BaseModel):
    """Template data for saving custom templates"""
    name: str
    workflow: Dict[str, Any]


class JobStatusResponse(BaseModel):
    """Response for job status query"""
    job_id: str
    status: JobStatus
    message: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
