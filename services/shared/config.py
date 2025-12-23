"""
Shared configuration for all services
"""
import json
import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Port configuration file path
PORT_CONFIG_PATH = PROJECT_ROOT / "port_config.json"


def load_port_config():
    """Load port configuration from JSON file"""
    if PORT_CONFIG_PATH.exists():
        try:
            with open(PORT_CONFIG_PATH, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load port config: {e}")
    
    # Return default values if config file doesn't exist or can't be loaded
    return {
        "backend_port": 8000,
        "api_service_port": 8000,
        "streamer_service_port": 8001,
        "worker_service_port": 8002,
        "frontend_dev_port": 5173,
        "frontend_preview_port": 7860
    }


# Load configuration at module level
PORT_CONFIG = load_port_config()

# Service ports
API_SERVICE_PORT = PORT_CONFIG.get("api_service_port", 8000)
STREAMER_SERVICE_PORT = PORT_CONFIG.get("streamer_service_port", 8001)
WORKER_SERVICE_PORT = PORT_CONFIG.get("worker_service_port", 8002)

# For backward compatibility
BACKEND_PORT = PORT_CONFIG.get("backend_port", 8000)
FRONTEND_DEV_PORT = PORT_CONFIG.get("frontend_dev_port", 5173)
FRONTEND_PREVIEW_PORT = PORT_CONFIG.get("frontend_preview_port", 7860)

# Redis configuration
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB = int(os.environ.get("REDIS_DB", 0))

# Job queue keys
JOB_QUEUE_KEY = "t2i_trainer:job_queue"
JOB_OUTPUT_PREFIX = "t2i_trainer:job_output:"
JOB_STATUS_PREFIX = "t2i_trainer:job_status:"

# Log directory
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Training status file
TRAINING_STATUS_FILE = LOG_DIR / "training_status.json"

# Config template directory
CONFIG_TEMPLATE_DIR = PROJECT_ROOT / "config_template"
CONFIG_TEMPLATE_DIR.mkdir(exist_ok=True)

# Custom templates directory
CUSTOM_TEMPLATES_DIR = PROJECT_ROOT / "frontend" / "public" / "custom_templates"
CUSTOM_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)


# Default training configuration
DEFAULT_CONFIG = {
    "script": "train_flux_lora_ui_kontext.py",
    "script_choices": [
        "train_longcat_edit.py",
        "train_longcat.py",
        "train_qwen_image_edit_new.py",
        "train_qwen_image_edit.py",
        "train_qwen_image.py",
        "train_flux_lora_ui_kontext_new.py",
        "train_flux_lora_ui_with_mask.py",
        "train_flux_lora_ui.py",
        "train_z_image.py"
    ],
    "output_dir": "/home/waas/kontext_output",
    "save_name": "flux-lora",
    "pretrained_model_name_or_path": "/datasets/T2ITrainer/models/flux-kontext-nf4",
    "train_data_dir": "/home/waas/kontext_test",
    "resume_from_checkpoint": None,
    "model_path": None,
    "report_to": "all",
    "rank": 16,
    "use_lokr": False,
    "rank_alpha": 1.0,
    "lokr_factor": 2,
    "train_batch_size": 1,
    "repeats": 1,
    "gradient_accumulation_steps": 1,
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,
    "optimizer": "adamw",
    "lr_scheduler": "constant",
    "learning_rate": 1e-4,
    "lr_warmup_steps": 0,
    "seed": 4321,
    "num_train_epochs": 5,
    "save_model_epochs": 1,
    "validation_epochs": 1,
    "skip_epoch": 0,
    "skip_step": 0,
    "validation_ratio": 0.1,
    "recreate_cache": False,
    "caption_dropout": 0.1,
    "config_path": "config.json",
    "resolution": "512",
    "resolution_choices": ["1024", "768", "512", "256", "1328"],
    "use_debias": False,
    "snr_gamma": 0,
    "cosine_restarts": 1,
    "max_time_steps": 0,
    "blocks_to_swap": 0,
    "mask_dropout": 0,
    "reg_ratio": 0.0,
    "reg_timestep": 0,
    "use_two_captions": False,
    "slider_positive_scale": 1.0,
    "slider_negative_scale": -1.0
}
