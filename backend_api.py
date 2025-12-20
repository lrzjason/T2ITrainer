from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
import subprocess
import sys
import json
import os
import glob
from pathlib import Path
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import threading
import time
import queue

app = FastAPI(title="T2I Trainer Backend API", version="1.0.0")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define default configuration that matches ui_flux_fill.py
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


class TrainingConfig(BaseModel):
    """Pydantic model for training configuration"""
    script: str = DEFAULT_CONFIG["script"]
    config_path: str = DEFAULT_CONFIG["config_path"]
    seed: Optional[int] = DEFAULT_CONFIG["seed"]
    mixed_precision: Optional[str] = DEFAULT_CONFIG["mixed_precision"]
    report_to: Optional[str] = DEFAULT_CONFIG["report_to"]
    lr_warmup_steps: Optional[int] = DEFAULT_CONFIG["lr_warmup_steps"]
    output_dir: Optional[str] = DEFAULT_CONFIG["output_dir"]
    save_name: Optional[str] = DEFAULT_CONFIG["save_name"]
    train_data_dir: Optional[str] = DEFAULT_CONFIG["train_data_dir"]
    optimizer: Optional[str] = DEFAULT_CONFIG["optimizer"]
    lr_scheduler: Optional[str] = DEFAULT_CONFIG["lr_scheduler"]
    learning_rate: Optional[float] = DEFAULT_CONFIG["learning_rate"]
    train_batch_size: Optional[int] = DEFAULT_CONFIG["train_batch_size"]
    repeats: Optional[int] = DEFAULT_CONFIG["repeats"]
    gradient_accumulation_steps: Optional[int] = DEFAULT_CONFIG["gradient_accumulation_steps"]
    num_train_epochs: Optional[int] = DEFAULT_CONFIG["num_train_epochs"]
    save_model_epochs: Optional[int] = DEFAULT_CONFIG["save_model_epochs"]
    validation_epochs: Optional[int] = DEFAULT_CONFIG["validation_epochs"]
    rank: Optional[int] = DEFAULT_CONFIG["rank"]
    skip_epoch: Optional[int] = DEFAULT_CONFIG["skip_epoch"]
    skip_step: Optional[int] = DEFAULT_CONFIG["skip_step"]
    gradient_checkpointing: Optional[bool] = DEFAULT_CONFIG["gradient_checkpointing"]
    validation_ratio: Optional[float] = DEFAULT_CONFIG["validation_ratio"]
    pretrained_model_name_or_path: Optional[str] = DEFAULT_CONFIG["pretrained_model_name_or_path"]
    model_path: Optional[str] = DEFAULT_CONFIG["model_path"]
    resume_from_checkpoint: Optional[str] = DEFAULT_CONFIG["resume_from_checkpoint"]
    recreate_cache: Optional[bool] = DEFAULT_CONFIG["recreate_cache"]
    resolution: Optional[str] = DEFAULT_CONFIG["resolution"]
    caption_dropout: Optional[float] = DEFAULT_CONFIG["caption_dropout"]
    cosine_restarts: Optional[int] = DEFAULT_CONFIG["cosine_restarts"]
    max_time_steps: Optional[int] = DEFAULT_CONFIG["max_time_steps"]
    blocks_to_swap: Optional[int] = DEFAULT_CONFIG["blocks_to_swap"]
    mask_dropout: Optional[float] = DEFAULT_CONFIG["mask_dropout"]
    reg_ratio: Optional[float] = DEFAULT_CONFIG["reg_ratio"]
    reg_timestep: Optional[int] = DEFAULT_CONFIG["reg_timestep"]
    use_two_captions: Optional[bool] = DEFAULT_CONFIG["use_two_captions"]
    slider_positive_scale: Optional[float] = DEFAULT_CONFIG["slider_positive_scale"]
    slider_negative_scale: Optional[float] = DEFAULT_CONFIG["slider_negative_scale"]
    use_lokr: Optional[bool] = DEFAULT_CONFIG["use_lokr"]
    rank_alpha: Optional[float] = DEFAULT_CONFIG["rank_alpha"]
    lokr_factor: Optional[int] = DEFAULT_CONFIG["lokr_factor"]
    image_configs: Optional[Dict[str, Any]] = {}
    caption_configs: Optional[Dict[str, Any]] = {}
    training_set: Any = None
    dataset_configs: Optional[List[Dict[str, Any]]] = []

def get_script_config_mapping():
    """Define mapping between scripts and their required config sections"""
    return {
        "train_qwen_image.py": {
            "type": "old",
            "preset": "preset_0_single.json",
            "mandatory": {"image_configs", "caption_configs", "training_set"}
        },
        "train_qwen_image_edit.py": {
            "type": "old",
            "preset": "preset_0_single.json",
            "mandatory": {"image_configs", "caption_configs", "training_set"}
        },
        "train_flux_lora_ui_kontext_new.py": {
            "type": "old",
            "preset": "preset_0_single.json",
            "mandatory": {"image_configs", "caption_configs", "training_set"}
        },
        "train_qwen_image_edit_new.py": {
            "type": "new",
            "preset": "preset_0_single.json",
            "mandatory": {"dataset_configs"}
        },
        "train_longcat_edit.py": {
            "type": "new",
            "preset": "preset_0_single.json",
            "mandatory": {"dataset_configs"}
        },
        "train_longcat.py": {
            "type": "new",
            "preset": "preset_0_single.json",
            "mandatory": {"dataset_configs"}
        },
    }


def get_template_dir():
    """Get template directory path"""
    template_dir = "./config_template"
    os.makedirs(template_dir, exist_ok=True)
    return template_dir


def list_templates(config_path: str):
    """List available templates"""
    template_dir = get_template_dir()
    templates = [os.path.basename(f) for f in glob.glob(os.path.join(template_dir, "*.json"))]
    templates.sort()
    # Add config_path as a choice as well
    config_name = os.path.basename(config_path) if os.path.isfile(config_path) else "config.json"
    choices = [config_name] + templates
    return choices


def load_template(template_name: str, config_path: str):
    """
    Load a template, merging missing sections into current config
    """
    if isinstance(template_name, list):
        template_name = template_name[0] if template_name else ""
    if not template_name:
        return {}

    tpl = {}
    config_name = os.path.basename(config_path)

    if template_name == config_name and os.path.isfile(config_path):
        # Load from current config file
        with open(config_path, "r", encoding="utf-8") as f:
            tpl = json.load(f)
    else:
        # Load from template file
        tpl_path = os.path.join(get_template_dir(), template_name)
        if not os.path.isfile(tpl_path):
            return {}

        with open(tpl_path, "r", encoding="utf-8") as f:
            tpl = json.load(f)

    # Start with default config
    current = {}
    # Only merge missing sections from template
    for key in ("image_configs", "caption_configs", "training_set", "dataset_configs"):
        if key in tpl and key not in current:
            current[key] = tpl[key]

    return current


def get_script_specific_config(selected_script: str, config_path: str = "config.json"):
    """
    Get config with script-specific requirements
    """
    pairs = get_script_config_mapping()

    if selected_script not in pairs:
        return {}

    preset_config = pairs[selected_script]
    preset_name = preset_config["preset"]
    preset_path = os.path.join(get_template_dir(), preset_name)

    editor = {}
    preset = {}
    mandatory = preset_config["mandatory"]

    # Try to load from config_path first, then from preset
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                preset = json.load(f)
        except Exception:
            pass  # ignore read/parse errors
    else:
        try:
            with open(preset_path, "r", encoding="utf-8") as f:
                preset = json.load(f)
        except Exception:
            print(f"Error reading {preset_path}")

    # Merge mandatory sections from preset
    for key in mandatory:
        if key in preset:
            editor[key] = preset[key]

    return editor


def save_config_to_file(config: TrainingConfig) -> str:
    """Save the training configuration to a JSON file"""
    config_path = config.config_path
    config_dir = os.path.dirname(config_path) or "."
    os.makedirs(config_dir, exist_ok=True)

    # Get the current configuration from file if it exists
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}

    # Update with the new configuration values from the model
    ui_vals = {
        "config_path": config.config_path,
        "script": config.script,
        "seed": config.seed,
        "mixed_precision": config.mixed_precision,
        "report_to": config.report_to,
        "lr_warmup_steps": config.lr_warmup_steps,
        "output_dir": config.output_dir,
        "save_name": config.save_name,
        "train_data_dir": config.train_data_dir,
        "optimizer": config.optimizer,
        "lr_scheduler": config.lr_scheduler,
        "learning_rate": config.learning_rate,
        "train_batch_size": config.train_batch_size,
        "repeats": config.repeats,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_train_epochs": config.num_train_epochs,
        "save_model_epochs": config.save_model_epochs,
        "validation_epochs": config.validation_epochs,
        "rank": config.rank,
        "use_lokr": config.use_lokr,
        "rank_alpha": config.rank_alpha,
        "lokr_factor": config.lokr_factor,
        "skip_epoch": config.skip_epoch,
        "skip_step": config.skip_step,
        "gradient_checkpointing": config.gradient_checkpointing,
        "validation_ratio": config.validation_ratio,
        "pretrained_model_name_or_path": config.pretrained_model_name_or_path,
        "model_path": config.model_path,
        "resume_from_checkpoint": config.resume_from_checkpoint,
        "recreate_cache": config.recreate_cache,
        "resolution": config.resolution,
        "caption_dropout": config.caption_dropout,
        "cosine_restarts": config.cosine_restarts,
        "max_time_steps": config.max_time_steps,
        "blocks_to_swap": config.blocks_to_swap,
        "dataset_configs": config.dataset_configs,
        "image_configs": config.image_configs,
        "caption_configs": config.caption_configs,
        "training_set": config.training_set,
    }
    cfg.update(ui_vals)

    # If json_editor is provided, update with its content (for template/JSON editing)
    # if config.json_editor:
    #     try:
    #         editor_partial = json.loads(config.json_editor, strict=False)
    #         # Update specific sections from the JSON editor
    #         for key in ("image_configs", "caption_configs", "training_set", "dataset_configs"):
    #             if key in editor_partial:
    #                 cfg[key] = editor_partial[key]
    #     except json.JSONDecodeError:
    #         pass  # If JSON is invalid, ignore the editor content

    # Write to file
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)

    # Also save to default config.json
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)

    return config_path


def run_training_script(script_path: str, config_path: str) -> dict:
    """Run a training script with the provided configuration file"""
    # Validate that the training script exists
    script_path_obj = Path(script_path)
    if not script_path_obj.exists():
        # Check in current directory if not found as full path
        script_file = Path(script_path_obj.name)
        if not script_file.exists():
            raise FileNotFoundError(f"Training script {script_path} not found")
        script_path = script_file.name

    # Build the command to run the training script
    cmd = [sys.executable, script_path, "--config_path", config_path]

    try:
        # Execute the training script in a separate thread to avoid blocking
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        return {
            "status": "success",
            "message": "Training completed successfully",
            "command": " ".join(cmd),
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        raise Exception("Training process timed out")
    except subprocess.CalledProcessError as e:
        raise Exception(f"Training script failed with return code {e.returncode}: {e.stderr}")


@app.post("/api/run")
async def run_training(config: TrainingConfig):
    """Run the training process with the provided configuration"""
    try:
        # Save configuration to file first
        config_path = save_config_to_file(config)

        # Run the training script
        result = run_training_script(config.script, config_path)

        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Training process timed out")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/save_config")
async def save_config(config: TrainingConfig):
    """Save configuration to a file without running training"""
    try:
        config_path = save_config_to_file(config)
        return {
            "status": "success",
            "message": f"Configuration saved to {config_path}",
            "config_path": config_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def load_config_from_file(config_path: str):
    """Load an existing configuration from file, with fallback to default config"""
    if not os.path.isfile(config_path):
        # If config file doesn't exist, create it with default config
        save_config_to_file(TrainingConfig(config_path=config_path))

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Update with missing default values
    for k in DEFAULT_CONFIG:
        if k not in config:
            config[k] = DEFAULT_CONFIG[k]

    return config


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


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "T2I Trainer Backend API is running"}


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


@app.get("/api/list_templates")
async def list_templates_endpoint(config_path: str = "config.json"):
    """List available templates"""
    templates = list_templates(config_path)

    return {
        "status": "success",
        "templates": templates
    }


@app.get("/api/available_scripts")
async def get_available_scripts():
    """Get list of available training scripts in the current directory"""
    import glob

    # Get all Python files in the current directory that match training script patterns
    script_patterns = [
        "train_*.py",  # All training scripts
    ]

    available_scripts = []
    for pattern in script_patterns:
        scripts = glob.glob(pattern)
        available_scripts.extend(scripts)

    # Remove duplicates and return
    unique_scripts = list(set(available_scripts))

    return {
        "status": "success",
        "scripts": unique_scripts
    }


@app.get("/api/default_config")
async def get_default_config():
    """Get the default configuration"""
    return {
        "status": "success",
        "config": DEFAULT_CONFIG
    }


# WebSocket connection manager to handle multiple clients
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass  # Connection may have been closed

manager = ConnectionManager()

# Global variable to track the currently running training process
running_process = None
training_thread = None

async def run_training_script_realtime(script_path: str, config_path: str, websocket: WebSocket):
    
    """Run a training script with the provided configuration file and stream output in real-time"""
    global running_process
    print(f"Starting run_training_script_realtime with script: {script_path}, config: {config_path}")  # Debug print

    # Validate that the training script exists
    script_path_obj = Path(script_path)
    if not script_path_obj.exists():
        # Check in current directory if not found as full path
        script_file = Path(script_path_obj.name)
        if not script_file.exists():
            raise FileNotFoundError(f"Training script {script_path} not found")
        script_path = script_file.name

    if os.path.exists(config_path):
        # get full path
        config_path = os.path.abspath(config_path)
    
    # Build the command to run the training script
    cmd = [sys.executable, script_path, "--config_path", config_path]

    try:
        # Execute the training script as a subprocess with real-time output
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"  # Force unbuffered output

        print(f"Starting subprocess with command: {' '.join(cmd)}")  # Debug print
        running_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # Redirect stderr to stdout
            env=env
        )
        print(f"Subprocess started with PID: {running_process.pid}")  # Debug print

        print(f"Starting subprocess: {' '.join(cmd)}")  # Debug print

        # Real-time output reading (async approach)
        async def read_output():
            """Read output from the subprocess asynchronously"""
            print("Output reading task started")  # Debug print
            try:
                while True:
                    # Read a line from the subprocess stdout
                    line = await running_process.stdout.readline()
                    line_str = line.decode(encoding="utf-8") if isinstance(line, bytes) else line
                    
                    if line_str is not None and line_str.strip() != "":
                        print(f"Read line: {repr(line_str)}")  # Debug print
                    
                    # Check both if line is empty and if the process has terminated
                    if not line_str and running_process.returncode is not None:
                        print("EOF reached, breaking output loop")  # Debug print
                        break
                    
                    # Only send non-empty lines to frontend
                    if line_str.strip():
                        # Check if this is the training end message
                        if "=========== End Training ===========" in line_str:
                            print("Detected training end message")  # Debug print
                            # Send a special training end message to frontend
                            end_message = {
                                "type": "training_end",
                                "data": "Training completed successfully"
                            }
                            try:
                                end_message_json = json.dumps(end_message)
                                print(f"Sending training end message: {end_message_json}")  # Debug print
                                await websocket.send_text(end_message_json)
                                print("Training end message sent successfully")  # Debug print
                            except Exception as e:
                                print(f"Error sending training end message to WebSocket: {e}")  # Debug print
                            
                            # Still send the original line as output
                            message = {
                                "type": "output",
                                "data": line_str.rstrip('\n\r').replace("\r", "\n")
                            }
                        else:
                            message = {
                                "type": "output",
                                "data": line_str.rstrip('\n\r').replace("\r", "\n")
                            }
                        
                        try:
                            message_json = json.dumps(message)
                            print(f"Sending JSON message: {message_json}")  # Debug print
                            await websocket.send_text(message_json)
                            print(f"Successfully sent output line: {repr(line_str[:50])}...")  # Debug print first 50 chars
                        except Exception as e:
                            print(f"Error sending output to WebSocket: {e}")  # Debug print
                            print(f"Failed to send message: {message}")  # Debug print
                            # Continue processing other messages instead of breaking the loop
                            continue                    
                    # Skip empty lines and whitespace-only lines
                    elif line_str and line_str.isspace():
                        print(f"Skipping whitespace line: {repr(line_str)}")  # Debug print
                        continue
                    elif line_str:
                        print(f"Skipping empty line: {repr(line_str)}")  # Debug print
                        continue
                
                # Wait for the process to complete and get the return code
                await running_process.wait()
                return_code = running_process.returncode
                print(f"Process completed with return code: {return_code}")  # Debug print
                
                # Send completion message
                result = {
                    "type": "complete",
                    "status": "success" if return_code == 0 else "error",
                    "return_code": return_code,
                    "message": "Training completed successfully" if return_code == 0 else f"Training failed with return code {return_code}"
                }
                print(f"Sending completion message: {result}")  # Debug print
                await websocket.send_text(json.dumps(result))
                print("Completion message sent successfully")  # Debug print
                
            except Exception as e:
                print(f"Error in output reading task: {e}")  # Debug print
                import traceback
                print(f"Full traceback: {traceback.format_exc()}")  # Debug print
                error_result = {
                    "type": "error",
                    "status": "error",
                    "message": str(e)
                }
                try:
                    print(f"Sending error message: {error_result}")  # Debug print
                    await websocket.send_text(json.dumps(error_result))
                    print("Error message sent successfully")  # Debug print
                except Exception as send_error:
                    print(f"Error sending error message: {send_error}")  # Debug print
                    pass
            finally:
                print("Output reading task ending")  # Debug print

        # Start the output reading task
        read_task = asyncio.create_task(read_output())
        print("Output task started")  # Debug print
        
        # Wait for the process to complete
        await read_task
        
    except Exception as e:
        print(f"Error in run_training_script_realtime: {e}")  # Debug print
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")  # Debug print
        error_result = {
            "type": "error",
            "status": "error",
            "message": str(e)
        }
        try:
            print(f"Sending error result: {error_result}")  # Debug print
            await websocket.send_text(json.dumps(error_result))
            print("Error result sent successfully")  # Debug print
        except Exception as send_error:
            print(f"Error sending error result: {send_error}")  # Debug print
            pass
    finally:
        print("Cleaning up running_process")  # Debug print
        running_process = None


@app.websocket("/ws/training")
async def websocket_training_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time training output"""
    print("WebSocket connection requested")  # Debug print
    await manager.connect(websocket)
    print(f"WebSocket connected. Active connections: {len(manager.active_connections)}")  # Debug print
    try:
        print("Waiting for configuration data...")  # Debug print
        while True:
            # Receive configuration data from the frontend
            data = await websocket.receive_text()
            print(f"Received configuration data: {data[:100]}...")  # Debug print first 100 chars
            config_data = json.loads(data)

            script_path = config_data.get("script")
            config_path = config_data.get("config_path")

            print(f"Processing training for script: {script_path}, config: {config_path}")  # Debug print

            # Save configuration to file first
            config_obj = TrainingConfig(**config_data)
            config_path = save_config_to_file(config_obj)
            print(f"Configuration saved to: {config_path}")  # Debug print

            # Send connection acknowledgment
            try:
                ack_msg = {"type": "connection", "status": "connected", "message": "WebSocket connected successfully"}
                print(f"Sending connection acknowledgment: {ack_msg}")  # Debug print
                await websocket.send_text(json.dumps(ack_msg))
                print("Sent connection acknowledgment")  # Debug print
                
                # Run the training script with real-time output
                print("Starting training script...")  # Debug print
                await run_training_script_realtime(script_path, config_path, websocket)
                print("Training script completed")  # Debug print
                
                # Exit the loop after training is completed
                break
            except Exception as e:
                print(f"Error sending connection ack: {e}")  # Debug print
                raise e

    except WebSocketDisconnect:
        print("WebSocket disconnected")  # Debug print
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")  # Debug print
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")  # Debug print
        error_result = {
            "type": "error",
            "status": "error",
            "message": str(e)
        }
        try:
            print(f"Sending error result: {error_result}")  # Debug print
            await websocket.send_text(json.dumps(error_result))
            print("Error result sent successfully")  # Debug print
        except Exception as send_error:
            print(f"Error sending error result: {send_error}")  # Debug print
            pass
        manager.disconnect(websocket)


@app.websocket("/ws/test")
async def websocket_test_endpoint(websocket: WebSocket):
    """WebSocket endpoint for testing communication"""
    print("Test WebSocket connection requested")  # Debug print
    await manager.connect(websocket)
    print(f"Test WebSocket connected. Active connections: {len(manager.active_connections)}")  # Debug print
    try:
        print("Waiting for test message...")  # Debug print
        while True:
            # Receive test message from the frontend
            data = await websocket.receive_text()
            print(f"Received test message: {data}")  # Debug print
            
            # Parse the JSON data
            try:
                message_data = json.loads(data)
                # Handle both string messages and config-like structures
                if isinstance(message_data, dict) and 'message' in message_data:
                    message = message_data['message']
                else:
                    message = str(message_data)
            except json.JSONDecodeError:
                message = data  # Use raw data if not JSON
            
            # Echo the message back with a prefix
            response = {
                "type": "output",
                "data": f"Echo from backend: {message}"
            }
            print(f"Sending test response: {response}")  # Debug print
            await websocket.send_text(json.dumps(response))
            print("Test response sent successfully")  # Debug print

    except WebSocketDisconnect:
        print("Test WebSocket disconnected")  # Debug print
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Test WebSocket error: {e}")  # Debug print
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")  # Debug print
        error_result = {
            "type": "error",
            "status": "error",
            "message": str(e)
        }
        try:
            print(f"Sending test error result: {error_result}")  # Debug print
            await websocket.send_text(json.dumps(error_result))
            print("Test error result sent successfully")  # Debug print
        except Exception as send_error:
            print(f"Error sending test error result: {send_error}")  # Debug print
            pass
        manager.disconnect(websocket)
                                    
    except WebSocketDisconnect:
        print("Test WebSocket disconnected")  # Debug print
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Test WebSocket error: {e}")  # Debug print
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")  # Debug print
        error_result = {
            "type": "error",
            "status": "error",
            "message": str(e)
        }
        try:
            print(f"Sending test error result: {error_result}")  # Debug print
            await websocket.send_text(json.dumps(error_result))
            print("Test error result sent successfully")  # Debug print
        except Exception as send_error:
            print(f"Error sending test error result: {send_error}")  # Debug print
            pass
        manager.disconnect(websocket)


# Replace the run_training function to use the new approach
@app.post("/api/run")
async def run_training(config: TrainingConfig):
    """Run the training process with the provided configuration (maintaining compatibility)"""
    try:
        # Save configuration to file first
        config_path = save_config_to_file(config)

        # For backward compatibility, run with the old method (blocking)
        result = run_training_script(config.script, config_path)

        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Training process timed out")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_custom_templates_dir():
    """Get custom templates directory path"""
    custom_templates_dir = "./frontend/public/custom_templates"
    os.makedirs(custom_templates_dir, exist_ok=True)
    return custom_templates_dir


def list_custom_templates():
    """List available custom templates"""
    custom_templates_dir = get_custom_templates_dir()
    templates = []
    if os.path.exists(custom_templates_dir):
        templates = [os.path.basename(f) for f in glob.glob(os.path.join(custom_templates_dir, "*.json"))]
        templates.sort()
    return templates


class TemplateData(BaseModel):
    name: str
    workflow: Dict[str, Any]


@app.post("/api/templates/custom")
async def save_custom_template(template_data: TemplateData):
    """Save a custom template to the custom_templates directory"""
    try:
        custom_templates_dir = get_custom_templates_dir()
        
        # Ensure the filename ends with .json
        filename = template_data.name
        if not filename.endswith('.json'):
            filename = f"{filename}.json"
        
        # Create the full path
        template_path = os.path.join(custom_templates_dir, filename)
        
        # Add metadata to the workflow data
        workflow_data = {
            **template_data.workflow,
            "timestamp": time.time(),
            "version": "1.0"
        }
        
        # Write the template to file
        with open(template_path, 'w', encoding='utf-8') as f:
            json.dump(workflow_data, f, indent=2, ensure_ascii=False)
        
        return {"status": "success", "message": f"Template '{filename}' saved successfully"}
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
        custom_templates_dir = get_custom_templates_dir()
        template_path = os.path.join(custom_templates_dir, template_name)
        
        # Ensure the path is within the custom_templates directory for security
        if os.path.commonpath([custom_templates_dir]) != os.path.commonpath([custom_templates_dir, template_path]):
            raise HTTPException(status_code=400, detail="Invalid template name")
        
        if os.path.exists(template_path):
            os.remove(template_path)
            return {"status": "success", "message": f"Template '{template_name}' deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Template not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stop_training")
async def stop_training():
    """Stop the currently running training process"""
    global running_process
    if running_process:
        running_process.terminate()
        running_process.wait()  # Wait for process to actually terminate
        running_process = None
        return {"status": "success", "message": "Training process stopped successfully"}
    else:
        return {"status": "error", "message": "No training process is currently running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)