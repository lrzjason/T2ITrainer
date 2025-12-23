"""
Shared utility functions for all services
"""
import json
import os
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from .config import (
    PROJECT_ROOT, LOG_DIR, TRAINING_STATUS_FILE,
    CONFIG_TEMPLATE_DIR, CUSTOM_TEMPLATES_DIR, DEFAULT_CONFIG
)


def get_current_log_filename() -> Path:
    """Generate log filename based on current date"""
    today = datetime.now().strftime("%Y-%m-%d")
    return LOG_DIR / f"training_log_{today}.json"


def init_log_file(log_file: Path):
    """Initialize log file with empty array if it doesn't exist"""
    if not log_file.exists():
        with open(log_file, 'w') as f:
            json.dump([], f)


def append_to_log(log_file: Path, entry: Dict[str, Any]):
    """Append an entry to the log file"""
    if log_file.exists():
        with open(log_file, 'r') as f:
            try:
                entries = json.load(f)
            except json.JSONDecodeError:
                entries = []
    else:
        entries = []
    
    entries.append(entry)
    
    with open(log_file, 'w') as f:
        json.dump(entries, f, indent=2)


def update_training_status(status: str):
    """Update the training status file"""
    status_data = {
        "current_status": status,
        "last_updated": datetime.now().isoformat()
    }
    
    with open(TRAINING_STATUS_FILE, 'w') as f:
        json.dump(status_data, f, indent=2)


def get_training_status() -> Dict[str, Any]:
    """Get the current training status"""
    if TRAINING_STATUS_FILE.exists():
        with open(TRAINING_STATUS_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {"current_status": "unknown", "last_updated": datetime.now().isoformat()}
    else:
        return {"current_status": "not_started", "last_updated": datetime.now().isoformat()}


def get_script_config_mapping() -> Dict[str, Dict[str, Any]]:
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


def list_templates(config_path: str) -> List[str]:
    """List available templates"""
    templates = [
        os.path.basename(f) 
        for f in glob.glob(str(CONFIG_TEMPLATE_DIR / "*.json"))
    ]
    templates.sort()
    config_name = os.path.basename(config_path) if os.path.isfile(config_path) else "config.json"
    return [config_name] + templates


def load_template(template_name: str, config_path: str) -> Dict[str, Any]:
    """Load a template, merging missing sections into current config"""
    if isinstance(template_name, list):
        template_name = template_name[0] if template_name else ""
    if not template_name:
        return {}

    tpl = {}
    config_name = os.path.basename(config_path)

    if template_name == config_name and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            tpl = json.load(f)
    else:
        tpl_path = CONFIG_TEMPLATE_DIR / template_name
        if not tpl_path.is_file():
            return {}
        with open(tpl_path, "r", encoding="utf-8") as f:
            tpl = json.load(f)

    current = {}
    for key in ("image_configs", "caption_configs", "training_set", "dataset_configs"):
        if key in tpl and key not in current:
            current[key] = tpl[key]

    return current


def get_script_specific_config(selected_script: str, config_path: str = "config.json") -> Dict[str, Any]:
    """Get config with script-specific requirements"""
    pairs = get_script_config_mapping()

    if selected_script not in pairs:
        return {}

    preset_config = pairs[selected_script]
    preset_name = preset_config["preset"]
    preset_path = CONFIG_TEMPLATE_DIR / preset_name

    editor = {}
    preset = {}
    mandatory = preset_config["mandatory"]

    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                preset = json.load(f)
        except Exception:
            pass
    else:
        try:
            with open(preset_path, "r", encoding="utf-8") as f:
                preset = json.load(f)
        except Exception:
            print(f"Error reading {preset_path}")

    for key in mandatory:
        if key in preset:
            editor[key] = preset[key]

    return editor


def list_custom_templates() -> List[str]:
    """List available custom templates"""
    templates = []
    if CUSTOM_TEMPLATES_DIR.exists():
        templates = [
            os.path.basename(f) 
            for f in glob.glob(str(CUSTOM_TEMPLATES_DIR / "*.json"))
        ]
        templates.sort()
    return templates


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load an existing configuration from file, with fallback to default config"""
    full_path = PROJECT_ROOT / config_path if not os.path.isabs(config_path) else Path(config_path)
    
    if not full_path.is_file():
        # Return default config if file doesn't exist
        return DEFAULT_CONFIG.copy()

    with open(full_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Update with missing default values
    for k in DEFAULT_CONFIG:
        if k not in config:
            config[k] = DEFAULT_CONFIG[k]

    return config


def save_config_to_file(config_data: Dict[str, Any]) -> str:
    """Save the training configuration to a JSON file"""
    config_path = config_data.get("config_path", "config.json")
    full_path = PROJECT_ROOT / config_path if not os.path.isabs(config_path) else Path(config_path)
    
    config_dir = full_path.parent
    config_dir.mkdir(parents=True, exist_ok=True)

    # Get the current configuration from file if it exists
    if full_path.is_file():
        with open(full_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}

    # Update with the new configuration values
    cfg.update(config_data)

    # Write to file
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)

    # Also save to default config.json
    default_config_path = PROJECT_ROOT / "config.json"
    with open(default_config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)

    return str(full_path)


def generate_job_id() -> str:
    """Generate a unique job ID"""
    import uuid
    return f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
