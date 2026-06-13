#To use this module, make sure you have the pyyaml library tracking in your setup environment (!pip install pyyaml -q).
#!/usr/bin/env python3
"""
config/load_config.py
A thread-safe project parsing utility to cleanly ingest raw parameter mappings.
"""

import os
import yaml
from typing import Dict, Any

def get_project_root() -> str:
    """Calculates the absolute path to the root folder of the active repository footprint."""
    # Navigates upward from this specific file out of the config directory boundary
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_pipeline_parameters() -> Dict[str, Any]:
    """Reads parameters.yaml file arrays and converts them into an accessible python dictionary."""
    root_dir = get_project_root()
    yaml_target_path = os.path.join(root_dir, "config", "parameters.yaml")
    
    if not os.path.exists(yaml_target_path):
        raise FileNotFoundError(f"Configuration file missing at projected vector path: {yaml_target_path}")
        
    with open(yaml_target_path, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)
        
    # Dynamically inject absolute paths to protect file streams during code execution loops
    config_data["ingestion"]["raw_data_path"] = os.path.join(root_dir, config_data["ingestion"]["raw_data_path"])
    config_data["ingestion"]["chroma_db_dir"] = os.path.join(root_dir, config_data["ingestion"]["chroma_db_dir"])
    
    return config_data

# Simple console diagnostic checker execution test hook
if __name__ == "__main__":
    try:
        cfg = load_pipeline_parameters()
        print("✅ Configuration pipeline configurations compiled cleanly.")
        print(f"Index storage verified target location: {cfg['ingestion']['chroma_db_dir']}")
    except Exception as e:
        print(f"❌ Configuration matrix ingestion failure context: {e}")
