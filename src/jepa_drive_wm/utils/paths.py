
from dataclasses import dataclass
from pathlib import Path


VJEPA_REPO = "/home/hashim/Desktop/Modules/vjepa2"
VJEPA_CHECKPOINTS_DIR = "/home/hashim/Desktop/Modules/model_checkpoints/vjepa21"

@dataclass 
class PATHS: 
    VJEPA_REPO: Path = Path(VJEPA_REPO)
    VJEPA_CHECKPOINTS_DIR: Path = Path(VJEPA_CHECKPOINTS_DIR)