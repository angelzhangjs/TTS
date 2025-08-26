"""
This file use to implement the Test time scaling framework 
using latent diffusion model as a backbone model 
we define the TTS framework as path searching, as two methods.
1. making the final path -- without rewards, pure search
2. making the intermediate latent space leading to better final path -- with rewards 

and fun all the model here. 
"""
# print all intermediate latent space results. 

import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf

def add_repo_to_path(repo_root: Path) -> None:
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

def load_latent_diffusion(config_path: Path, checkpoint_path: Path, device: str | None = None):
    add_repo_to_path(Path("/Users/angel/tts_LDM/latent-diffusion"))
    from ldm.util import instantiate_from_config

    config = OmegaConf.load(str(config_path))
    state = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = state["state_dict"] if "state_dict" in state else state

    model = instantiate_from_config(config.model)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device)

def get_default_paths():
    repo_root = Path("/Users/angel/tts_LDM/latent-diffusion")
    config_path = repo_root / "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
    checkpoint_path = repo_root / "models/ldm/text2img-large/model.ckpt"
    return config_path, checkpoint_path


if __name__ == "__main__":
    cfg_path, ckpt_path = get_default_paths()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    model = load_latent_diffusion(cfg_path, ckpt_path)
    print("LatentDiffusion model loaded.")
 

