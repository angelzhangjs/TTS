import argparse
import torch
from omegaconf import OmegaConf
from accelerate import Accelerator
from latentdiffusion.ldm.util import instantiate_from_config
from pathlib import Path

def load_StableDiffusion_model(config_path, ckpt_path):
    # Load the model from the config path and ckpt path
    if not Path(config_path).exists() or not Path(ckpt_path).exists():
        return "path not exists"
    config = OmegaConf.load(config_path)
    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state["state_dict"] if "state_dict" in state else state
    model = instantiate_from_config(config.model)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model
    
def parse_args():
    parser = argparse.ArgumentParser(description= "Generate a image with stable diffusion model using command-line arguments")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ckpt", type=str, default="ckpt.pt")
    parser.add_argument("--output", type=str, default="")
    # Core Model Paths 
    parser.add_argument("--model_path", type=str, default="./pretrained_models/Infinity/infinity_2b_reg.pth", help="Path to the Infinity model.")
    parser.add_argument("--vae_path", type=str, default="./pretrained_models/Infinity/infinity_vae_d32reg.pth", help="Path to the VAE model.")
    parser.add_argument("--text_encoder_ckpt", type=str, default="./pretrained_models/flan-t5-xl", help="Path to the text encoder checkpoint.")
    # Output 
    parser.add_argument('--save_dir', type=str, default = './output', help = "Directory to save the generated images")
    parser.add_argument('--output_filename', type=str, default = 'Optional filename for the output image. If None, a name based on prompt and seed will be generated.')
    #Prompt
    parser.add_argument('--prompt', type = str, help = "text prompt for sample image generation.")
    # Generation seed. 
    parser.add_argument('--seed', type = int, default = 42, help = "Random seed for generation. If None, a random seed will be used.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator()

    # Load the model 
    pipe = load_StableDiffusion_model(args.config, args.ckpt)
    if pipe == "path not exists":
        raise FileNotFoundError(f"Missing config: {args.config}")
        raise FileNotFoundError(f"Missing checkpoint: {args.ckpt}")
    else:
        print("Stable Diffusion model loaded.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device)
    if accelerator.is_main_process: print("Stable Diffusion model loaded.")
    
if __name__ == "__main__":
    main()