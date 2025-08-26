import argparse
import os
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from accelerate import Accelerator
from latentdiffusion.ldm.util import instantiate_from_config
from latentdiffusion.ldm.models.diffusion.ddim import DDIMSampler

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

def generate_diffusion_sample(model, prompt: str, outdir: str, height: int, width: int, 
                    steps: int, scale: float, eta: float, n_samples: int, seed: int | None):
    os.makedirs(outdir, exist_ok=True)

    if seed is None:
        seed = torch.seed() % (2**31)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if scale != 1.0:
                uc = model.get_learned_conditioning(n_samples * [""])
            c = model.get_learned_conditioning(n_samples * [prompt])
            shape = [4, height // 8, width // 8]
            samples, _ = sampler.sample(
                S=steps,
                conditioning=c,
                batch_size=n_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc,
                eta=eta,
            )
        
            x_samples = model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

            # # Save each sample
            saved_files = []
            for idx, x in enumerate(x_samples):
                array = 255.0 * rearrange(x.cpu().numpy(), 'c h w -> h w c')
                image = Image.fromarray(array.astype(np.uint8))
                safe_prompt = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '-' for ch in prompt)[:60]
                filename = f"{safe_prompt or 'sample'}_seed{seed}_{idx:02d}.png"
                filepath = os.path.join(outdir, filename)
                image.save(filepath)
                saved_files.append(filepath)
    return saved_files

def parse_args():
    parser = argparse.ArgumentParser(description= "Generate a image with stable diffusion model using command-line arguments")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--app_config", type=str, default=None, help="YAML for prompts and app settings")
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
    parser.add_argument('--random_prompt', action='store_true', help = "Ignore --prompt and use a randomly generated prompt")
    # Generation seed. 
    parser.add_argument('--seed', type = int, default = None, help = "Random seed for generation. If None, a random seed will be used.")
    # Sampling params
    parser.add_argument('--height', type = int, default = 512, help = "Image height in pixels")
    parser.add_argument('--width', type = int, default = 512, help = "Image width in pixels")
    parser.add_argument('--steps', type = int, default = 50, help = "Number of DDIM sampling steps")
    parser.add_argument('--scale', type = float, default = 7.5, help = "Classifier-free guidance scale")
    parser.add_argument('--eta', type = float, default = 0.0, help = "DDIM eta; 0.0 is deterministic")
    parser.add_argument('--n_samples', type = int, default = 1, help = "Number of samples to generate")
    
    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator()

    # Load the model 
    pipe = load_StableDiffusion_model(args.config, args.ckpt)
    
    ######check if the model is loaded successfully
    if pipe == "path not exists":
        raise FileNotFoundError(f"Missing config: {args.config}")
        raise FileNotFoundError(f"Missing checkpoint: {args.ckpt}")
    else:
        print("Stable Diffusion model loaded.")
    if accelerator.is_main_process:
        print("Stable Diffusion model loaded.")
    #########################################################
    
    # Determine prompt from CLI or config.yaml
    prompt_value = args.prompt
    # Load app config (separate from model config) for prompts
    cfg = None
    if args.app_config:
        try:
            cfg = OmegaConf.load(args.app_config)
        except Exception:
            cfg = None

    if args.random_prompt:
        # pick from config random_prompts if present
        candidates = None
        if cfg is not None:
            candidates = OmegaConf.select(cfg, "app.random_prompts")
            if candidates is None:
                candidates = OmegaConf.select(cfg, "random_prompts")
        if isinstance(candidates, (list, tuple)) and len(candidates) > 0:
            prompt_value = random.choice(list(candidates))
        elif isinstance(candidates, str) and candidates.strip():
            prompt_value = candidates.strip()
        else:
            prompt_value = prompt_value or "a beautiful landscape, digital art"
    else:
        if not prompt_value and cfg is not None:
            prompt_value = OmegaConf.select(cfg, "app.prompt") or OmegaConf.select(cfg, "prompt")
        if not prompt_value:
            # fallback if nothing provided
            prompt_value = "a beautiful landscape, digital art"

    # Generate and save
    saved = generate_diffusion_sample(
        model=pipe,
        prompt=prompt_value,
        outdir=args.save_dir,
        height=args.height,
        width=args.width,
        steps=args.steps,
        scale=args.scale,
        eta=args.eta,
        n_samples=args.n_samples,
        seed=args.seed,
    )

    if accelerator.is_main_process:
        print(f"Prompt: {prompt_value}")
        print("Saved files:")
        for f in saved:
            print(f)
    
if __name__ == "__main__":
    main()