from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch 
import os
from transformers import CLIPImageProcessor
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import os

def generate_image_with_latent(prompt: str, output_path: str,
                               model_id: str = "runwayml/stable-diffusion-v1-5", 
                               num_inference_steps: int = 50, 
                               guidance_scale: float = 8, 
                               use_ddim: bool = True,
                               num_samples: int = 1,
                               seed: int = None):
    """
    Generate image(s) with a text prompt using Stable Diffusion pipeline.
    Saves the image(s) to output_path (or numbered versions for multiple samples).
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set seed for reproducibility
    if seed is not None: 
        torch.manual_seed(seed)

    # Ensure output directory exists
    outdir = os.path.dirname(output_path)
    if outdir and not os.path.exists(outdir):  
        os.makedirs(outdir, exist_ok=True)

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

    # Replace scheduler with DDIM if requested
    if use_ddim:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    saved_files = []
    
    # Generate multiple samples
    for i in range(num_samples):
        # Generate image with different seed for each sample
        if seed is not None:
            torch.manual_seed(seed + i)
            
        images = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images
        
        # Create filename for each sample
        if num_samples == 1:
            filepath = output_path
        else:
            base, ext = os.path.splitext(output_path)
            filepath = f"{base}_{i+1:02d}{ext}"
            
        images[0].save(filepath)
        saved_files.append(filepath)
        print(f"Image {i+1}/{num_samples} saved to {filepath}")
    
    return saved_files

if __name__ == "__main__":
    
    # Generate 10 samples with the same prompt
    # saved_files = generate_image_with_latent(
    #     prompt="a watercolor fox in a misty forest",
    #     output_path="./output/fox_samples.png",
    #     num_samples=20,
    #     seed=42,  # Fixed seed for reproducible variations
    #     num_inference_steps=50,
    #     guidance_scale=7.5
    # )
    
    # saved_files = generate_image_with_latent(
    #     prompt="Three origami cranes in white, gold, and blue are perched on a black lacquered shelf above a blooming bonsai tree.",
    #     output_path="./output/tree_samples.png",
    #     num_samples=20,
    #     seed=42,  # Fixed seed for reproducible variations
    #     num_inference_steps=50,
    #     guidance_scale=7.5
    # )
    saved_files = generate_image_with_latent(
        prompt="Two pink flamingos standing on a wooden dock, with three turquoise kayaks floating on the calm lake behind them.",
        output_path="./output/flamingos_samples.png",
        num_samples=10,
        seed=42,  # Fixed seed for reproducible variations
        num_inference_steps=50,
        guidance_scale=7.5
    )
    
    print(f"\nGenerated {len(saved_files)} samples:")
    for file in saved_files:
        print(f"  - {file}") 


