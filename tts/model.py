from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline
import torch 
import os
from tts.tts_search import beam_search_latents, best_of_n_search

def generate_with_latent_search(prompt: str, output_path: str,
                               model_id: str = "runwayml/stable-diffusion-v1-5",
                               search_method: str = "best_of_n",
                               num_beams: int = 4, 
                               n_candidates: int = 4,
                               num_inference_steps: int = 50,
                               guidance_scale: float = 7.5,
                               height: int = 512, width: int = 512,
                               **search_kwargs):
    """
    Generate image using latent space search methods for better quality. 
    """
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Ensure output directory exists
    outdir = os.path.dirname(output_path)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    
    # Load pipeline with memory optimizations
    amp_dtype = torch.float16 if device.type == "cuda" else torch.float32
    if "stable-diffusion-3" in model_id:
        pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=amp_dtype).to(device)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=amp_dtype).to(device)
    
    # Memory optimizations
    pipe.enable_attention_slicing()
    try:
        pipe.enable_vae_slicing()
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_sequential_cpu_offload()
    except Exception:
        pass
    
    # Apply search method
    if search_method == "beam_search":
        image, extra_info = beam_search_latents(
            pipe, prompt, 
            num_beams=num_beams,
            num_steps=num_inference_steps,
            height=height, width=width,
            guidance_scale=guidance_scale,
            search_steps=search_kwargs.get("search_steps", [10, 20, 30])
        )
    elif search_method == "best_of_n":
        image, extra_info = best_of_n_search(
            pipe, prompt,
            num_steps=num_inference_steps,
            height=height, width=width,
            guidance_scale=guidance_scale,
            selection_step=search_kwargs.get("selection_step", max(1, num_inference_steps // 2))
        )
    else:
        raise ValueError(f"Unknown search method: {search_method}")
    
    # Save image 
    image.save(output_path)
    print(f"Generated with {search_method}: {output_path}")
    print(f"Search info: {extra_info}")
    
    return output_path, extra_info

if __name__ == "__main__":
    
    # sd_3 = "stabilityai/stable-diffusion-3-medium-diffusers"
    # sd_1 = "runwayml/stable-diffusion-v1-5"
    # saved_files = generate_image_with_latent(
    #     prompt="An oil painting, where a green vintage car, a blue scooter on the left of it and a black bycycle on the right of it, are parked on the road, with two birds in the sky",
    #     output_path="./output/car_samples_sd3.png", 
    #     model_id=sd_1,
    #     num_samples=10,
    #     seed=42,  # Fixed seed for reproducible variations
    #     num_inference_steps=50, 
    #     guidance_scale=7.5
    # ) 
    
    # print(f"\nGenerated {len(saved_files)} samples:")
    # for file in saved_files:
    #     print(f"  - {file}") 
    
    # Example: Latent search methods for better quality
    print("\n=== Testing Latent Search Methods ===")
    prompt = "A majestic dragon flying over a medieval castle at sunset"
    sd_1 = "runwayml/stable-diffusion-v1-5" 
    # Beam search - explores multiple paths
    best_oF_N_result, _ = generate_with_latent_search(
        prompt=prompt,
        output_path="./output/dragon_beam_search.png",
        model_id=sd_1,
        search_method="best_of_n",
        n_candidates=4,
        num_inference_steps=50,
        guidance_scale=7.5
    )
    print(f"Beam search completed: {best_oF_N_result}")
        
############## Draft #############
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
    # saved_files = generate_image_with_latent(
    #     prompt="Two pink flamingos standing on a wooden dock, with three turquoise kayaks floating on the calm lake behind them.",
    #     output_path="./output/flamingos_samples.png",
    #     num_samples=10,
    #     seed=42,  # Fixed seed for reproducible variations
    #     num_inference_steps=50,
    #     guidance_scale=7.5
    # ) 
    


