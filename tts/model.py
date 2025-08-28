import os
import sys

# Fix CUDA device visibility issues
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Memory optimization environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings

from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline
import torch 

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tts.tts_search import beam_search_latents, best_of_n_search

def generate_with_latent_search(prompt: str, output_path: str,
                               model_id: str = "runwayml/stable-diffusion-v1-5",
                               search_method: str = "best_of_n",
                               num_beams: int = 4, 
                               n_candidates: int = 4,
                               num_inference_steps: int = 50,
                               guidance_scale: float = 7.5,
                               height: int = 512, width: int = 512,
                               force_cpu: bool = False,
                               **search_kwargs):
    """
    Generate image using latent space search methods for better quality. 
    """
    # Clear CUDA cache and set device safely
    if force_cpu:
        print("Forcing CPU usage as requested")
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            device = torch.device("cuda:0")
            # Test if the device is actually accessible
            torch.zeros(1).to(device)
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        except (RuntimeError, AssertionError) as e:
            print(f"Warning: CUDA device not accessible ({e}), falling back to CPU")
            device = torch.device("cpu")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    
    # Ensure output directory exists
    outdir = os.path.dirname(output_path)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    
    # Load pipeline with aggressive memory optimizations
    amp_dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    if "stable-diffusion-3" in model_id:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id, 
            torch_dtype=amp_dtype,
            variant="fp16" if device.type == "cuda" else None,
            use_safetensors=True,
            safety_checker=None,  # Disable NSFW filter
            requires_safety_checker=False
        ).to(device)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=amp_dtype,
            variant="fp16" if device.type == "cuda" else None,
            use_safetensors=True,
            safety_checker=None,  # Disable NSFW filter
            requires_safety_checker=False
        ).to(device)
    
    # Enable memory efficient optimizations only if using GPU
    if device.type == "cuda" and not force_cpu:
        try:
            # Only enable basic, safe optimizations
            pipe.enable_vae_slicing()
            print("✓ Enabled VAE slicing")
        except Exception as e:
            print(f"⚠ VAE slicing failed: {e}")
    else:
        print("Using CPU - skipping GPU optimizations")
    
    # Apply search method
    if search_method == "beam_search":
        image, candidates = beam_search_latents(
            pipe, prompt, 
            num_beams=num_beams,
            num_steps=num_inference_steps,
            height=height, width=width,
            guidance_scale=guidance_scale,
            search_steps=search_kwargs.get("search_steps", [10, 20, 30])
        )
    elif search_method == "best_of_n":
        print(f"Debug: n_candidates={n_candidates}, search_method={search_method}")
        # For n_candidates=1, just use standard pipeline
        if n_candidates <= 1:
            print("✓ Using standard pipeline (n_candidates=1)")
            try:
                image = pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(device=device).manual_seed(42)
                ).images[0]
                candidates = "standard_generation"
                print("✓ Standard pipeline completed successfully")
            except Exception as e:
                print(f"❌ Error in GPU pipeline: {e}")
                print("🔄 Trying CPU fallback...")
        
                # Move pipeline to CPU and try again
                pipe = pipe.to("cpu")
                device = torch.device("cpu")
                
                image = pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(device=device).manual_seed(42)
                ).images[0]
                candidates = "standard_generation_cpu_fallback"
                print("✓ CPU fallback completed successfully")
        else:
            print(f"Using best_of_n_search with {n_candidates} candidates")
            image, candidates = best_of_n_search(
                pipe, prompt,
                n_candidates=n_candidates,
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
    print(f"Search info: {candidates}")
    
    return output_path, candidates

if __name__ == "__main__":
     
    # Example: Latent search methods for better quality
    print("\n=== Testing Latent Search Methods ===")
    prompt = "An oil painting, where a green vintage car, a blue scooter on the left of it and a black bicycle on the right of it, are parked on the road, with two birds in the sky. "
    flamingo_prompt = "Two pink flamingos standing on a wooden dock, with three turquoise kayaks floating on the calm lake behind them."
    cube_prompt = "A red cube on the left of a blue cube, a white bowl on the top of the blue cube."
    bottle_bicycle_prompt = "a photo of a bottle and a bicycle"
    
    sd_1 = "runwayml/stable-diffusion-v1-5" 
    # Best-of-N search - explores multiple candidates
    best_of_N_result, _ = generate_with_latent_search(
        prompt= bottle_bicycle_prompt,
        output_path="./output/bottle_best_of_n.png",
        num_beams=3,
        model_id=sd_1,
        search_method="best_of_n",
        n_candidates=1,  # Minimal memory usage
        num_inference_steps=30,  # Faster generation
        guidance_scale=7.5,
        height=512, width=512,  # Use standard dimensions to avoid architecture issues
        force_cpu=True  # Use CPU to avoid tensor shape issues
    )
    print(f"best of N search completed: {best_of_N_result}")
    
        
