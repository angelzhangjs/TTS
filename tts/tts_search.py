"""
Implement the diffusion latent search framework here, to search the latent space of for a better diffusion path over the reverse process. 
 -use beam search 
 -use Best-of-N search 
"""
import torch
import numpy as np
from typing import List, Tuple, Optional
import os 
from PIL import Image

def beam_search_latents(pipe, 
                        prompt: str, 
                        num_beams: int = 4, 
                        num_steps: int = 50, 
                       height: int = 512, 
                       width: int = 512, 
                       guidance_scale: float = 7.5,
                       search_steps: List[int] = [10, 20, 30, 40]):
    """
    Beam search in latent space during diffusion process.
    At specified steps, generate multiple candidates and keep the best ones. 
    """ 
    device = getattr(pipe, "_execution_device", pipe.device)
    
    # Encode prompt
    text_inputs = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, 
                                truncation=True, return_tensors="pt")
    text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]
    text_embeddings = text_embeddings.to(device=pipe.device, dtype=pipe.unet.dtype)
    
    # Unconditional embeddings for classifier-free guidance
    uncond_input = pipe.tokenizer([""], padding="max_length", max_length=pipe.tokenizer.model_max_length, 
                                 return_tensors="pt")
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]
    uncond_embeddings = uncond_embeddings.to(device=pipe.device, dtype=pipe.unet.dtype)
    
    # Combine embeddings 
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    # Initialize latents
    latents_shape = (1, pipe.unet.config.in_channels, height // 8, width // 8) 
    latents = torch.randn(latents_shape, device=device, dtype=pipe.unet.dtype)
    latents = latents * pipe.scheduler.init_noise_sigma 
    
    # Beam search candidates: [(latent, score)]
    beam_candidates = [(latents.clone(), 0.0)] 
    
    # Denoising loop with beam search 
    pipe.scheduler.set_timesteps(num_steps) 
    
    for i, t in enumerate(pipe.scheduler.timesteps):
        if i in search_steps and len(beam_candidates) < num_beams:
            # Expand beam: create variations of current best candidates
            new_candidates = [] 
            
            for latent, score in beam_candidates[:num_beams//2]:  # Keep top half
                # Create variations by adding small noise
                for _ in range(2):
                    noise_scale = 0.1 * float(t) / float(pipe.scheduler.timesteps[0])
                    variation = latent + torch.randn_like(latent) * noise_scale
                    new_candidates.append((variation, score))
            
            beam_candidates.extend(new_candidates) 
        
        # Process all candidates
        scored_candidates = []
        
        for latent, prev_score in beam_candidates:
            # Predict noise
            latent_model_input = torch.cat([latent] * 2)
            t_unet = torch.tensor([t], device=device, dtype=torch.long).repeat(latent_model_input.shape[0])
            noise_pred = pipe.unet(latent_model_input, t_unet, encoder_hidden_states=text_embeddings).sample
            
            # Classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Scheduler step
            latent = pipe.scheduler.step(noise_pred, t, latent).prev_sample
            
            # Score based on prediction quality (lower noise = better)
            score = prev_score - torch.mean(noise_pred**2).item()
            scored_candidates.append((latent, score))
        
        # Keep top candidates
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        beam_candidates = scored_candidates[:num_beams]
    
    # Return best latent
    best_latent = beam_candidates[0][0]
    
    # Decode to image
    with torch.no_grad():
        image = pipe.vae.decode(best_latent / pipe.vae.config.scaling_factor).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
    
    return Image.fromarray(image), beam_candidates


def best_of_n_search(pipe, 
                    prompt: str, 
                    n_candidates: int = 8, 
                    num_steps: int = 50,
                    height: int = 512, 
                    width: int = 512, 
                    guidance_scale: float = 7.5,
                    selection_step: int = 25):
    """
    Best-of-N search: Generate N candidates, evaluate at mid-point, continue with best.
    """
    device = getattr(pipe, "_execution_device", pipe.device)
    
    # Encode prompt
    text_inputs = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, 
                                truncation=True, return_tensors="pt")
    text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]
    
    # Unconditional embeddings
    uncond_input = pipe.tokenizer([""], padding="max_length", max_length=pipe.tokenizer.model_max_length, 
                                 return_tensors="pt")
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0] 
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    # Generate N initial latents
    latents_shape = (n_candidates, pipe.unet.config.in_channels, height // 8, width // 8)
    latents = torch.randn(latents_shape, device=device, dtype=text_embeddings.dtype)
    latents = latents * pipe.scheduler.init_noise_sigma
    
    pipe.scheduler.set_timesteps(num_steps) 
    
    # First phase: denoise all candidates to selection point
    for i, t in enumerate(pipe.scheduler.timesteps[:selection_step]):
        # Batch process all candidates
        latent_model_input = torch.cat([latents, latents])  # For CFG
        text_emb_batch = text_embeddings.repeat(n_candidates, 1, 1)
        
        t_unet = torch.tensor([t], device=device, dtype=torch.long).repeat(latent_model_input.shape[0])
        noise_pred = pipe.unet(latent_model_input, t_unet, encoder_hidden_states=text_emb_batch).sample
        
        # Split and apply CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
    
    # Selection: choose best candidate based on some criteria
    with torch.no_grad(): 
        # Score based on CLIP similarity or other metrics
        scores = []
        for i in range(n_candidates):
            # Simple scoring: lower total variation = smoother image
            score = -torch.var(latents[i]).item()
            scores.append(score)
        
        best_idx = np.argmax(scores) 
        best_latent = latents[best_idx:best_idx+1]
    
    # Second phase: continue with best candidate
    for i, t in enumerate(pipe.scheduler.timesteps[selection_step:]):
        latent_model_input = torch.cat([best_latent] * 2)
        t_unet = torch.tensor([t], device=device, dtype=torch.long).repeat(latent_model_input.shape[0])
        noise_pred = pipe.unet(latent_model_input, t_unet, encoder_hidden_states=text_embeddings[:2]).sample
        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        best_latent = pipe.scheduler.step(noise_pred, t, best_latent).prev_sample
    
    # Decode final image
    with torch.no_grad():
        image = pipe.vae.decode(best_latent / pipe.vae.config.scaling_factor).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
    
    return Image.fromarray(image), scores[best_idx]

