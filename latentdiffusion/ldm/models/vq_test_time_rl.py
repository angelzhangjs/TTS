import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def _try_build_lpips(device: torch.device):
    """Return an LPIPS module if available, otherwise None."""
    try:
        import lpips  # type: ignore

        lp = lpips.LPIPS(net='alex')
        lp = lp.to(device)
        lp.eval()
        return lp
    except Exception:
        return None


class DefaultReward(nn.Module):
    """
    Default reward that measures improvement in reconstruction quality.

    Uses a blend of LPIPS (if available) and L1; otherwise, only L1.
    """

    def __init__(self, alpha: float = 0.8, device: Optional[torch.device] = None):
        super().__init__()
        self.alpha = alpha
        self.device = device or torch.device("cpu")
        self.lpips = _try_build_lpips(self.device)

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor, reconstructions: torch.Tensor) -> torch.Tensor:
        # inputs, reconstructions: (B, C, H, W) in [-1, 1]
        l1 = F.l1_loss(reconstructions, inputs, reduction="none").mean(dim=(1, 2, 3))
        if self.lpips is None:
            return -l1  # higher is better
        lp = self.lpips(inputs, reconstructions).squeeze(-1).squeeze(-1)
        return -(self.alpha * lp + (1.0 - self.alpha) * l1)


class _PolicyNet(nn.Module):
    """
    Tiny per-position policy over K+1 actions: {keep, replace with one of K candidates}.

    Input: prequant activations (B, C, H, W)
    Output: logits (B, K+1, H, W)
    """

    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        hidden = max(32, min(4 * in_channels, 256))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.GroupNorm(num_groups=min(8, hidden), num_channels=hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, num_actions, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VQTestTimeRLRefiner:
    """
    Test-time RL framework for VQ-VAE to improve generation outputs by editing codebook indices.

    High-level algorithm (REINFORCE):
      1) From input image x, obtain pre-quant activations h and predicted code indices ind.
      2) For each spatial location, pre-compute K candidate replacement indices from the codebook
         (nearest neighbors of the local pre-quant vector). Action space per location is K+1
         including a 'keep current index' action.
      3) Train a small policy network for T steps to maximize a reward (e.g., -LPIPS - L2)
         by sampling actions and updating policy via REINFORCE. Maintain the best reconstruction.

    Notes:
      - This is designed for small budgets and small images. For large images, set `max_positions`
        to limit the number of spatial positions being refined per forward.
      - Reward is batched per-image; gradients only flow into the policy parameters.
      - The underlying VQ model remains frozen.
    """

    def __init__(
        self,
        vq_model: nn.Module,
        embed_dim: int,
        n_embed: int,
        num_candidates: int = 4,
        steps: int = 50,
        lr: float = 1e-2,
        temperature: float = 1.0,
        max_positions: Optional[int] = 512,
        reward_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        seed: Optional[int] = 42,
    ):
        """
        vq_model: trained VQModel (ldm.models.autoencoder.VQModel) in eval mode
        embed_dim, n_embed: codebook shape (must match vq_model.quantize)
        num_candidates: K nearest-neighbor proposals per position
        steps: REINFORCE update steps
        lr: policy learning rate
        temperature: softmax temperature for sampling
        max_positions: limit number of spatial positions considered (subset)
        reward_fn: callable(inputs, reconstructions) -> tensor of shape (B,)
        seed: RNG seed for reproducibility
        """
        self.vq = vq_model
        self.vq.eval()
        self.device = next(self.vq.parameters()).device
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.K = int(num_candidates)
        self.T = int(steps)
        self.lr = float(lr)
        self.temp = float(temperature)
        self.max_positions = max_positions
        self.rng = torch.Generator(device=self.device)
        if seed is not None:
            self.rng.manual_seed(seed)

        # Build default reward if none provided
        if reward_fn is None:
            self.reward = DefaultReward(device=self.device)
        else:
            # Wrap arbitrary callable into nn.Module-like object
            class _Wrapper(nn.Module):
                def __init__(self, fn):
                    super().__init__()
                    self.fn = fn

                @torch.no_grad()
                def forward(self, a, b):
                    return self.fn(a, b)

            self.reward = _Wrapper(reward_fn)

        # Policy initialized lazily when input is seen (depends on channels)
        self.policy: Optional[_PolicyNet] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

    @torch.inference_mode()
    def _gather_codebook(self) -> torch.Tensor:
        """Returns codebook embeddings (n_embed, embed_dim)."""
        # VectorQuantizer2 typically exposes .embedding.weight of shape (n_embed, embed_dim)
        emb = self.vq.quantize.embedding.weight  # type: ignore[attr-defined]
        return emb.detach()

    @torch.no_grad()
    def _precompute_candidates(
        self, prequant: torch.Tensor, indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute K nearest codebook entries per position as candidate replacements.

        prequant: (B, C, H, W)
        indices: (B, H, W) current code indices

        Returns
          - pos_mask: (B, H, W) boolean mask of selected positions (possibly subset)
          - cand_idx: (B, K, H, W) candidate indices per position
          - cur_idx: (B, 1, H, W) current indices (for 'keep' action)
          - features: (B, C, H, W) reference features for the policy input (prequant)
        """
        B, C, H, W = prequant.shape
        codebook = self._gather_codebook()  # (N, C)

        # Flatten spatial grid to (B*H*W, C)
        feats = rearrange(prequant, "b c h w -> (b h w) c")  # (M, C)
        # Compute squared Euclidean distances to codebook: (M, N)
        # dist^2 = ||x||^2 + ||e||^2 - 2 x @ e^T
        x2 = (feats.pow(2).sum(dim=1, keepdim=True))  # (M, 1)
        e2 = (codebook.pow(2).sum(dim=1, keepdim=True)).T  # (1, N)
        logits = - (x2 + e2 - 2.0 * feats @ codebook.T)  # negative distances as similarity
        # Top-K per position
        topk_sim, topk_idx = logits.topk(k=min(self.K, self.n_embed), dim=1)  # (M, K)

        # Reshape back to (B, K, H, W)
        cand_idx = rearrange(topk_idx, "(b h w) k -> b k h w", b=B, h=H, w=W)
        cur_idx = indices[:, None, :, :]
        pos_mask = torch.ones((B, H, W), dtype=torch.bool, device=prequant.device)

        if self.max_positions is not None and self.max_positions < (H * W):
            # Randomly subsample positions to refine
            total = H * W
            sel = torch.rand((B, total), generator=self.rng, device=prequant.device).argsort(dim=1)
            take = self.max_positions
            keep_linear = torch.zeros((B, total), dtype=torch.bool, device=prequant.device)
            keep_linear.scatter_(1, sel[:, :take], True)
            pos_mask = keep_linear.view(B, H, W)

        return pos_mask, cand_idx, cur_idx, prequant

    def _init_policy(self, in_channels: int, num_actions: int):
        if self.policy is None:
            self.policy = _PolicyNet(in_channels=in_channels, num_actions=num_actions).to(self.device)
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def _sample_actions(self, logits: torch.Tensor, temperature: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        logits: (B, A, H, W)
        Returns: actions (B, 1, H, W), log_probs (B, H, W)
        """
        B, A, H, W = logits.shape
        flat_logits = rearrange(logits, "b a h w -> b (h w) a") / max(1e-6, temperature)
        dist = torch.distributions.Categorical(logits=flat_logits)
        a = dist.sample()  # (B, H*W)
        logp = dist.log_prob(a)  # (B, H*W)
        a = a.view(B, 1, H, W)
        logp = logp.view(B, H, W)
        return a, logp

    def _apply_actions(
        self,
        actions: torch.Tensor,
        pos_mask: torch.Tensor,
        cand_idx: torch.Tensor,
        cur_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build new index map from actions.

        actions: (B, 1, H, W) in [0..K] where 0 means KEEP, 1..K map to cand_idx[:, i-1]
        pos_mask: (B, H, W) which positions are editable; others are forced to KEEP
        cand_idx: (B, K, H, W)
        cur_idx: (B, 1, H, W)
        """
        B, _, H, W = actions.shape
        K = cand_idx.shape[1]

        # Force KEEP where not selected for editing
        forced_keep = (~pos_mask).unsqueeze(1)
        actions = actions.clone()
        actions[forced_keep] = 0

        # Build gather index for candidates: action 0 -> current, 1..K -> candidates
        all_choices = torch.cat([cur_idx, cand_idx], dim=1)  # (B, K+1, H, W)
        a_clamped = actions.clamp(min=0, max=K)
        # Gather along action dimension
        a_onehot = F.one_hot(a_clamped.squeeze(1), num_classes=K + 1).permute(0, 3, 1, 2).float()
        new_idx = (all_choices * a_onehot).sum(dim=1, keepdim=False).round().long()  # (B, H, W)
        return new_idx

    def refine(
        self,
        inputs: torch.Tensor,
        num_steps: Optional[int] = None,
        temperature: Optional[float] = None,
        return_best: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run test-time RL to refine code indices and improve reconstruction quality.

        inputs: (B, C, H, W) in [-1, 1]
        num_steps: override default steps
        temperature: override default temperature
        return_best: if True, returns the best reconstruction found

        Returns: (best_recon or last_recon, best_indices)
        """
        assert inputs.dim() == 4, "inputs must be (B, C, H, W)"
        B, C, H, W = inputs.shape
        steps = int(num_steps or self.T)
        temp = float(self.temp if temperature is None else temperature)

        # Freeze VQ model
        self.vq.eval()
        for p in self.vq.parameters():
            p.requires_grad_(False)

        # Obtain pre-quant activations and indices
        with torch.no_grad():
            prequant = self.vq.encode_to_prequant(inputs.to(self.device))  # (B, Cq, Hq, Wq)
            _, _, info = self.vq.quantize(prequant)
            # info is typically a tuple ending with indices
            try:
                _, _, indices = info  # type: ignore[misc]
            except Exception:
                # Fallback: call forward to get indices
                _, _, indices = self.vq(inputs, return_pred_indices=True)

        # Precompute candidate indices for each position
        pos_mask, cand_idx, cur_idx, feats = self._precompute_candidates(prequant, indices)
        _, Cq, Hq, Wq = feats.shape

        # Build policy net over (K+1) actions
        num_actions = self.K + 1
        self._init_policy(in_channels=Cq, num_actions=num_actions)
        assert self.policy is not None and self.optimizer is not None

        # Track best solution
        with torch.no_grad():
            base_recon = self.vq.decode_code(indices)
            base_reward = self.reward(inputs, base_recon)  # (B,)
            best_reward = base_reward.clone()
            best_recon = base_recon.clone()
            best_idx = indices.clone()

        # Moving baseline for REINFORCE
        running_baseline = base_reward.mean().item()
        beta = 0.9

        for t in range(steps):
            self.policy.train()
            self.optimizer.zero_grad(set_to_none=True)

            logits = self.policy(feats)  # (B, A, Hq, Wq)
            actions, logp = self._sample_actions(logits, temp)
            new_idx = self._apply_actions(actions, pos_mask, cand_idx, cur_idx)  # (B, Hq, Wq)

            with torch.no_grad():
                recon = self.vq.decode_code(new_idx)
                r = self.reward(inputs, recon)  # (B,)
                mean_r = r.mean().item()
                running_baseline = beta * running_baseline + (1 - beta) * mean_r

                # Track best
                improved = r > best_reward
                if improved.any():
                    best_reward = torch.where(improved, r, best_reward)
                    best_idx = torch.where(improved.view(B, 1, 1), new_idx, best_idx)
                    best_recon = torch.where(improved.view(B, 1, 1, 1), recon, best_recon)

            # REINFORCE loss
            advantage = (r - running_baseline)
            # Only count positions we acted on (editable mask)
            mask = pos_mask.float()  # (B, Hq, Wq)
            # Broadcast per-image advantage to spatial
            weighted_logp = (advantage.view(B, 1, 1) * logp).mean(dim=(1, 2))  # (B,)
            # Encourage higher reward => lower loss = -E[logp * advantage]
            loss = -(weighted_logp.mean())
            loss.backward()
            self.optimizer.step()

        if return_best:
            return best_recon.detach(), best_idx.detach()
        # Else, return the last sampled recon/indices
        with torch.no_grad():
            final_recon = self.vq.decode_code(new_idx)
        return final_recon.detach(), new_idx.detach()




