import torch
import torch.nn as nn
from typing import Optional
import numpy as np
from .model.encoders import EncoderC, EncoderP
from .model.decoder import Decoder
from .model.blocks import AdvDiscriminator, HyperDelta, LowRankAdapter, DeltaHead
from .config import ModelConfig

class DualEncoderVAE(nn.Module):
    def __init__(self, cfg: ModelConfig,
                 num_batches: int,
                 num_celltypes: int,
                 num_genes_vocab: int,
                 gene_emb_init: Optional[np.ndarray] = None):
        super().__init__()
        self.cfg = cfg
        self.input_dim  = cfg.input_dim
        self.latent_dim = cfg.latent_dim

        # Embeddings for context categorical covariates. We will embed each separately then sum
        self.batch_emb = nn.Embedding(max(1, num_batches), self.cfg.context_dim)
        self.ct_emb    = nn.Embedding(max(1, num_celltypes), self.cfg.context_dim)
        self.h1_emb    = nn.Embedding(num_embeddings=2, embedding_dim=self.cfg.context_dim)  # {0,1}; no padding here

        # Target gene embedding (id 0 reserved for control)
        self.gene_emb = nn.Embedding(num_genes_vocab, self.cfg.gene_embed_dim, padding_idx=0)
        if gene_emb_init is not None:
            assert gene_emb_init.shape == (num_genes_vocab, self.cfg.gene_embed_dim), \
                f"gene_emb_init shape {gene_emb_init.shape} != ({num_genes_vocab}, {self.cfg.gene_embed_dim})"
            with torch.no_grad():
                self.gene_emb.weight.copy_(torch.from_numpy(gene_emb_init))
                if self.gene_emb.padding_idx is not None:
                    self.gene_emb.weight[self.gene_emb.padding_idx].zero_()
        else:
            # usual random init; zero control row
            nn.init.normal_(self.gene_emb.weight, mean=0.0, std=0.02)
            if self.gene_emb.padding_idx is not None:
                with torch.no_grad():
                    self.gene_emb.weight[self.gene_emb.padding_idx].zero_()

        self.enc_c = EncoderC(self.cfg.input_dim, cfg)
        self.enc_p = EncoderP(self.cfg.input_dim, cfg)
        
        # --- Delta head selection ---
        if cfg.delta_type.lower() == "hyper":
            self.delta_type = "hyper"
            self.delta_module = HyperDelta(z_dim=self.cfg.latent_dim,
                                           g_dim=self.cfg.gene_embed_dim,
                                           u_dim=self.cfg.context_dim,
                                           hidden=self.cfg.hidden_dims[-1],
                                           rank=self.cfg.delta_rank,
                                           dropout=self.cfg.dropout,
                                           use_ln=self.cfg.hyperdelta_use_ln
                                           )
        else:
            self.delta_type = "delta"            
            self.delta_module = DeltaHead(cfg)
        
        # Low-rank adapter (always present; no-op if A,B=None)
        self.adapter = LowRankAdapter(dim=self.cfg.latent_dim, rank=self.cfg.delta_rank)
        self.dec     = Decoder(cfg)
        self.use_lib = self.cfg.use_libsize_covariate
        
        if self.use_lib:
            h = self.cfg.libsize_proj_hidden
            self.lib_proj = nn.Sequential(
                nn.Linear(1, h), nn.GELU(),
                nn.Linear(h, self.cfg.context_dim),
                nn.LayerNorm(self.cfg.context_dim) if self.cfg.layernorm else nn.Identity(),
            )
                
        # Optional adversarial discriminator on z_c to enforce g-invariance
        if self.cfg.use_adv_invariance:
            self.adv = AdvDiscriminator(self.cfg.latent_dim, num_genes_vocab, self.cfg.adv_hidden) 
        else:
            self.adv = None
        
        # optional: per-gene weights plugged by training script
        self.gene_w = None
    
    def build_u(self, batch_id, ct_id, is_h1, libsize: torch.Tensor | None = None):
        """Build dense context vector u from categorical ids."""
        if batch_id.dtype != torch.long: batch_id = batch_id.long()
        if ct_id.dtype    != torch.long: ct_id    = ct_id.long()
        if is_h1.dtype    != torch.long: is_h1    = is_h1.long()
        
        u = self.batch_emb(batch_id) + self.ct_emb(ct_id) + self.h1_emb(is_h1)
        if self.use_lib and (libsize is not None):
            if libsize.dim() == 1:
                libsize = libsize.unsqueeze(-1)
            u = u + self.lib_proj(libsize.float())
        return u
        
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    # Helper to get Δ,(A,B) for a batch of (g,u)
    def _delta_pack(self, g: torch.Tensor, u: torch.Tensor):
        
        # hyper Δ → base delta_vec + low-rank (A,B)
        if self.delta_type == "hyper":
            dpack = self.delta_module(g, u)
            return dpack["delta_vec"], dpack["A"], dpack["B"]
        
        else:
            Delta_vec = self.delta_module(g, u)
            return Delta_vec, None, None

    def _apply_adapter(self, z, A, B):
        """Call low-rank adapter if A,B are provided, else identity."""
        return self.adapter(z, A, B) if (A is not None and B is not None) else z

    def forward(self, batch):
        x         = batch["x"]
        gid       = batch["gid"].long()
        is_control = batch["is_control"].float()
        lib = batch.get("libsize", None)
        u = self.build_u(batch["batch_id"], batch["ct_id"], batch["is_h1"], lib)
        g = self.gene_emb(gid)

        mu_c, logvar_c = self.enc_c(x, u)
        z_c = self.reparameterize(mu_c, logvar_c)

        mu_p, logvar_p = self.enc_p(x, u, g)
        z_p = self.reparameterize(mu_p, logvar_p)
        
        Delta_vec, A, B = self._delta_pack(g, u)

        # reconstructions
        x_rec_c = self.dec(z_c, u)
        x_rec_p = self.dec(z_p, u)
        
        # cross predictions with vector Δ and adapter
        z_cp = self._apply_adapter(z_c + Delta_vec, A, B)
        z_pc = self._apply_adapter(z_p - Delta_vec, A, B)
        
        x_pred_from_c = self.dec(z_cp, u)
        x_pred_from_p_to_c = self.dec(z_pc, u)
        
        out = {"mu_c": mu_c,
               "logvar_c": logvar_c,
               "z_c": z_c,
               "mu_p": mu_p,
               "logvar_p": logvar_p,
               "z_p": z_p,
               "Delta": Delta_vec,
               "A": A,
               "B": B,
               "u": u,
               "g": g,
               "x_rec_c": x_rec_c,
               "x_rec_p": x_rec_p,
               "x_pred_from_c": x_pred_from_c,
               "x_pred_from_p_to_c": x_pred_from_p_to_c,
               "is_control": is_control,
               "gid": gid,
               }
        return out

    # Inference utilities
    @torch.no_grad()
    def encode_controls(self, x, batch_id, ct_id, is_h1, libsize=None, sample: bool = False):
        """
        Encode controls. If sample=True, draw z ~ N(mu, sigma^2) once.
        If sample=False, return mu (deterministic).
        """
        u = self.build_u(batch_id, ct_id, is_h1, libsize=libsize)
        mu_c, logvar_c = self.enc_c(x, u)
        if sample:
            std = torch.exp(0.5 * logvar_c)
            eps = torch.randn_like(std)
            z = mu_c + eps * std
        else:
            z = mu_c
        return z, u
    
    @torch.no_grad()
    def predict_perturbation(
        self,
        z_c: torch.Tensor,                 # [B, Dz]
        u: torch.Tensor,                   # [B, Du]
        gid: torch.Tensor,                 # [B] (long)
        neighbor_mix_k: int | None = None, # if None -> cfg.neighbor_mix_k
        neighbor_mix_tau: float | None = None,  # if None -> cfg.neighbor_mix_tau
        include_self_in_mix: bool = True,
        delta_gain: float = 1.0,           # <--- NEW: amplify perturbation magnitude at inference
    ):
        """
        Predict perturbed expression from control latents with optional neighbor mixing
        in gene-embedding space. 'delta_gain' scales the final Δ before adaptation/decoding.
        """
        # ---- resolve knobs
        k   = int(neighbor_mix_k  if neighbor_mix_k  is not None else int(self.cfg.neighbor_mix_k))
        tau = float(neighbor_mix_tau if neighbor_mix_tau is not None else float(self.cfg.neighbor_mix_tau))
        tau = max(1e-6, tau)  # numeric safety

        device = z_c.device
        gid    = gid.long().to(device)
        B      = z_c.size(0)

        # ---- query gene embeddings
        g = self.gene_emb(gid)  # [B, Dg]
        
        # ---- no neighbor mixing → single gene
        if k <= 0:
            Delta_vec, A_mix, B_mix = self._delta_pack(g, u)     # Δ: [B, Dz]
            # amplify Δ just before applying adapter
            if delta_gain != 1.0:
                Delta_vec = Delta_vec * float(delta_gain)
            z_adapt = self._apply_adapter(z_c + Delta_vec, A_mix, B_mix)
            return self.dec(z_adapt, u)

        # ---- neighbor mixing (exclude control id 0 from the pool)
        E_all  = self.gene_emb.weight          # [G, Dg]
        E_pool = E_all[1:]                     # [G-1, Dg]
        Gpool  = E_pool.size(0)
        k_eff  = max(1, min(k, Gpool))

        # cosine similarity in embedding space (normalized)
        g_n = torch.nn.functional.normalize(g, dim=-1)       # [B, Dg]
        E_n = torch.nn.functional.normalize(E_pool, dim=-1)  # [G-1, Dg]
        sims = g_n @ E_n.t()                                 # [B, G-1]

        topv, topi = torch.topk(sims, k=k_eff, dim=1)        # [B, k_eff], [B, k_eff]

        if include_self_in_mix:
            C_g = torch.cat([g.unsqueeze(1), E_pool[topi]], dim=1)          # [B, K, Dg]
            C_s = torch.cat([torch.ones(B, 1, device=device), topv], dim=1) # [B, K]
        else:
            C_g = E_pool[topi]                                              # [B, k_eff, Dg]
            C_s = topv                                                      # [B, k_eff]
        K = C_g.size(1)
        w = torch.softmax(C_s / tau, dim=1)                                 # [B, K]

        # Tile u to match K neighbors
        Bu = u.unsqueeze(1).expand(B, K, u.size(-1)).contiguous()           # [B, K, Du]

        # Flatten (B*K, ·) → run the Δ head once
        g_flat = C_g.reshape(B * K, C_g.size(-1))
        u_flat = Bu.reshape(B * K, Bu.size(-1))

        Δ_flat, A_flat, B_flat = self._delta_pack(g_flat, u_flat)           # Δ: [B*K, Dz]

        # Unflatten back to [B, K, ·]
        Δ_all = Δ_flat.view(B, K, -1)                                       # [B, K, Dz]
        A_all = B_all = None
        if A_flat is not None and B_flat is not None:
            A_all = A_flat.view(B, K, self.latent_dim, -1)                  # [B, K, Dz, r]
            B_all = B_flat.view(B, K, self.latent_dim, -1)                  # [B, K, Dz, r]

        # Weight and sum across neighbors
        w_vec = w.unsqueeze(-1)                                             # [B, K, 1]
        Δ_mix = (w_vec * Δ_all).sum(dim=1)                                  # [B, Dz]

        if A_all is not None and B_all is not None:
            w_ab = w.view(B, K, 1, 1)                                       # [B, K, 1, 1]
            A_mix = (w_ab * A_all).sum(dim=1)                               # [B, Dz, r]
            B_mix = (w_ab * B_all).sum(dim=1)                               # [B, Dz, r]
        else:
            A_mix = B_mix = None

        # Amplify final Δ, then adapt & decode
        if delta_gain != 1.0:
            Δ_mix = Δ_mix * float(delta_gain)

        z_adapt = self._apply_adapter(z_c + Δ_mix, A_mix, B_mix)
        return self.dec(z_adapt, u)

    # ---- Utilities to grow embeddings if label spaces expand ----
    def _extend_embedding(self, emb: nn.Embedding, new_num: int, init_std: float = 0.02) -> nn.Embedding:
        old_num, dim = emb.num_embeddings, emb.embedding_dim
        if new_num <= old_num:
            return emb
        device = emb.weight.device
        new_emb = nn.Embedding(new_num, dim, padding_idx=emb.padding_idx).to(device)
        with torch.no_grad():
            new_emb.weight[:old_num].copy_(emb.weight)
            nn.init.normal_(new_emb.weight[old_num:], mean=0.0, std=init_std)
            if emb.padding_idx is not None:
                new_emb.weight[emb.padding_idx].zero_()
        return new_emb

    def extend_context_spaces(self, num_batches: int = None, num_celltypes: int = None, init_std: float = 0.02):
        if num_batches is not None:
            self.batch_emb = self._extend_embedding(self.batch_emb, max(1, num_batches), init_std)
        if num_celltypes is not None:
            self.ct_emb = self._extend_embedding(self.ct_emb, max(1, num_celltypes), init_std)