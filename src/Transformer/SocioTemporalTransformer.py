import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size * in_channels, embed_dim)

    def forward(self, x):
        B, L, C = x.shape
        num_patches = L // self.patch_size
        x = x.view(B, num_patches, self.patch_size * C)
        return self.projection(x)

class StaticContextEncoder(nn.Module):
    """Encodes the Household DNA."""
    def __init__(self, embed_dim):
        super().__init__()
        # Define embeddings for [residents, income, hometype, work]
        self.emb_res = nn.Embedding(20, 8)
        self.emb_inc = nn.Embedding(20, 16)
        self.emb_typ = nn.Embedding(10, 8)
        self.emb_wrk = nn.Embedding(10, 8)
        self.fusion = nn.Linear(8+16+8+8, embed_dim)

    def forward(self, x):
        # x shape: [Batch, 4]
        e1 = self.emb_res(x[:, 0])
        e2 = self.emb_inc(x[:, 1])
        e3 = self.emb_typ(x[:, 2])
        e4 = self.emb_wrk(x[:, 3])
        return self.fusion(torch.cat([e1, e2, e3, e4], dim=1)).unsqueeze(1)
    
class InterpretableSocioTransformer(nn.Module):
    def __init__(self, cardinalities, dynamic_features=1, patch_size=30, embed_dim=512, forecast_len=240):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbedding(patch_size, dynamic_features, embed_dim)
        
        # New: Static encoder returns individual tokens [Batch, 4, Dim]
        # instead of a single summed vector
        self.static_embeddings = nn.ModuleList([
            nn.Embedding(card, embed_dim) for card in cardinalities
        ])
        
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True),
            num_layers=3
        )
        
        # The Fusion Bridge: Multihead Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        
        # --- NEW: Forecast Decoder ---
        # Instead of projecting back to patch_size, we project the FINAL latent state 
        # to the entire forecast horizon (24 steps)
        self.forecast_head_mu = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, forecast_len) # Outputs [Batch, 24]
        )
        
        self.forecast_head_sigma = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, forecast_len) # Outputs [Batch, 24]
        )
        
    def forward(self, x_dyn, x_stat):
        B, L, C = x_dyn.shape
        
        # 1. Process Time
        h_time = self.patch_embed(x_dyn) 
        h_time = self.temporal_encoder(h_time) # [B, Num_Patches, Dim]
        
        # 2. Process Static Features as separate tokens
        # h_static shape: [B, 4, Dim]
        h_static = torch.stack([
            emb(x_stat[:, i]) for i, emb in enumerate(self.static_embeddings)
        ], dim=1)
        
        # 3. Cross-Attention Fusion
        # Query: Time | Key/Value: Static
        # attn_weights gives us the interpretability!
        h_fused, attn_weights = self.cross_attn(
            query=h_time, 
            key=h_static, 
            value=h_static
        )
        
        h_final = h_fused[:, -1, :] # [B, 128]
        
        # 4. Output
        mu = self.forecast_head_mu(h_final)     # [B, 24]
        sigma = torch.exp(self.forecast_head_sigma(h_final)) + 1e-6

        # return weights for interpretability
        if self.training:
            return mu, sigma
        else:
            return mu, sigma, attn_weights