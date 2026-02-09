import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    """Splits 40320 sequence into 60-min patches."""
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
    
class SocioTemporalTransformer(nn.Module):
    def __init__(self, seq_len=40320, patch_size=10, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        
        self.patch_embed = PatchEmbedding(patch_size, 1, embed_dim)
        self.static_embed = StaticContextEncoder(embed_dim)
        
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        # Transformer Core
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=4)
        
        # Probabilistic Heads
        self.head_mu = nn.Linear(embed_dim, patch_size)
        self.head_sigma = nn.Linear(embed_dim, patch_size)

    def generate_causal_mask(self, sz):
        # Mask is 2D: [sz, sz]. -inf means 'cannot look here'
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, x_dyn, x_stat):
        B, L, C = x_dyn.shape
        
        # 1. Temporal Encoding
        h_time = self.patch_embed(x_dyn) + self.pos_emb
        
        # 2. Apply Causal Mask so it can't see the future
        mask = self.generate_causal_mask(self.num_patches).to(x_dyn.device)
        
        # 3. Transform
        h_time = self.transformer(h_time, mask=mask)
        
        # 4. Mix in Static context (DNA) via addition or concat
        h_static = self.static_embed(x_stat) # [B, 1, Dim]
        h_fused = h_time + h_static # Broadcast static DNA across all time patches
        
        # 5. Output
        mu = self.head_mu(h_fused).view(B, -1)
        sigma = torch.exp(self.head_sigma(h_fused).view(B, -1)) + 1e-6
        
        return mu, sigma