import torch
import torch.nn as nn


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


class InterpretableSocioTransformer(nn.Module):
    def __init__(
        self,
        cardinalities,
        dynamic_features=1,
        patch_size=10,
        embed_dim=512,
        num_head=8,
        num_layers=3,
        smoke_test=False,
    ):
        super().__init__()
        if smoke_test:
            embed_dim = 64
            num_head = 2
            num_layers = 1
        self.patch_size = patch_size
        self.patch_embed = PatchEmbedding(patch_size, dynamic_features, embed_dim)

        # New: Static encoder returns individual tokens [Batch, 4, Dim]
        # instead of a single summed vector
        self.static_embeddings = nn.ModuleList(
            [nn.Embedding(card, embed_dim) for card in cardinalities]
        )

        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_head, batch_first=True
            ),
            num_layers=num_layers,
        )

        # The Fusion Bridge: Multihead Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_head, batch_first=True
        )

        self.mu_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),  # Changed from 240 to 1
        )

        self.sigma_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),  # Changed from 240 to 1
        )

        self.static_dropout = nn.Dropout(p=0.2)

    def forward(self, x_dyn, x_stat):
        B, L, C = x_dyn.shape

        # 1. Process Time
        h_time = self.patch_embed(x_dyn)
        h_time = self.temporal_encoder(h_time)

        # 2. Process Static Features as separate tokens
        h_static = torch.stack(
            [emb(x_stat[:, i]) for i, emb in enumerate(self.static_embeddings)], dim=1
        )
        h_static = self.static_dropout(h_static)

        # 3. Cross-Attention Fusion
        # Query: Time | Key/Value: Static
        # attn_weights gives us the interpretability!
        h_fused, attn_weights = self.cross_attn(
            query=h_time, key=h_static, value=h_static
        )

        h_final = h_fused[:, -1, :]

        # 4. Output
        mu = self.mu_head(h_final)
        sigma = torch.exp(self.sigma_head(h_final)) + 1e-6

        # return weights for interpretability
        if self.training:
            return mu, sigma
        else:
            return mu, sigma, attn_weights
