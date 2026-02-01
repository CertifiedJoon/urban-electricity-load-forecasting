import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """
    Splits the 40,320 sequence into patches (e.g., 1-hour blocks).
    Input: [Batch, Length, Channels] -> Output: [Batch, Num_Patches, Embed_Dim]
    """

    def __init__(self, patch_size=5, in_channels=4, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        # Linear projection that acts as the "Feature Extractor" for each hour
        self.projection = nn.Linear(patch_size * in_channels, embed_dim)

    def forward(self, x):
        # x shape: [Batch, Length, Channels]
        B, L, C = x.shape
        num_patches = L // self.patch_size

        # Reshape to separate patches
        # [Batch, Num_Patches, Patch_Size * Channels]
        x = x.view(B, num_patches, self.patch_size * C)

        # Project to embedding dimension
        return self.projection(x)


class StaticContextEncoder(nn.Module):
    """
    Encodes the socio-economic "DNA" (Income, Residents, Working Status).
    """

    def __init__(self, embed_dim=128):
        super().__init__()
        # Define embeddings based on the IDEAL metadata ranges
        self.emb_income = nn.Embedding(
            num_embeddings=15, embedding_dim=16
        )  # [cite: 38]
        self.emb_work = nn.Embedding(num_embeddings=10, embedding_dim=16)  # [cite: 78]
        self.emb_residents = nn.Embedding(
            num_embeddings=10, embedding_dim=8
        )  # [cite: 38]
        self.emb_hometype = nn.Embedding(
            num_embeddings=5, embedding_dim=8
        )  # [cite: 38]

        # MLP to fuse them into a single context vector
        self.fusion = nn.Sequential(
            nn.Linear(16 + 16 + 8 + 8, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x_static):
        # x_static columns: [income, work, residents, hometype]
        inc = self.emb_income(x_static[:, 0])
        wrk = self.emb_work(x_static[:, 1])
        res = self.emb_residents(x_static[:, 2])
        hom = self.emb_hometype(x_static[:, 3])

        # Concatenate and fuse
        concat = torch.cat([inc, wrk, res, hom], dim=1)
        return self.fusion(concat)  # Shape: [Batch, Embed_Dim]


class SocioTemporalTransformer(nn.Module):
    def __init__(
        self,
        seq_len=40320,
        patch_size=60,
        in_channels=4,
        static_dim=4,
        embed_dim=128,
        n_heads=8,
        n_layers=4,
    ):
        super().__init__()

        # 1. Encoders
        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)
        self.static_embed = StaticContextEncoder(embed_dim)

        # Positional Encoding (Learnable)
        self.num_patches = seq_len // patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # 2. Transformer Encoder (Processes the Temporal Sequence)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # 3. Cross-Attention Mechanism (Injects Socio-Economic Context)
        # Query = Temporal Sequence, Key/Value = Static Context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=n_heads, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

        # 4. Probabilistic Head (Gaussian: Mean & Variance)
        # We project back from Embed_Dim to original Patch_Size (for reconstruction/forecast)
        self.head_mu = nn.Linear(embed_dim, patch_size)
        self.head_sigma = nn.Linear(embed_dim, patch_size)
        self.softplus = nn.Softplus()  # Ensures sigma is positive

    def forward(self, x_dynamic, x_static):
        # A. Embed Dynamic Data
        # Input: [Batch, 40320, 4] -> Output: [Batch, 672, 128]
        h_temporal = self.patch_embed(x_dynamic)
        h_temporal = h_temporal + self.pos_embedding

        # B. Encode Temporal Patterns
        h_temporal = self.temporal_encoder(h_temporal)

        # C. Embed Static Context
        # Input: [Batch, 4] -> Output: [Batch, 1, 128]
        h_static = self.static_embed(x_static).unsqueeze(1)

        # D. Cross-Attention Fusion
        # The Temporal data "looks at" the Static context
        # We expand static to match batch size implies broadcasting is handled by MultiheadAttention if set up right,
        # but usually we need K, V to be valid sequences.
        # Here we treat the Static Context as a sequence of length 1.
        attn_out, _ = self.cross_attn(query=h_temporal, key=h_static, value=h_static)

        # Residual Connection & Norm
        h_fused = self.layer_norm(h_temporal + attn_out)

        # E. Output Heads (Predicting the next step or reconstructing)
        mu = self.head_mu(h_fused)  # Shape: [Batch, Num_Patches, Patch_Size]
        sigma = self.head_sigma(h_fused)
        sigma = self.softplus(sigma) + 1e-6  # Stability

        # Reshape back to linear sequence: [Batch, 40320]
        B, N, P = mu.shape
        mu = mu.view(B, N * P)
        sigma = sigma.view(B, N * P)

        return mu, sigma


def gaussian_nll_loss(target, mu, sigma):
    """
    Minimizes the negative log-likelihood of the Gaussian distribution.
    target: Actual energy consumption
    mu: Predicted mean
    sigma: Predicted standard deviation
    """
    # [cite: 132] electric-combined is in Watts, log-scaled
    var = sigma**2
    loss = 0.5 * (torch.log(var) + (target - mu) ** 2 / var)
    return loss.mean()
