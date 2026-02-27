import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    """Gated Linear Unit (Equation 2 in the paper)"""

    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model * 2)

    def forward(self, x):
        x = self.fc(x)
        out, gate = x.chunk(2, dim=-1)
        return out * torch.sigmoid(gate)


class GRN(nn.Module):
    """Gated Residual Network (Equation 3 in the paper)"""

    def __init__(self, d_input, d_model, dropout=0.1, context_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(d_input, d_model)
        # Context vector addition (Equation 4)
        self.context_projection = (
            nn.Linear(context_dim, d_model, bias=False) if context_dim else None
        )
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(d_model)
        self.skip = nn.Linear(d_input, d_model) if d_input != d_model else nn.Identity()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, context=None):
        h = self.fc1(x)
        if self.context_projection is not None and context is not None:
            # Add context before ELU activation
            h = h + self.context_projection(context).unsqueeze(1)
        h = self.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        h = self.glu(h)
        return self.norm(self.skip(x) + h)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network (Equation 6-8 in the paper)"""

    def __init__(self, num_vars, d_model, dropout=0.1):
        super().__init__()
        self.num_vars = num_vars
        self.weight_network = GRN(num_vars * d_model, num_vars, dropout=dropout)
        self.variable_networks = nn.ModuleList(
            [GRN(d_model, d_model, dropout=dropout) for _ in range(num_vars)]
        )

    def forward(self, embeddings):
        # embeddings shape: [Batch, Num_Vars, d_model]
        flattened = embeddings.flatten(start_dim=1)  # [Batch, Num_Vars * d_model]

        # Calculate selection weights (Softmax over variables)
        selection_weights = torch.softmax(
            self.weight_network(flattened), dim=-1
        ).unsqueeze(-1)

        # Process each variable through its own GRN
        processed_vars = torch.stack(
            [net(embeddings[:, i]) for i, net in enumerate(self.variable_networks)],
            dim=1,
        )

        # Weighted sum of variables
        fused_context = (selection_weights * processed_vars).sum(dim=1)
        return fused_context, selection_weights.squeeze(-1)


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


class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        cardinalities,
        patch_size=10,
        d_model=256,  # Keep at 128 or 256 for 16GB GPU
        num_heads=4,
        dropout=0.1,
        smoke_test=False,
    ):
        super().__init__()
        if smoke_test:
            d_model = 64
            num_heads = 2
        self.d_model = d_model

        # 1. Temporal Patching
        self.patch_embed = PatchEmbedding(patch_size, 1, d_model)

        # 2. Static Covariate Encoders & VSN
        self.num_static = len(cardinalities)
        self.static_embeddings = nn.ModuleList(
            [nn.Embedding(c, d_model) for c in cardinalities]
        )
        self.static_vsn = VariableSelectionNetwork(self.num_static, d_model, dropout)

        # Static Context Generators (Equation 10-12)
        self.context_h = nn.Linear(d_model, d_model)  # LSTM initial hidden state
        self.context_c = nn.Linear(d_model, d_model)  # LSTM initial cell state
        self.context_enrichment = nn.Linear(d_model, d_model)

        # 3. Local Processing (LSTM)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.post_lstm_gate = GLU(d_model)
        self.post_lstm_norm = nn.LayerNorm(d_model)

        # 4. Static Enrichment
        self.static_enrichment_grn = GRN(d_model, d_model, dropout, context_dim=d_model)

        # 5. Temporal Self-Attention (Using PyTorch standard for memory efficiency)
        # Note: TFT uses shared values, but standard MHA is faster and optimized for 16GB
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.post_attn_gate = GLU(d_model)
        self.post_attn_norm = nn.LayerNorm(d_model)

        # 6. Quantile Output Head (Predicting 10th, 50th, 90th percentiles)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ELU(),
            nn.Linear(d_model // 2, 3),  # 3 outputs for 3 quantiles
        )

    def forward(self, x_dyn, x_stat):
        B, L, C = x_dyn.shape

        # --- A. Process Static Covariates ---
        # Embed all 11 static features
        static_embs = torch.stack(
            [emb(x_stat[:, i]) for i, emb in enumerate(self.static_embeddings)], dim=1
        )

        # VSN creates a single rich context vector and gives us feature importance weights
        c_static, static_weights = self.static_vsn(static_embs)

        # Generate LSTM initial states based entirely on Socio Features
        h_0 = self.context_h(c_static).unsqueeze(0)  # [1, Batch, d_model]
        c_0 = self.context_c(c_static).unsqueeze(0)  # [1, Batch, d_model]
        c_e = self.context_enrichment(c_static)  # [Batch, d_model]

        # --- B. Process Temporal Data ---
        # Patch the 86,400 minutes -> 2,880 patches
        h_time = self.patch_embed(x_dyn)  # [Batch, Patches, d_model]

        # Local Processing (LSTM) conditioned on Socio Features
        lstm_out, _ = self.lstm(h_time, (h_0, c_0))
        lstm_out = self.post_lstm_norm(h_time + self.post_lstm_gate(lstm_out))

        # Static Enrichment (Inject socio context into every time patch)
        enriched = self.static_enrichment_grn(lstm_out, context=c_e)

        # --- C. Temporal Self-Attention ---
        # The model finds long-term dependencies (e.g., Weekly Seasonality)
        attn_out, temporal_attn_weights = self.attention(enriched, enriched, enriched)
        fused = self.post_attn_norm(enriched + self.post_attn_gate(attn_out))

        # --- D. Forecast Head ---
        # Take the final token (representing 'now') to predict the future point
        final_token = fused[:, -1, :]
        quantiles = self.output_layer(final_token)  # [Batch, 3]

        if self.training:
            return quantiles
        else:
            return quantiles, temporal_attn_weights, static_weights
