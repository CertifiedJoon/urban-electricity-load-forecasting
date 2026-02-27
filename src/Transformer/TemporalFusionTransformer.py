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
    def __init__(self, num_vars, d_model, dropout=0.1):
        super().__init__()
        self.num_vars = num_vars
        self.d_model = d_model

        # This GRN produces the selection weights
        self.weight_network = GRN(num_vars * d_model, num_vars, dropout=dropout)

        # These GRNs process each variable individually
        self.variable_networks = nn.ModuleList(
            [GRN(d_model, d_model, dropout=dropout) for _ in range(num_vars)]
        )

    def forward(self, embeddings):
        """
        embeddings:
          - Static:  [Batch, Num_Vars, d_model]
          - Temporal: [Batch, Time, Num_Vars, d_model]
        """
        # 1. Flatten only the last two dimensions (Num_Vars and d_model)
        # Static becomes [Batch, 96] | Temporal becomes [Batch, Time, 96]
        flattened = embeddings.flatten(start_dim=-2)

        # 2. Calculate selection weights
        # weights shape: [Batch, (Time), Num_Vars, 1]
        weights = torch.softmax(self.weight_network(flattened), dim=-1).unsqueeze(-1)

        # 3. Process each variable through its dedicated GRN
        # We handle 3D and 4D tensors dynamically
        if embeddings.dim() == 4:  # Temporal
            # Process variable i: embeddings[:, :, i, :]
            processed_vars = torch.stack(
                [
                    net(embeddings[:, :, i, :])
                    for i, net in enumerate(self.variable_networks)
                ],
                dim=-2,
            )
        else:  # Static
            processed_vars = torch.stack(
                [
                    net(embeddings[:, i, :])
                    for i, net in enumerate(self.variable_networks)
                ],
                dim=-2,
            )

        # 4. Weighted sum across the Num_Vars dimension
        fused_context = (weights * processed_vars).sum(dim=-2)

        return fused_context, weights.squeeze(-1)


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
        self, cardinalities, patch_size=10, d_model=256, num_heads=4, smoke_test=False
    ):
        super().__init__()
        if smoke_test:
            d_model = 32
            num_heads = 2
        self.patch_size = patch_size
        self.d_model = d_model

        # --- 1. Encoders & Patching ---
        self.power_patch_embed = nn.Linear(patch_size, d_model)

        # Time embeddings (Hour: 24, DayOfWeek: 7)
        self.hour_embed = nn.Embedding(24, d_model)
        self.day_embed = nn.Embedding(7, d_model)

        # Static embeddings
        self.static_embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=c, embedding_dim=d_model)
                for c in cardinalities
            ]
        )

        # --- 2. Variable Selection Networks (VSNs) ---
        self.static_vsn = VariableSelectionNetwork(len(cardinalities), d_model)
        self.past_vsn = VariableSelectionNetwork(3, d_model)  # Power, Hour, Day
        self.future_vsn = VariableSelectionNetwork(
            2, d_model
        )  # Hour, Day (Known future)

        # --- 3. Seq2Seq LSTM (Local Processing) ---
        self.encoder_lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.decoder_lstm = nn.LSTM(d_model, d_model, batch_first=True)

        # Static Context for LSTM initialization
        self.context_h = nn.Linear(d_model, d_model)
        self.context_c = nn.Linear(d_model, d_model)
        self.context_enrichment = nn.Linear(d_model, d_model)

        # --- 4. Attention & Output ---
        self.enrichment_grn = GRN(d_model, d_model, context_dim=d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ELU(),
            nn.Linear(d_model // 2, 3),  # [q10, q50, q90]
        )

    def forward(self, x_past_power, x_past_time, x_future_time, x_stat):
        B = x_past_power.size(0)

        # A. Static Context
        static_embs = torch.stack(
            [emb(x_stat[:, i]) for i, emb in enumerate(self.static_embeddings)], dim=1
        )
        c_static, static_weights = self.static_vsn(static_embs)

        # Initialize LSTM states using static context
        h_0 = self.context_h(c_static).unsqueeze(0)
        c_0 = self.context_c(c_static).unsqueeze(0)
        c_e = self.context_enrichment(c_static)

        # B. Patching & Past VSN
        # Reshape and project power: [B, 86400, 1] -> [B, 2880, patch_size] -> [B, 2880, d_model]
        past_power_patched = self.power_patch_embed(
            x_past_power.squeeze(-1).view(B, -1, self.patch_size)
        )

        # Subsample time features to match patches (take the first minute of each patch)
        past_hour = self.hour_embed(x_past_time[:, :: self.patch_size, 0])
        past_day = self.day_embed(x_past_time[:, :: self.patch_size, 1])

        past_fused, _ = self.past_vsn(
            torch.stack([past_power_patched, past_hour, past_day], dim=2)
        )

        # C. Future VSN
        future_hour = self.hour_embed(x_future_time[:, :: self.patch_size, 0])
        future_day = self.day_embed(x_future_time[:, :: self.patch_size, 1])

        future_fused, _ = self.future_vsn(torch.stack([future_hour, future_day], dim=2))

        # D. Encoder-Decoder LSTM
        # Run encoder on history
        past_fused = past_fused.contiguous()
        future_fused = future_fused.contiguous()
        h_0 = h_0.contiguous()
        c_0 = c_0.contiguous()
        encoder_out, (h_n, c_n) = self.encoder_lstm(past_fused, (h_0, c_0))

        # Run decoder on future (using the last state of the encoder)
        h_n = h_n.contiguous()
        c_n = c_n.contiguous()
        decoder_out, _ = self.decoder_lstm(future_fused, (h_n, c_n))

        # Combine encoder and decoder outputs for attention
        full_seq = torch.cat([encoder_out, decoder_out], dim=1)

        # E. Enrichment & Attention
        enriched = self.enrichment_grn(full_seq, context=c_e)
        attn_out, attn_weights = self.attention(enriched, enriched, enriched)

        # F. Sequence Forecast
        # We only want to predict the FUTURE steps (the last N patches)
        num_future_patches = future_fused.size(1)
        future_representations = attn_out[:, -num_future_patches:, :]

        # Output shape: [Batch, Future_Patches, 3_Quantiles]
        quantiles = self.output_layer(future_representations)
        if self.training:
            return quantiles
        else:
            return quantiles, attn_weights, static_weights
