import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys


def visualize_rolling_week_point(model, dataset, home_id, device="cuda"):
    model.eval()
    history_len = 43200  # 60 Days
    lead_time = 240  # 4 Hours
    plot_len = 10080  # 1 Week
    step_jump = 10  # Jump in minutes for plotting speed

    full_data, static_feat = dataset.get_full_home_stream(home_id)

    predictions, uncertainties, actuals_at_target = [], [], []
    attention_accumulator = []

    print(f"Generating 4-hour Lead-Time Forecast for Home {home_id}...")

    with torch.no_grad():
        for t in range(history_len, history_len + plot_len, step_jump):
            sys.stdout.write(f"\r Simulating {(t - history_len) / plot_len * 100}% ")
            sys.stdout.flush()

            # 1. Slice history
            x_hist = full_data[t - history_len : t].unsqueeze(0).to(device)
            s_feat = static_feat.unsqueeze(0).to(device)

            if x_hist.shape[1] != history_len:
                print(f"Warning: Unexpected history slice size: {x_hist.shape}")

            mu, sigma, attn = model(x_hist, s_feat)

            # 2. Store results
            predictions.append(dataset.denormalize(mu.item()))
            uncertainties.append(sigma.item() * dataset.stats["std"])

            # ACTUAL: We compare the prediction made at 't' with the reality at 't + 240'
            actuals_at_target.append(
                dataset.denormalize(full_data[t + lead_time, 0].item())
            )

            # 3. Attention (Last token)
            attention_accumulator.append(attn[0, -1, :].cpu().numpy())

    attn_matrix = np.array(attention_accumulator)

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    time_axis = np.arange(len(predictions))
    num_features = attn_matrix.shape[1]  # This is 11

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # --- TOP PLOT: POWER ---
    ax1.plot(
        time_axis, actuals_at_target, color="black", alpha=0.3, label="Actual (t+4h)"
    )
    ax1.plot(time_axis, predictions, color="green", label="Forecast")
    ax1.fill_between(
        time_axis,
        np.array(predictions) - np.array(uncertainties),
        np.array(predictions) + np.array(uncertainties),
        color="green",
        alpha=0.1,
    )
    ax1.set_ylabel("Power (Normalized)")
    ax1.legend(loc="upper right")

    # --- BOTTOM PLOT: ATTENTION ALIGNED ---
    # We use extent=[xmin, xmax, ymin, ymax] to map indices to Hours
    ax2.imshow(
        attn_matrix.T,
        aspect="auto",
        origin="upper",
        extent=[time_axis[0], time_axis[-1], num_features, 0],
        cmap="Blues",
    )
    # Clean up labels
    ax2.set_ylabel("Socio Features")
    ax2.set_xlabel("Time (Hours into Week)")

    # Force the x-axis to exactly the week range
    ax1.set_xlim(time_axis[0], time_axis[-1])

    plt.tight_layout()
    plt.savefig("interpretation.png")


def visualize_tft_rolling_week(
    model, dataset, home_id, feature_names=None, device="cuda", smoke_test=False
):
    model.eval()

    # Config
    history_mins = 43200  # 60 days
    lead_mins = 240  # 4 hours
    plot_len = 10080  # 1 week
    if smoke_test:
        plot_len = 300
    step_jump = 10  # Match patch size (10 mins)
    patches_past = history_mins // step_jump  # 4320

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(11)]

    full_power, full_time, static_feat = dataset.get_full_home_stream(home_id)

    q10_list, q50_list, q90_list, actuals = [], [], [], []
    temporal_attn_list = []
    vsn_weights_list = []

    print(f"Generating Seq2Seq Interpretable forecast for Home {home_id}...")

    with torch.no_grad():
        for t in range(history_mins, history_mins + plot_len, step_jump):
            x_past_power = full_power[t - history_mins : t].unsqueeze(0).to(device)
            x_past_time = full_time[t - history_mins : t].unsqueeze(0).to(device)
            x_future_time = full_time[t : t + lead_mins].unsqueeze(0).to(device)
            s_feat = static_feat.unsqueeze(0).to(device)

            # Forward pass
            # quantiles: [1, 8, 3] | attn_weights: [1, 2888, 2888] | static_weights: [1, 11]
            quantiles, attn_weights, static_weights = model(
                x_past_power, x_past_time, x_future_time, s_feat
            )

            # 1. Extract Prediction (Last patch of the 8-patch future sequence)
            q10, q50, q90 = (
                quantiles[0, -1, 0].item(),
                quantiles[0, -1, 1].item(),
                quantiles[0, -1, 2].item(),
            )

            if hasattr(dataset, "denormalize"):
                q10, q50, q90 = map(dataset.denormalize, [q10, q50, q90])
                actual_val = dataset.denormalize(full_power[t + lead_mins, 0].item())
            else:
                actual_val = full_power[t + lead_mins, 0].item()

            q10_list.append(q10)
            q50_list.append(q50)
            q90_list.append(q90)
            actuals.append(actual_val)

            # 2. Extract Temporal Attention
            # Look at the last future token (-1), and get its attention over the past history patches (:patches_past)
            past_attention = attn_weights[0, -1, :patches_past].cpu().numpy()
            temporal_attn_list.append(past_attention)

            # 3. Extract Static VSN weights
            vsn_weights_list.append(static_weights[0].cpu().numpy())

    # --- AGGREGATION ---
    time_axis = np.arange(len(q50_list)) * step_jump / 60  # Hours into the week

    # Reshape attention: 4320 patches -> 30 days of 144 patches each
    attn_matrix = np.array(temporal_attn_list)  # [Steps, 4320]
    daily_temporal_attn = attn_matrix.reshape(len(attn_matrix), 30, 144).mean(
        axis=2
    )  # [Steps, 60]

    avg_vsn_weights = np.array(vsn_weights_list).mean(axis=0)  # [11]

    # --- PLOTTING ---
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 1.5, 1])

    # Top Plot: Sequence Forecast + Uncertainty
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time_axis, actuals, color="black", alpha=0.5, label="Actual (+4h)")
    ax1.plot(time_axis, q50_list, color="darkgreen", label="P50 Forecast")
    ax1.fill_between(
        time_axis,
        q10_list,
        q90_list,
        color="mediumseagreen",
        alpha=0.3,
        label="P10 - P90 Interval",
    )
    ax1.set_xlim(time_axis[0], time_axis[-1])
    ax1.set_ylabel("Power Usage")
    ax1.set_title(f"Full TFT 4-Hour Lead Forecast: Home {home_id}")
    ax1.legend(loc="upper right")

    # Middle Plot: Temporal Attention (History)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    im = ax2.imshow(
        daily_temporal_attn.T,
        aspect="auto",
        origin="upper",
        cmap="Blues",
        extent=[time_axis[0], time_axis[-1], 30, 0],
    )
    ax2.set_ylabel("Days in Past")
    ax2.set_yticks(np.linspace(0, 30, 7))
    ax2.set_yticklabels([f"{int(d)}d" for d in np.linspace(30, 0, 7)])
    fig.colorbar(im, ax=ax2, pad=0.01, aspect=10, label="Attention Weight")

    # Bottom Plot: Variable Selection Network (Static Features) [Image of Variable Selection Network weights visualization]
    ax3 = fig.add_subplot(gs[2])
    sns.barplot(x=feature_names, y=avg_vsn_weights, ax=ax3, palette="viridis")
    ax3.set_ylabel("VSN Importance Weight")
    ax3.set_title("Static Feature Importance (Socio-Economic Profiles)")
    ax3.set_ylim(0, max(avg_vsn_weights) * 1.2)
    plt.xticks(rotation=15)

    plt.tight_layout()
    plt.savefig("interpretation.png")


from matplotlib.colors import LinearSegmentedColormap


def visualize_density_heatmap(model, dataset, home_id, device="cuda", smoke_test=False):
    model.eval()

    # Configuration
    history_mins = 43200  # 60 days
    lead_mins = 240  # 4 hours
    plot_len = 10080  # 1 week
    if smoke_test:
        plot_len = 300  # smoketest
    patch_size = 10  # 30-min steps

    # 1. Fetch Data
    full_power, full_time, static_feat = dataset.get_full_home_stream(home_id)

    num_time_steps = plot_len // patch_size
    time_axis = np.arange(num_time_steps) * (patch_size / 60)  # Converted to Hours

    # Extract the true actuals for the plotting window to set Y-axis boundaries
    start_idx = history_mins
    end_idx = history_mins + plot_len
    actuals = full_power[start_idx:end_idx, 0].numpy().reshape(-1, 10).mean(axis=1)

    if hasattr(dataset, "denormalize"):
        actuals = np.array([dataset.denormalize(val) for val in actuals])

    max_pow = np.max(actuals) * 1.5
    min_pow = max(0, np.min(actuals) - 0.5)
    num_power_bins = 250
    power_y = np.linspace(min_pow, max_pow, num_power_bins)

    # 2. Initialize the 2D Heatmap Matrix
    # Shape: [Power_Bins, Time_Steps]
    density_matrix = np.zeros((num_power_bins, num_time_steps))

    print(f"Generating 4-Hour Overlapping Heatmap for Home {home_id}...")

    with torch.no_grad():
        for step_idx in range(num_time_steps):
            t = history_mins + step_idx * patch_size

            # Prepare inputs
            x_past_power = full_power[t - history_mins : t].unsqueeze(0).to(device)
            x_past_time = full_time[t - history_mins : t].unsqueeze(0).to(device)
            x_future_time = full_time[t : t + lead_mins].unsqueeze(0).to(device)
            s_feat = static_feat.unsqueeze(0).to(device)

            # Forward pass (Handling the 3-item eval tuple return)
            if device == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    quantiles, _, _ = model(
                        x_past_power, x_past_time, x_future_time, s_feat
                    )
            else:
                quantiles, _, _ = model(
                    x_past_power, x_past_time, x_future_time, s_feat
                )

            # quantiles shape: [1, 8, 3] -> (P10, P50, P90)
            q_vals = quantiles[0].cpu().numpy()

            # 3. Project density onto the matrix for all 8 future steps
            for i in range(lead_mins // patch_size):
                target_step_idx = (
                    step_idx + i + 1
                )  # The exact time column this patch predicts

                # If the prediction falls within our plotting window
                if target_step_idx < num_time_steps:
                    q10, q50, q90 = q_vals[i, 0], q_vals[i, 1], q_vals[i, 2]

                    if hasattr(dataset, "denormalize"):
                        q10, q50, q90 = map(dataset.denormalize, [q10, q50, q90])

                    # Estimate Gaussian standard deviation from the P10-P90 spread.
                    # P10 to P90 covers ~2.56 standard deviations in a normal distribution
                    sigma = max((q90 - q10) / 2.56, 1e-3)

                    # Calculate Gaussian probability density over the entire Y-axis grid
                    pdf = np.exp(-0.5 * ((power_y - q50) / sigma) ** 2)

                    # Accumulate density
                    density_matrix[:, target_step_idx] += pdf

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(16, 6))

    # A dark colormap (inferno or magma) looks incredible for density plots
    cmap = plt.cm.inferno

    # Plot the heatmap
    im = ax.pcolormesh(time_axis, power_y, density_matrix, shading="auto", cmap=cmap)

    # Overlay the actual ground truth as a high-contrast line (cyan looks great on inferno)
    ax.plot(
        time_axis, actuals, color="cyan", linewidth=1.5, label="Actual Power", alpha=0.9
    )

    ax.set_title(f"Rolling 4-Hour Probability Density Heatmap - Home {home_id}")
    ax.set_xlabel("Hours into Test Week")
    ax.set_ylabel("Power Usage")
    ax.legend(loc="upper left")

    # Add colorbar for density magnitude
    fig.colorbar(im, ax=ax, label="Cumulative Forecast Consensus")

    plt.tight_layout()
    plt.show()
