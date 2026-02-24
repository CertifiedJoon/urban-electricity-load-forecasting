import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def visualize_rolling_week_point(model, dataset, home_id, device="cuda"):
    model.eval()
    history_len = 86400  # 60 Days
    lead_time = 240  # 4 Hours
    plot_len = 10080  # 1 Week
    step_jump = 10  # Jump in minutes for plotting speed

    full_data, static_feat = dataset.get_full_home_stream(home_id)

    predictions, uncertainties, actuals_at_target = [], [], []
    attention_accumulator = []

    print(f"Generating 4-hour Lead-Time Forecast for Home {home_id}...")

    with torch.no_grad():
        for t in range(history_len, history_len + plot_len, step_jump):
            # 1. Slice history
            x_hist = full_data[t - history_len : t].unsqueeze(0).to(device)
            s_feat = static_feat.unsqueeze(0).to(device)

            # Check for "11 patch" bug: x_hist should be [1, 86400, 1]
            if x_hist.shape[1] != history_len:
                print(f"Warning: Unexpected history slice size: {x_hist.shape}")

            mu, sigma, attn = model(x_hist, s_feat)

            # 2. Store results
            predictions.append(mu.item())
            uncertainties.append(sigma.item())

            # ACTUAL: We compare the prediction made at 't' with the reality at 't + 240'
            actuals_at_target.append(full_data[t + lead_time, 0].item())

            # 3. Attention (Last token)
            avg_attn = attn[0, :, -1, :].mean(dim=0).cpu().numpy()
            attention_accumulator.append(avg_attn)

    # --- FIXING THE RESHAPE ERROR ---
    attn_matrix = np.array(attention_accumulator)
    num_steps, num_patches = attn_matrix.shape

    # Calculate days based on actual patches returned (e.g., 8640)
    # Total minutes (86400) / Num patches (8640) = Patch Size (10)
    # Patches per day = 1440 / Patch Size
    patches_per_day = 1440 // (history_len // num_patches)
    num_days_detected = num_patches // patches_per_day

    # Dynamic Reshape
    daily_attn = (
        attn_matrix[:, : num_days_detected * patches_per_day]
        .reshape(num_steps, num_days_detected, patches_per_day)
        .mean(axis=2)
    )

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    time_axis = np.arange(len(predictions)) * step_jump / 60  # Convert to hours

    # Top: Point Prediction vs Future Reality
    ax1.plot(
        time_axis, actuals_at_target, label="Actual (at t+4h)", color="black", alpha=0.3
    )
    ax1.plot(time_axis, predictions, label="Forecast (for t+4h)", color="green")
    ax1.fill_between(
        time_axis,
        np.array(predictions) - np.array(uncertainties),
        np.array(predictions) + np.array(uncertainties),
        color="green",
        alpha=0.1,
    )
    ax1.set_ylabel("Power (Normalized)")
    ax1.set_title(f"4-Hour Lead-Time Rolling Forecast: Home {home_id}")
    ax1.legend()

    # Bottom: Attention Heatmap
    sns.heatmap(daily_attn.T, ax=ax2, cmap="YlGnBu", cbar=False)
    ax2.set_ylabel("Days Ago")
    ax2.set_xlabel("Hours into Testing Week")

    # Correct Y-labels for 60 days
    yticks = np.linspace(0, num_days_detected, 6)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([f"{int(num_days_detected - d)}d" for d in yticks])

    plt.tight_layout()
    plt.show()
