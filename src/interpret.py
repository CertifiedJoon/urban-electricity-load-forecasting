import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys


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
            sys.stdout.write(f"\r Simulating {(t - history_len) / plot_len * 100}% ")
            sys.stdout.flush()

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
        np.array(predictions) - 1,
        np.array(predictions) + 1,
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
