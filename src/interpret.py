import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_rolling_week(model, dataset, home_id, device="cuda"):
    model.eval()
    
    # 1. Grab a continuous 1-week block + the required history
    # Let's say history is 60 days (86400 mins) and we want to plot 1 week (10080 mins)
    history_len = 86400 
    plot_len = 10080 
    
    # Load raw data for the specific home
    # (Assuming your dataset has a way to get a specific home's continuous stream)
    full_data, static_feat = dataset.get_full_home_stream(home_id) 
    
    predictions = []
    uncertainties = []
    actuals = []
    # To store attention: [Plot_Steps, Num_Patches]
    attention_accumulator = []

    print(f"Generating rolling forecast for Home {home_id}...")
    
    with torch.no_grad():
        # Step through the week in 10-minute jumps to save time (or 1-min for high-res)
        for t in range(history_len, history_len + plot_len, 30):
            # Slice the month of history
            x_hist = full_data[t - history_len : t].unsqueeze(0).to(device)
            s_feat = static_feat.unsqueeze(0).to(device)
            
            # Predict next 4 hours
            mu, sigma, attn = model(x_hist, s_feat)
            
            # Grab the 1st step of the prediction for the continuous line
            predictions.append(mu[0, 0].item())
            uncertainties.append(sigma[0, 0].item())
            actuals.append(full_data[t, 0].item())
            
            # ATTENTION INTERPRETATION
            # attn shape: [Heads, Q_len, K_len]
            # Since we use the last token to predict, we look at attn[0, :, -1, :]
            # Average across heads
            avg_attn = attn[:, -1, :].mean(dim=1).cpu().numpy()
            attention_accumulator.append(avg_attn)

    # 3. Aggregate Attention by Day (4320 patches -> 60 days)
    # Each day has (1440 / patch_size) patches. 
    patches_per_day = 1440 // 30 # 48
    attn_matrix = np.array(attention_accumulator) # [Time, 2880]
    
    # Reshape to [Time, Days, Patches_per_day] and mean over patches
    daily_attn = attn_matrix.reshape(len(attention_accumulator), , patches_per_day).mean(axis=2)

    # 4. PLOTTING
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Top: Power Prediction
    time_axis = np.arange(len(predictions))
    ax1.plot(time_axis, actuals, label='Actual Power', color='black', alpha=0.5)
    ax1.plot(time_axis, predictions, label='4hr Forecast (Lead Step)', color='blue')
    ax1.fill_between(time_axis, 
                     np.array(predictions) - np.array(uncertainties), 
                     np.array(predictions) + np.array(uncertainties), 
                     color='blue', alpha=0.2, label='Uncertainty ($\sigma$)')
    ax1.set_ylabel("Watts (Normalized)")
    ax1.legend()
    ax1.set_title(f"Rolling Weekly Forecast: Home {home_id}")

    # Bottom: Attention Heatmap (Days back)
    sns.heatmap(daily_attn.T, ax=ax2, cmap="YlGnBu", cbar_kws={'label': 'Attention Weight'})
    ax2.set_ylabel("Days in History")
    ax2.set_xlabel("Time during Test Week")
    # Invert Y so "1 day ago" is at the top
    ax2.set_yticklabels(np.arange(60, 0, -5)) 
    
    plt.tight_layout()
    plt.save_fig("interpretation.png")