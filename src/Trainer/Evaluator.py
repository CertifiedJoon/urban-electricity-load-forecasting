import matplotlib.pyplot as plt
import torch


def visualize_forecast(model, val_loader, device="cuda"):
    model.eval()
    batch = next(iter(val_loader))

    with torch.no_grad():
        mu, sigma = model(batch["x_dynamic"].to(device), batch["x_static"].to(device))

    # Take first sample in batch
    mu_sample = mu[0].cpu().numpy()
    sigma_sample = sigma[0].cpu().numpy()
    actual = batch["target"][0].cpu().numpy()

    # Inverse Log-Scaling to see Watts (Optional)
    # mu_sample = np.expm1(mu_sample)

    plt.figure(figsize=(15, 6))
    plt.plot(actual, label="Actual Load", color="black", alpha=0.6)
    plt.plot(mu_sample, label="Predicted Mean", color="blue")

    # Fill the probabilistic ribbon
    plt.fill_between(
        range(len(mu_sample)),
        mu_sample - 1.96 * sigma_sample,
        mu_sample + 1.96 * sigma_sample,
        color="blue",
        alpha=0.2,
        label="95% Confidence Interval",
    )

    plt.title("Household Load Forecast with Socio-Economic Context")
    plt.xlabel("Minutes (Month-long Window)")
    plt.ylabel("Log-Power (Watts)")
    plt.legend()
    plt.show()
