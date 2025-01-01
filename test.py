import torch
import matplotlib.pyplot as plt

# Define the parameters
logit_mean = -6.0
logit_std = 2.0
batch_size = 1000

# Generate and transform values using the second function
u1 = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
u1 = torch.nn.functional.sigmoid(-u1 / 2)

# Generate and transform values using the first function with adjusted parameters
# adjusted_logit_mean = 3  # Adjusted mean
# adjusted_logit_std = 1.0    # Adjusted standard deviation
adjusted_logit_mean = 0.0  # Adjusted mean
adjusted_logit_std = 1.0    # Adjusted standard deviation
u2 = torch.normal(mean=adjusted_logit_mean, std=adjusted_logit_std, size=(batch_size,), device="cpu")
u2 = torch.nn.functional.sigmoid(u2)

# Plot the histograms
plt.hist(u1.numpy(), bins=50, density=True, alpha=0.6, color='b', label=f'logit_snr sigmoid(-u1 / 2) (mean={logit_mean}, std={logit_std})')
plt.hist(u2.numpy(), bins=50, density=True, alpha=0.6, color='r', label=f'logit_normal sigmoid(u2) (mean={adjusted_logit_mean}, std={adjusted_logit_std})')
plt.title('Comparison of Probability Distributions')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.show()