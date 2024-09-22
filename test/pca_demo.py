import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Load the Stable Diffusion pipeline from Hugging Face Diffusers
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")

# Text prompts for evaluation
prompts = [
    "A red apple on a wooden table",
    "A futuristic car in a city at night",
    "A lion roaring in the savannah",
    "A painting of a surreal landscape with floating islands",
    "A medieval castle in the mountains",
    "A man hiking in the desert"
]

# Function to get latent representations from the T2I model (Stable Diffusion)
def get_latent_representations(prompts, pipe):
    latents = []
    
    for prompt in prompts:
        with torch.no_grad():
            # Generate the latents using the text prompt
            # The `latents` returned are before decoding into images
            latents_for_prompt = pipe(prompt, output_type="latent").latents
            latents.append(latents_for_prompt.cpu().numpy())
    
    latents = np.vstack([latent.reshape(1, -1) for latent in latents])  # Flatten latent tensors into 2D arrays
    return latents

# Get the latent representations for the prompts
latents = get_latent_representations(prompts, pipe)

# Apply PCA to latent representations
def apply_pca(latents, n_components=2):
    pca = PCA(n_components=n_components)
    latents_pca = pca.fit_transform(latents)
    
    explained_variance = pca.explained_variance_ratio_
    
    print(f"Explained variance by principal components: {explained_variance}")
    return latents_pca, explained_variance

# Apply PCA to reduce dimensions to 2D for visualization
latents_pca, explained_variance = apply_pca(latents)

# Visualize the 2D PCA result
def plot_pca(latents_pca, prompts):
    plt.figure(figsize=(10, 8))
    plt.scatter(latents_pca[:, 0], latents_pca[:, 1], c='blue', edgecolors='k', s=100)
    
    for i, prompt in enumerate(prompts):
        plt.text(latents_pca[i, 0] + 0.01, latents_pca[i, 1] + 0.01, prompt, fontsize=9)
    
    plt.title("PCA of Latent Space from Stable Diffusion")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.show()

plot_pca(latents_pca, prompts)

# # Function to generate an image from a prompt using the Stable Diffusion pipeline
# def generate_image(prompt, pipe):
#     image = pipe(prompt).images[0]
#     return image

# # Example: Generate an image for one of the prompts
# image = generate_image(prompts[0], pipe)
# image.show()