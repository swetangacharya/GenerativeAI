---
layout: default
title: "Deep Learning & Generative AI Hub"
description: "A comprehensive guide to neural network architectures, flow matching, and diffusion models."
---

# A Generative AI.

Welcome to the central repository for advanced generative modeling concepts. I will try to explain the working of VAE the way I understand. Your suggestions are always welcome.

---

## 📑 Table of Contents
* [1. Understanding Pytorch Model](#1-understanding-pytorch-model)
* [2. Classifier-Free Guidance Mechanics](#2-classifier-free-guidance-mechanics)
* [3. Visualizing Network Architecture](#3-visualizing-network-architecture)
* [4. Key Takeaways](#4-key-takeaways)

---

## 1. Understanding Pytorch Model   

Link: https://colab.research.google.com/drive/1iWtifdm-9QSWJ3vtOHdnJJrp7My7tamV#scrollTo=0UxwsoidKA0e

  - Encoder has 28 x 28 = 784 pixels images from MINST dataset as input
  - only one hidden layer has 400 hidden dimensions.
  - Output layer has 20 dimensions. Below is how model looks like. we will have vector of length 20 for $\mu$ and $\sigma$, i.e., variance in log scale. it will give you
    $q(z|x) = N(\mu, diag(\sigma^2))$
  - two variables $\mu$ and $\sigma^2$ will be combined using reparameterization trick.  $z= \mu + \sigma^2 \circledcirc \epsilon$, where $\epsilon$ is the noise sampled separately from $N(0,I)$ 
  -  **h = F.relu(self.fc1(x))**: Computes the hidden layer representations. It passes the input tensor x (784 features) through the first linear layer to get 400 features, then applies a Rectified Linear Unit (ReLU) activation function. This introduces non-linearity, allowing the model to learn complex patterns instead of just simple linear combinations.
  -  **mu = self.fc_mu(h)**: Calculates the latent mean. It projects the 400-dimensional hidden features h into a 20-dimensional mean vector (mu). This vector represents the center point of the cluster in the latent space where the input image belongs.
  - **logvar = self.fc_logvar(h)**: Calculates the latent log-variance. Running in parallel to the mean layer, it projects the same 400-dimensional features h into a separate 20-dimensional log-variance vector (logvar). This vector represents the spread, uncertainty, or noise level for each latent dimension.
```python
VAE(
  (fc1): Linear(in_features=784, out_features=400, bias=True)
  (fc_mu): Linear(in_features=400, out_features=20, bias=True)
  (fc_logvar): Linear(in_features=400, out_features=20, bias=True)
  (fc2): Linear(in_features=20, out_features=400, bias=True)
  (fc3): Linear(in_features=400, out_features=784, bias=True)
)
#And the forward function is 
  h = F.relu(self.fc1(x))
  mu = self.fc_mu(h)
  logvar = self.fc_logvar(h)
  return mu, logvar

```
**Loss Function**  
  - we are going to use BinaryCrossEntropy(BCE) loss, where every pixel of original and reconstructed image will be compared and if there is a huge difference between in one to one comparison then penalty is huge.
  - In a VAE, the total loss function (called the ELBO) is a balance between two parts:
    1. Reconstruction Loss: How accurate the image is.
    2. KL Divergence Loss: How well the latent space fits a standard normal distribution.

  - The mathematical derivation of the VAE optimization objective naturally dictates that both losses should be computed as a total sum over the dimensions. If you were to     use reduction="mean", the reconstruction loss would get divided by the number of pixels (784) and the batch size, making it artificially tiny compared to the KL       divergence loss. This imbalance would cause the model to ignore image quality entirely.
### PyTorch Flattening Example
The following code snippet demonstrates how to flatten all dimensions of a tensor except for the batch size:

```python
import torch

# Representing a batch of 32 images (3 channels, 64x64 pixels)
x = torch.randn(32, 3, 64, 64)

# Flatten everything except the batch dimension
# Input shape: [32, 3, 64, 64] -> Output shape: [32, 12288]
flattened_x = x.view(x.size(0), -1)

print(f"Original shape: {x.shape}")
print(f"Flattened shape: {flattened_x.shape}")
```

---

## 2. Classifier-Free Guidance Mechanics

In generative networks, **Classifier-Free Guidance (CFG)** balances sample quality and diversity without relying on a separate classifier network. 

> 💡 **Key Insight:** Increasing the guidance scale ($w$) shifts the generated samples closer to the conditioning prompt, sharpening the details but reducing overall structural variety.

* **Unconditioned Score:** $\epsilon_\theta(x_t, \emptyset)$
* **Conditioned Score:** $\epsilon_\theta(x_t, c)$
* **Guided Output:** $\tilde{\epsilon}_\theta(x_t, c) = (1 + w)\epsilon_\theta(x_t, c) - w\epsilon_\theta(x_t, \emptyset)$

---

## 3. Visualizing Network Architecture

Visualizing the vector fields or flow map trajectories helps intuition when working with continuous-time generative models like Flow Matching.

![Flow Matching Vector Field Trajectory](images/vector-field-flow.png)
*Figure 1: Implicit representation of a vector field acting as a coordinate-based trajectory planner.*

---

## 4. Key Takeaways

* **Dimensionality Alignment:** Always ensure tracking components match your internal tensor dimensions exactly.
* **Vector Field Trajectories:** Diffeomorphisms ensure smooth, invertible transformations between noise and data distributions.
* **Batch Verification:** Use standard loop utilities to extract and inspect initial batches before launching long training jobs.

