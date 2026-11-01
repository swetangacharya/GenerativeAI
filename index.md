---
layout: default
title: "Deep Learning & Generative AI Hub"
description: "A comprehensive guide to neural network architectures, flow matching, and diffusion models."
---

# A Generative AI.

Welcome to the central repository for advanced generative modeling concepts. I will try to explain the working of VAE the way I understand. Your suggestions are always welcome.

---

## 📑 Table of Contents
* [1. Core Tensor Operations](#1-core-tensor-operations)
* [2. Classifier-Free Guidance Mechanics](#2-classifier-free-guidance-mechanics)
* [3. Visualizing Network Architecture](#3-visualizing-network-architecture)
* [4. Key Takeaways](#4-key-takeaways)

---

## 1. Core Tensor Operations

When preparing data for deep learning models, reshaping tensors correctly is critical. For instance, flattening feature maps before passing them into linear layers is a routine step.

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

