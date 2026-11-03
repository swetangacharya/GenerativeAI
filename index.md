---
layout: default
title: "Deep Learning & Generative AI Hub"
description: "A comprehensive guide to Pytorch neural network architectures, VAE, flow matching, and diffusion models."
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
    1. Reconstruction Loss: How accurate the image is. **reconstruction_loss = F.binary_cross_entropy(recon_x,x, reduction="sum")**
    2. KL Divergence Loss: How well the latent space fits a standard normal distribution. **kl_loss = -0.5 * torch.sum( 1 + logvar - mu.pow(2) - logvar.exp())**


  - The mathematical derivation of the VAE optimization objective naturally dictates that both losses should be computed as a total sum over the dimensions. If you were to     use reduction="mean", the reconstruction loss would get divided by the number of pixels (784) and the batch size, making it artificially tiny compared to the KL       divergence loss. This imbalance would cause the model to ignore image quality entirely.
  - F.binary_cross_entropy(input, target): Measures the difference between target binary labels ($y \in {0,1}$)  and predicted probabilities ($p \in [0,1]$)
    
   $\color{blue}l(x,y)= -[y \cdot logx + (1-y) \cdot log(1-x)]$   
  -
  - Here, y is ground truth and x is predict input, now if we have ground truth=1 and predicted truth is 0 then   
    $loss(0,1)= -[1 \cdot log(0)+ 0 \cdot log(1-0)]= \infty$ , because $log(0)=\infty$, model punish heavily for predicting completely wrong.   

  - Now if we have x=0.2 and y=0
    $loss(0.2,0)= -[0 \cdot log(0.2)+ 1 \cdot log(1-0.2)]= 0.223$ , model won't punish heavily for predicting slightly wrong.


- we can also use binary_cross_entropy_with_logits(input,target):
F.binary_cross_entropy_with_logits(input, target): Combines a Sigmoid layer and BCE loss into a single function for better numerical stability. It takes raw model logits rather than probabilities.

### Pytorch VAE Model Example
Think of your model as a production line. The layers you define in __init__ are the machines, and forward() is the conveyor belt that moves the data through those machines in a specific order.  

```python
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        ### Encoder part
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        ### Decoder part
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        """
        Encoder network. Given x, produce: mu, logvar = log(sigma^2)
        These define: q(z|x) = N(mu, diag(sigma^2))
        """
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        # Reparameterized sample
        z = mu + std * eps
        return z
    def decode(self, z):
        h = F.relu(self.fc2(z))
        x_recon = torch.sigmoid(self.fc3(h))
        return x_recon

    def forward(self, x):
        x= torch.flatten(x, start_dim=1)
        # Encoder
        mu, logvar = self.encode(x)
        # Sample z
        z = self.reparameterize(mu, logvar)
        # Decoder
        x_recon = self.decode(z)
        return x_recon, mu, logvar


```

---
reconstruction loss: BCE(x_pred,x_true), it will ask encoder to brigde this gap.
so total loss is push-pull kind of scenario. where reconstruction loss force encoder to do better and bring image as close as real. while KL loss saves it from overfitting.

so, our ELBO is $\ell (\theta,\phi;x)=E_{z \sim q_{\phi(\cdot|x)}}[log p_{\theta}(x|z)] - D_{KL}(q_{\phi}(z|x)||p(z))$
$D_{KL}(q_{\phi}(z|x)||p(z)) = - \frac{1}{2} \sum_{i=1}^d(1+log(\sigma_i^2)-\mu_i^2- \sigma_i^2)$

**Breaking down each term:**   
$\mu^2$ **(Mean Penalty)**
	- Encourages the mean μ of the latent distribution to stay close to 0.
    - If the encoder pushes the mean far away from zero (e.g., $\mu= 5$), this term grows large, penalizing the network.

$\sigma^2$ **(Variance Penalty)**   
- Encourages the variance $\sigma^2$ to be close to 1.If $\sigma^2$ gets very large, this linear term grows large and penalizes the network.
  
$- log(\sigma^2)$ **(entropy/information term)**
- Prevents the variance from collapsing to zero ($\sigma^2 \to 0$, which would make the encoder deterministic like a standard autoencoder

```python
def vae_loss(recon_x, x, mu, logvar):
    # Flatten original x to match recon_x.
    x = x.view(x.size(0), -1)
    reconstruction_loss = F.binary_cross_entropy(recon_x,x,reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Total negative ELBO
    total_loss = reconstruction_loss + kl_loss
    return total_loss, reconstruction_loss, kl_loss
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    # No gradients are required during evaluation.
    with torch.no_grad():

        for images, _ in loader:
            images = images.to(device)
            recon_images, mu, logvar = model(images)
            loss,recon_loss, kl_loss = vae_loss(recon_images,images,mu,logvar)
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

    N = len(loader.dataset)
    return (total_loss / N,total_recon / N,total_kl / N)


```

```python
train_loss, train_recon, train_kl = train_one_epoch(
        model,
        train_loader,
        optimizer,
        device
    )

    test_loss, test_recon, test_kl = evaluate(
        model,
        test_loader,
        device
    )
```
## now for every epoch we make a forward pass the images, get the VAE loss , evaluate and calculate the average loss
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

