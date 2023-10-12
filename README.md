# Denoising Variational Lossy Autoencoder
<!---
Code to accompany [Unsupervised Structured Noise Removal with Variational Lossy Autoencoder]() TODO: update with Arxiv link<br>
-->
<sup>1</sup>Benjamin Salmon and <sup>2</sup>Alexander Krull<br>
<sup>1, 2</sup>University of Birmingham<br>
<sup>1</sup>brs209@student.bham.ac.uk, <sup>2</sup>a.f.f.krull@bham.ac.uk<br>
This project includes code from the [ladder-vae-pytorch](https://github.com/addtt/ladder-vae-pytorch) project, which is licensed under the MIT License.


<img src="https://github.com/krulllab/DVLAE/blob/main/resources/matrix.png" width=50% height=50%>

Most unsupervised denoising methods are based on the assumption that imaging noise is either pixel-independent, \ie, spatially uncorrelated, or signal-independent, \ie, purely additive.
However, in practice many imaging setups, especially in microscopy, suffer from a combination of signal-dependent noise (\eg Poisson shot noise) and axis-aligned correlated noise (\eg stripe shaped scanning or readout artifacts).
In this paper, we present the first unsupervised deep learning-based denoiser that can remove this type of noise without access to any clean images or a noise model.
Unlike self-supervised techniques, our method does not rely on removing pixels by masking or subsampling so can utilize using all available information.
We implement a Variational Autoencoder (VAE) with a specially designed autoregressive decoder capable of modelling the noise component of an image 
but incapable of independently modelling the underlying clean signal component.
As a consequence, our VAE's encoder learns to encode only underlying clean signal content and to discard imaging noise.
We also propose an additional decoder for mapping the encoder's latent variables back into image space, thereby sampling denoised images.
Experimental results demonstrate that our approach surpasses existing methods for self- and unsupervised image denoising while being robust with respect to the size of the autoregressive receptive field.

<!---
### BibTeX
```

``` TODO: update with Arxiv citation<br>
-->


### Dependencies
We recommend installing the dependencies in a conda environment. If you haven't already, install miniconda on your system by following this [link](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).<br>
Once conda is installed, create and activate an environment by entering these lines into a command line interface:<br>
1. `conda create --name dvlae`
2. `conda activate dvlae`


Next, install PyTorch and torchvision for your system by following this [link](https://pytorch.org/get-started/locally/).<br> 
After that, you're ready to install the dependencies for this repository:<br>
`pip install lightning jupyterlab matplotlib tifffile scikit-image tensorboard`

### Example notebooks
This repository contains 3 notebooks that will first download then denoise the [Structured Convallaria dataset](https://ieeexplore.ieee.org/abstract/document/9098336?casa_token=ROPuswhAvi0AAAAA:BYQUOnGY51SEqy3CAe7ZTzoOpjjfq8oWrwcJF6KfF4KzIlrjpCL0mR7H7TjDV802pTiJfe0ufg). 1-training trains the denoising network, 2-prediction uses the trained network to denoise an inference set and 3-evaluation calculates the Peak Signal-to-Noise Ratio (PSNR) of the denoised results.
