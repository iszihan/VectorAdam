# VectorAdam for Rotation Equivariant Geometry Optimization

This repository is the official PyTorch implementation of 

VectorAdam for Rotation Equivariant Geometry Optimization  
[Selena Ling](https://www.iszihan.github.io), [Nicholas Sharp](https://nmwsharp.com/), [Alec Jacobson](https://www.cs.toronto.edu/~jacobson/) 

NeurIPS 2022

## Requirements

To use our VectorAdam implementation, you just need to have PyTorch installed in your environment.

The demo script are tested with PyTorch=1.11 and matplotlib=3.5.1. We also provide the environment file `vectoradam.yml`, which can be used to create a conda environment as in 
```
conda env create -f vectoradam.yml -n [env-name]
```
Note that this is tested on Ubuntu 18.04 only.

## Usage

To use VectorAdam in your project, 

```
optimizer = VectorAdam(
    [{'params': X, 'axis': -1}, 
     {'params': Y, 'axis':  1], 
     lr=lr, betas=betas, eps=eps))
```

The above example will apply VectorAdam's vector-wise operations to X along the last axis and Y along the 1st axis, with specified learning rate, betas and epsilon hyperparameters.

## Demo
We provide a demo with `laplacian2d_demo.ipynb` that reproduces the 2D results we have in Figure 4.