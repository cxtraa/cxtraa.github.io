---
title: Using einops
layout: post
parent: Machine learning
nav_order: 1
---

# Einops is all you need
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

[Einops](https://einops.rocks/) is a powerful, general purpose Python library designed to allow complex tensor operations to be done with minimal, fully readable code. For example, the following operation transposes a `2*2` matrix:

``` python
einops.rearrange(my_2d_matrix, "i j -> j i")
```

We will explore the main methods of the einops library, namely `.rearrange()`, `.reduce()`, and `.einsum()`. These 3 functions have simple syntax and allow you to perform any tensor multipliation operation you might encounter in an ML setting.

# einops.rearrange()

`einops.rearrange()` allows you to reshape a tensor easily. Let's look at a few examples which illustrate this:

### Transposing a 2D matrix

``` python
einops.rearrange(matrix, "i j -> j i")
```

This is a simple example where we simply reverse the `i` and `j` dimensions of the matrix, producing its transpose.

### Joining tensors together

``` python
einops.rearrange(tensors, "b c h w -> c (b h) w")
```

This joins my tensors together vertically. Effectively `b h` is converted to a single dimension `(b h)`.

### Flatten tensor

```python
einops.rearrange(tensors, "b c h w -> (b c h w)")
```

This example is fairly trivial given the one before it.

### Decomposition of tensors

``` python
einops.rearrange(tensor, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2)
```

Note that we gave a 'hint' to the compiler that the batch dimension was going to be split into `b1` and `b2`. We must specify the value of one of these because there are multiple decompositions possible of the batch dimension. 

### Stretching

``` python
einops.rearrange(tensor, "b c h w -> b c (h 2)")
```

This vertically stretches our tensor by a factor of two.

### Tiling

``` python
einops.rearrange(tensor, "b c h w -> b c (2 h)")
```

Notice that when the number prefaces the variable in `2 h`, this indicates we are tiling the tensor (copying it) rather than stretching it.

# einops.reduce()

The `.reduce()` method allows us to remove redundant dimensions from our tensors. Some examples are shown below.

### Converting image to greyscale

``` python
einops.reduce(im, "c h w -> h w")
```

This removes the channel data of the image which contains its RGB data. Hence we get a greyscale image.

# einops.einsum()

`.einsum()` is certaintly the most important method discussed so far. It allows us to compute things like inner/outer products and multiply tensors together. Here are the basic rules:
- If a dimension is included in the input tensors but not the output, then this implies summation over that dimension.
- If a dimension appears in the input and output tensors, then we multiply this dimension as normal.

### Finding the dot product of two vectors

To dot product two vectors `vec1` and `vec2`:
``` python
einops.einsum(vec1, vec2, "i i -> ")
```

We specify the input tensors `vec1` and `vec2`. Now, when computing the dot product, the position along the vector `i` is the same for both vectors, as we move `i` along both at the same rate. Now, we would like to *sum over i*, so we exclude `i` in the output vector.

### Multiplying a matrix with a vector

To multiply a matrix `mat` with a vector `vec`:

``` python
einops.einsum(mat, vec, "i j, j -> i")
```

The rationale should hopefully be clear here - the `j` dimension is collapsed leaving only `i`, implying summation over `j`. 

### Multiplying two matrices

To multiply two matrices `mat1` and `mat2`:

``` python
einops.einsum(mat1, mat2, "i j, j k -> i k")
```

We collapse along `j`, and sum over `j`, leaving us with a matrix of shape `(i, k)`.

### Calculating the outer product of two vectors

To find the outer product of two vectors `vec1`, `vec2`:

``` python
einops.einsum(vec1, vec2, "i, j -> i j")
```

The outer product produces a matrix of all possible products of scalar values in `vec1` and `vec2`. It involves no summation, so no variables are omitted in the output.

# Conclusion

Here, we discussed some basic usages of the `einops` library. In practice, `einops` is extensively used within machine learning. I use `einops` when working with PyTorch on my research, and it is significantly less error-prone than relying on `NumPy` methods such as `.squeeze()`. The syntax provided by `einops` is consistent, readable, and flexible.
