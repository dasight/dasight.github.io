---
layout: post
title:  "Distributed Model Training with Tensor Parallelism over Ray"
date:   2026-01-29 22:58:09 +0800
categories: jekyll update
usemathjax: true
---

# Introduction

As large language models and other deep neural networks continue to scale, a single GPU or GPUS of a single machine often can’t provide enough capacity to train efficiently. While data parallelism is a common first step, it eventually hits limits: each GPU still needs to fit the full model. To push beyond these constraints, practitioners increasingly rely on model parallelism, where the model itself is partitioned across multiple GPUs.

This blogpost focuses on tensor parallelism, a practical and widely used form of model parallel training that splits individual weight tensors (and their associated computations) across devices. In tensor parallelism, construction blocks such as linear projections or attention blocks are sharded so that each GPU holds only a slice of the parameters, performs a slice of the computation, and collaborates via collective communication (e.g., all-reduce / all-gather) to produce correct forward activations and backward gradients. This approach can unlock training for models that would otherwise exceed per-GPU memory limits, and it can improve throughput when carefully implemented.

At the same time, implementing tensor parallel training in real systems is not only about PyTorch primitives—it also requires robust orchestration: launching processes, assigning GPUs, wiring distributed process groups, handling failures, and integrating with clusters that may be shared or elastic. Ray provides a flexible runtime for distributed execution that can simplify these operational concerns. By treating training workers as schedulable actors and tasks, Ray makes it easier to scale from a single node to multi-node clusters, manage resources, and build custom training workflows—especially when you can’t rely on higher-level wrappers and need direct control over the training loop.

In the sections that follow, we’ll take [Llama 3.x](https://ai.meta.com/blog/meta-llama-3/) as the example to explain the core mechanics of tensor parallelism, including common sharding patterns like row/column-parallel linear layers, how communication is introduced in forward and backward passes, and how to structure a Ray-based training architecture that launches and coordinates tensor-parallel workers. We’ll also present the real code for implementing these constructs. The source code used in this blogpost can be found at my [Github repository](https://github.com/dasight/tensor-parallel-with-ray).

# Inside Tensor Parallelism

Tensor parallelism (TP) is a model-parallel technique that splits individual tensors inside a layer—most commonly the large weight matrices in attention projections and MLP blocks—across multiple GPUs, so no single device needs to store or compute the full layer. Instead of replicating the whole model on every worker (as in data parallelism), each GPU owns a shard of the parameters, runs the corresponding shard of the forward and backward computations, and then uses fast collective communication (e.g., all-reduce, all-gather, reduce-scatter) to exchange the minimal information needed to produce correct activations and gradients. In practice, TP turns a single “too-big” matrix multiply into several smaller ones that run in parallel, trading extra communication for the ability to scale model size and throughput beyond a single GPU’s limits.

## Distributed Matrix Multiplication over Multiple GPUs

The core process of Tensor parallelism is to split the weight matrix, forward or backward separately, and merge the results. The figure below show the procedure of distributed matrix multiplication, which splits a weight matrix `A` of a model vertically into three parts, multiplies with the full input matrix `X` separately, and gets the resulting matrix shards in three different devices.

![Split Matrix by Column-Wise](/assets/2026-01-29-tp-col-par.png)

In the TP process above, different columns can be computed in the difference devices in parallel. For this reason, it is generally named column parallel or column-wise parallel. And in the output of the column parallel, the value of each shard element is already the final value of the resulting matrix element, and we only require to concatenate the resulting shards to make the resulting matrix.

Another type of Tensor parallelism is to split the weight matrix horizontally, as shown below. By this means, the weight matrix `A` is splitted horizontally, while the input matrix `X` is also splitted. However, unlike the column-wise parallel, the elements of the resulting shards from the row-wise parallel are not the final values of the resulting matrix. They have to be summed up together to make the final result.

![Split Matrix by Row-Wise](/assets/2026-01-29-tp-row-par.png)

### Why both row parallel and column parallel are required in Tensor parallelism?

A common question from the new learners of distributed model training are why both row-wise and column-wise parallels need to be addressed. The answer is for the training performance.

Suppose we only have column-wise parallel, and require to implement two continous matrix multiplications with an activation function sitting between them (It is a very common operation in the large language model). This process can be shown as a simple pipeline like below:

$$ X_1 \xrightarrow{\;W_1\;} X_2 \xrightarrow{\;Activate\;} X_3 \xrightarrow{\;W_2\;} X_4 $$

Since the shards of the resulting matrixes $X_2$ and $X_3$ (the activation function is an element-wise operation and won't change the distribution layouts) are also distributed in different GPUs, we need to execute an all-gether operation against $X_3$ to collect all of the resulting shards, since the input of the next column-wise parallel operation $X_3 \cdot W_2$ requires the full matrix of $X_3$.

However, if we use row-wise parallel for $X_3 \cdot W_2$, we don't have to execute any collective communication, as it happens to require the input matrix to be vertically sharded.

## MLP Layer in Tensor Parallelism

(TBC)

## Attention Layer in Tensor Parallelism

(TBC)

# Combine Tensor and Pipeline Parallelism Together

(TBC)
