---
title: "Module 5 -- Optimization"
classes: wide
---
# What is Optimization?
* Optimization is a mathematical framework for finding the best solution for maximization or minimization problems. For example, we have learned from the statistics section that we need to find the best parameter \\(\pmb{\theta}\\) for a distribution that can explain the observed data with the highest probability. In this case, the maximum likelihood estimation boils down to an optimization problem in which our goal is to minimize the negative likelihood (NLL), known as _loss (cost) function_ denoted by \\(\mathcal{L}(\pmb{\theta})\\). Hence, we can write our optimization problem as:
  \\[\pmb{\theta}^* = \text{argmin} _{\theta\in \mathbb{\Theta}}~\mathcal{L}(\pmb{\theta})\\]
  - Here, we assume that the parameter space is a continuous \\(p\\)-dimensional space such that \\(\pmb{\theta}\in \mathbb{\Theta}\\); hence, we deal with continuous optimization. Later in this course, we talk about discrete optimization problems.
  - A point that satisfies the above equation is called a **_global minimum_**, which is typically computationally hard to compute.
  - However, in most cases, we are looking for **_local optimum(s)_** which are the points with smaller (larger) or equal loss function than _nearby_ points for minimization (maximization) problems. This can be expressed as following: \\(\pmb{\theta^*}\\) is called a local minimum if
 
    \\[\exists \delta > 0, ~ \forall \theta ~~ s.t. \|\|\pmb{\theta}-\pmb{\theta}^*\|\|, ~ \mathcal{L}(\pmb{\theta^*}) \leq \mathcal{L}(\pmb{\theta}) \\]
  - It is possible (in fact this is very common in ML) to have more than one local minimum  with the same objective value; this
is known as a flat local minimum.
  - For a continuous and differentiable function, a **_stationary point_** is a point for which the gradient is zero, i.e., \\(\pmb{\theta} ^*\\) is stationary point iff \\(\nabla_{\pmb{\theta}} \mathcal{L}(\mathbb{\theta}) = 0\\) for \\(\pmb{\theta} = \pmb{\theta^*}\\)
  - The following figure shows the global and local minimums and maximums.

    <p align="center">
            <img width="600" alt="Screenshot 2023-07-10 at 7 21 57 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/4149331b-b489-4384-a027-e3e720d13a26">
    <br>
            <em>Global and local minimums and maximums of a continuous and differentiable function.</em>
     </p>
  
* In machine learning fitting a model to the given data samples is a core problem which essentially means solving an optimization problem. An ML model is optimized to produce the best prediction (smallest error). For example (as we will see later), in the linear regression problems, we try to fit a linear model by optimizing its parameters such that the output model can be as close as possible to the observed samples.
* 
