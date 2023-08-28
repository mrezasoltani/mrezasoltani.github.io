---
title: "Module 5 -- Optimization"
classes: wide
---
## What is Optimization?
* Optimization is a mathematical framework for finding the best solution for maximization or minimization problems. For example, we have learned from the statistics section that we need to find the best parameter \\(\pmb{\theta}\\) for a distribution that can explain the observed data with the highest probability. In this case, the maximum likelihood estimation boils down to an optimization problem in which our goal is to minimize the negative likelihood (NLL), known as _loss (cost) function or objective function_ denoted by \\(\mathcal{L}(\pmb{\theta})\\). Hence, we can write our optimization problem as:
  \\[\pmb{\theta}^* = \text{argmin} _{\pmb{\theta}\in \mathcal{\Theta}}~\mathcal{L}(\pmb{\theta})\\]
  - Here, we assume that the parameter space is a continuous \\(p\\)-dimensional space such that \\(\pmb{\theta}\in \mathcal{\Theta}\\); hence, we deal with continuous optimization. Later in this course, we talk about discrete optimization problems.
  -When \\(\mathcal{\Theta}\\) is the domain of the objective function, then the optimization problem is called **_unconstrained optimization_**. For example,  \\(\mathcal{\Theta} = \mathbb{R}^p\\). However, in ML optimization problems are **_constrained_**. It is common to show the set of constraints as a set of \\(m\\) equalities, e.g., \\(h_k(\pmb{\theta}) = 0\\) for \\(k=1,2,\ldots, m\\), and a set of \\(n\\) inequalities, e.g.,  \\(g_k(\pmb{\theta}) \leq 0\\) for \\(k=1,2,\ldots, n\\). Hence, for a constrained optimization problem, we write the problem as:
  \\[\pmb{\theta}^* = \text{argmin} _{\pmb{\theta}\in \mathcal{C}}~\mathcal{L}(\pmb{\theta})\\]
  - **Feasible Set (Solution).** The feasible set is a subset of the parameter space that satisfies all constraints:
    \\[\mathcal{S} = \{\pmb{\theta}): h_k(\pmb{\theta}) = 0, ~ k=1,2,\ldots, m, g_k(\pmb{\theta}) \leq 0, ~ k=1,2,\ldots, n\}\\]
  - A point that satisfies the above equation is called a **_global minimum_**, which is typically computationally hard to compute.
  - However, in most cases, we are looking for **_local optimum(s)_** which are the points with smaller (larger) or equal loss function than _nearby_ points for minimization (maximization) problems. This can be expressed as following: \\(\pmb{\theta^*}\\) is called a local minimum if
 
    \\[\exists \delta > 0, ~ \forall \theta ~~ s.t. \|\|\pmb{\theta}-\pmb{\theta}^*\|\|, ~ \mathcal{L}(\pmb{\theta} ^{\*}) \leq \mathcal{L}(\pmb{\theta}) \\]
  - Similaer definition holds for the local maximum.
  - 
  - It is possible (in fact this is very common in ML) to have more than one local minimum  with the same objective value; this is known as a flat local minimum.

  - For a continuous and differentiable function, a **_stationary point_** is a point for which the gradient is zero, i.e., \\(\pmb{\theta^*}\\) is stationary point iff \\(\nabla_{\pmb{\theta}} \mathcal{L}(\pmb{\theta}) = 0\\) for \\(\pmb{\theta}=\\) \\(\pmb{\theta}^{\*}\\).
  - The following figure shows the global and local minimums and maximums:

    <p align="center">
            <img width="600" alt="Screenshot 2023-07-10 at 7 21 57 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/88e20434-99a4-4e5c-8ec7-2dbf12413c2c">
    <br>
            <em>Global and local minimums and maximums of a continuous and differentiable function.</em>
     </p>
  - The following conditions provide the necessary and sufficient conditions for a local minimum assuming the function is a continuous and twice differentiable:
    - **Necessary Condition.** \\(\pmb{\theta}^{\*}\\) is a local minimum if the gradient evaluated at \\(\pmb{\theta}^{\*}\\), i.e.,  \\(\nabla\mathcal{L}(\pmb{\theta}^{\*}) = 0\\) (a stationary point), and the Hessian at \\(\pmb{\theta}^{\*}\\) is a PSD matrix, i.e., \\(\nabla^2 \mathcal{L}(\pmb{\theta}^{\*}) \succeq 0\\).
    - **Sufficient condition.** If \\(\nabla\mathcal{L}(\pmb{\theta}^{\*}) = 0\\) and  \\(\nabla^2 \mathcal{L}(\pmb{\theta}^{\*}) \succ 0\\), i.e., the Hessian is a positive definite (PD) matrix, then \\(\pmb{\theta}^{\*}\\) is a local minimum.
  - **Saddle Point.** A saddle point is a stationary point (i.e., gradient is zero) such that the Hessian at this point has both negative and positive eigenvalues. This means that a saddle point could be a local minimum in some direction and a local maximum in other directions.
* A main problem in machine learning is to fit a model to given data samples, which essentially means solving an optimization problem. An ML model is optimized to produce the best prediction (smallest error). For example (as we will see later), in the linear regression problems, we try to fit a linear model by optimizing its parameters such that the output model can be as close as possible to the observed samples.
* 
