---
title: "Module 5 -- Optimization"
classes: wide
---
## What is Optimization?
* Optimization is a mathematical framework for finding the best solution for maximization or minimization problems. For example, we have learned from the statistics section that we need to find the best parameter \\(\pmb{\theta}\\) for a distribution that can explain the observed data with the highest probability. In this case, the maximum likelihood estimation boils down to an optimization problem in which our goal is to minimize the negative likelihood (NLL), known as _loss (cost) function or objective function_ denoted by \\(\mathcal{L}(\pmb{\theta})\\). Hence, we can write our optimization problem as:
  \\[\pmb{\theta}^* = \text{argmin} _{\pmb{\theta}\in \mathcal{\Theta}}~\mathcal{L}(\pmb{\theta})\\]
  - Here, we assume that the parameter space is a continuous \\(p\\)-dimensional space such that \\(\pmb{\theta}\in \mathcal{\Theta}\\); hence, we deal with continuous optimization. Later in this course, we talk about discrete optimization problems.
  - The variables we optimize over is called the **_decision variables_** or _**optimization variables**_. In pur notation, \\(\pmb{\theta} = \[\theta_1, \theta_2, \ldots, \theta_p\]\\) is the decision variables.
  - When \\(\mathcal{\Theta}\\) is the domain of the objective function, then the optimization problem is called **_unconstrained optimization_**. For example,  \\(\mathcal{\Theta} = \mathbb{R}^p\\). 
  - A point that satisfies the above equation is called a **_global minimum_**, which is typically computationally hard to compute.
  - However, in most cases, we are looking for **_local optimum(s)_** which are the points with smaller (larger) or equal loss function than _nearby_ points for minimization (maximization) problems. This can be expressed as following: \\(\pmb{\theta^*}\\) is called a local minimum if
 
    \\[\exists \delta > 0, ~ \forall \theta ~~ s.t. \|\|\pmb{\theta}-\pmb{\theta}^*\|\|, ~ \mathcal{L}(\pmb{\theta} ^{\*}) \leq \mathcal{L}(\pmb{\theta}) \\]
  - A similar definition holds for the local maximum.
  - It is possible (in fact this is very common in ML) to have more than one local minimum  with the same objective value; this is known as a flat local minimum.
  - For a continuous and differentiable function, a **_stationary point_** is a point for which the gradient is zero, i.e., \\(\pmb{\theta^*}\\) is stationary point iff \\(\nabla_{\pmb{\theta}} \mathcal{L}(\pmb{\theta}) = 0\\) for \\(\pmb{\theta}=\\) \\(\pmb{\theta}^{\*}\\).
  - The following figure shows the global and local minimums and maximums:

    <p align="center">
            <img width="600" alt="Screenshot 2023-07-10 at 7 21 57 PM" 
              src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/05c5bc8c-c75b-4338-9669-c94215fb5f38">
        <br>
            <em>Global and local minimums and maximums of a continuous and differentiable function.</em>
     </p>

  - The following conditions provide the necessary and sufficient conditions for a local minimum assuming the function is continuous and twice differentiable:
    - **Necessary Condition.** If \\(\pmb{\theta}^{\*}\\) is a local minimum, then the gradient evaluated at \\(\pmb{\theta}^{\*}\\) should be zero, i.e.,  \\(\nabla\mathcal{L}(\pmb{\theta}^{\*}) = 0\\) (a stationary point). Furthermore, the Hessian at \\(\pmb{\theta}^{\*}\\) should be a PSD matrix, i.e., \\(\nabla^2 \mathcal{L}(\pmb{\theta}^{\*}) \succeq 0\\).
    - **Sufficient condition.** If \\(\nabla\mathcal{L}(\pmb{\theta}^{\*}) = 0\\) and  \\(\nabla^2 \mathcal{L}(\pmb{\theta}^{\*}) \succ 0\\), i.e., the Hessian is a positive definite (PD) matrix, then \\(\pmb{\theta}^{\*}\\) is a local minimum.
  - **Saddle Point.** A saddle point is a stationary point (i.e., the gradient is zero) such that the Hessian at this point has both negative and positive eigenvalues. This means that a saddle point could be a local minimum in some directions and a local maximum in other directions.
  - However, optimization problems in ML are **_constrained_**. For these problems, our goal is to find the best solution (minimum or maximum) in a subset of the domain of the objective function. It is common to show the set of constraints as a set of \\(m\\) equalities, e.g., \\(h_k(\pmb{\theta}) = 0\\) for \\(k=1,2,\ldots, m\\), and a set of \\(n\\) inequalities, e.g.,  \\(g_k(\pmb{\theta}) \leq 0\\) for \\(k=1,2,\ldots, n\\). Hence, for a constrained optimization problem, we write the problem as:
  \\[\pmb{\theta}^* = \text{argmin} _{\pmb{\theta}\in \mathcal{S}}~\mathcal{L}(\pmb{\theta})\\]
  - **Feasible Set (Solution).** The feasible set is a subset of the parameter space that satisfies all constraints:
    \\[\mathcal{S} = \\{\pmb{\theta}: h_k(\pmb{\theta}) = 0, ~ k=1,2,\ldots, m, ~~ g_k(\pmb{\theta}) \leq 0, ~ k=1,2,\ldots, n\\}\\]
  - Adding constraints to an optimization problem may result in finding any point (regardless of its cost) in the feasible set challenging on its own. This sub-problem is called _feasibity problem_.
  - A typical approach when there are constraints on the optimization variables (constraint optimization problems) is to add the constraints to the objective function as the penalty terms by some multiplier. These added terms measure how much we violate each constraint. This method is called _Lagrangian multipliers_.
  
## Convex vs. Non-convex Optimization
* Optimization problems can be classified into different categories in terms of their complexity. One broad criterion is to distinguish optimization problems if they are convex or not.
  - **Convex Optimization.** Before defining what a convex optimization problem is we need to understand the convex sets and convex functions.
    - **Convex Set.** a Set \\(\mathcal{S}\\) is convext set if for any \\(\pmb{\theta}_1, ~ \pmb{\theta}_2 \in \mathcal{S}\\) and for any \\(\lambda\in \[0,1\]\\), we have \\(\lambda\pmb{\theta}_1 + (1-\lambda)\pmb{\theta}_2 \in \mathcal{S}\\).
    - This means that any point connecting two points in the set is also in the set. The following figure shows convex and non-convex sets.

    <p align="center">
          <img width="600" alt="Screenshot 2023-07-10 at 7 21 57 PM"   
            src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/d9d60965-0448-427c-936f-14e5a53fb515">
      <br>
            <em>Convex and Non-convex sets.</em>
     </p>

    - **Convex Functions.** A function \\(f(\pmb{\theta})\\) is called a convex function if its domain is a convex set and for any two points \\(\pmb{\theta}_1, \pmb{\theta}_2 \\) in its domain, we have:
    \\[f(\lambda\pmb{\theta}_1 + (1-\lambda)\pmb{\theta}_2) \leq \lambda f(\pmb{\theta}_1) + (1-\lambda)f(\pmb{\theta}_2)\\]
      - If the above inequality is a strict inequality, the function \\(f\\) is called strictly convex.
      - Equivalently, a function is convex if the set of all points above the graph of the function called _epigraph_ denoted by \\(\text{epi}(f)\\) is a convex set. That is, the following set is a convex set.
        \\[\text{epi}(f) = \\{(\pmb{\theta}, t):  f(\pmb{\theta}) \leq t\\}\\]
      - The following figure shows the epigraph of a function, a convex function, and a non-convex function.
      
        <p align="center">
            <img width="600" alt="Screenshot 2023-07-10 at 7 21 57 PM"       
              src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/f4f06eb6-944d-40d8-9991-8de17d1bd139">
         <br>
            <em>Epigraph of a function, a convex function, and a non-convex function.</em>
        </p>
     
      - If \\(f(\pmb{\theta})\\) is a convex function, \\(-f(\pmb{\theta})\\) is a **_concave_** function.
      - Convex functions are bowl-shaped functions.
      - Some common examples of 1-D convex functions are as follows:
          \\[x^2, e^x, -\log(x), x\log(x)\\]
      - One important property of the convex functions is that all local minimums are global minimums.
      - For convex functions, the above sufficient condition for a point to be a local minimum is also a necessary condition.
        - **Theorem.** Suppose \\(f : \mathbb{R}^n \rightarrow \mathbb{R}\\) is twice differentiable function. Then \\(f\\) is convex iff the 
            Hessian, \\(H = \nabla^2f\\) is a positive semi-definite matrix. Furthermore, \\(f\\) is strictly convex if \\(H\\) is a positive 
            definite matrix.
      - A differentiable convex function \\(f\\) is _**strongly convex**_ with parameter \\(m > 0\\) if for all \\(\pmb{\theta}_1, \pmb{\theta}_2\\) in domain \\(f\\), we have:
        \\[(\nabla f(\pmb{\theta}_1) - \nabla f(\pmb{\theta}_2))^T(\pmb{\theta}_1 - \pmb{\theta}_2) \leq m\|\|\pmb{\theta}_1 - \pmb{\theta}_2\|\|_2^2\\]
        - If the convex function is twice differentiable, then the function is a strongly convex function if \\(f(\pmb{\theta}) \sucseq m\mathrm{I}\\) for all \\(\pmb{\theta}\\) in the domain of \\(f\\).
* A main problem in machine learning is fitting a model to given data samples, essentially solving an optimization problem. An ML model is optimized to produce the best prediction (smallest error). For example (as we will see later), in the linear regression problems, we try to fit a linear model by optimizing its parameters such that the output model can be as close as possible to the observed samples.

## Gradient Decent
* Gradient Descent (GD) is an optimization algorithm used to minimize (or maximize) a function by iteratively adjusting the decision variables in the direction that leads to the steepest rate of change (e.g., decrease for the minimization problems) of the function. It's a fundamental technique and versatile algorithm that underlies many machine learning techniques and models, especially in training models such as linear regression, support vector machines, and neural networks.
* While the basic concept is relatively simple, there are many nuances and variations that can impact its performance, convergence speed, and robustness.
* The main components of a GD algorithm are as follows:
  - **Initialization.** Initialize the parameters (θ) with some initial values. This could be random or based on domain knowledge.
  - For a minimization problem, the gradient of the objective function with respect to its decision variables represents the direction of the steepest decrease in the function. It points to the direction in which the function's output decreases the most. As a result, to minimize the function, we move in the opposite direction of the gradient.
  - **Learning Rate**. The learning rate is a hyperparameter that determines the step size taken in each iteration of the algorithm. It scales the gradient vector and controls the speed of convergence. A small learning rate may cause slow convergence, while a large one might cause overshooting or even divergence.
  - **Iterative Update.** The core of Gradient Descent is the iterative update of the parameters. In each iteration, you compute the gradient of the objective function with respect to the current parameters, multiply it by the learning rate, and subtract it from the current parameters to update them. The update equation for a single parameter might look like this:
 
    - This update decreases the value of the objective function, bringing the optimization variables closer to the optimal values (i.e., local minima).

  - **Convergence Criteria.** You repeat the iterative updates until a convergence criterion is met. This could be a predefined number of iterations or until the change in the objective function's value between iterations becomes very small.

  - Batch Gradient Descent vs. Stochastic Gradient Descent. There are variations of Gradient Descent. Batch Gradient Descent computes the gradient using the entire training dataset in each iteration, which can be slow for large datasets. Stochastic Gradient Descent (SGD) computes the gradient using only a single data point (or a small batch) in each iteration, making it faster but leading to more noisy updates.

  - Mini-Batch Gradient Descent. Mini-Batch Gradient Descent combines the benefits of both batch and stochastic versions. It computes the gradient using a small randomly selected subset (mini-batch) of the training dataset.
