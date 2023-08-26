---
title: "Module 5 -- Optimization"
classes: wide
---
# What is Optimization?
* Optimization is a mathematical framework referring to finding the best solution for maximization or minimization problems. For example, we have learned from the statistics section that we need to find the best parameter \\(\pmb{\theta}\\) for a distribution that can explain the observed data with the highest probability. In this case, the maximum likelihood estimation boils down to an optimization problem in which our goal is to minimize the negative likelihood (NLL), known as _loss (cost) function_ denoted by \\(\mathcal{L}(\pmb{\theta})\\). Hence, we can write our optimization problem as:
  \\[\pmb{\theta}^* = \text{argmin} _{\pmb{\theta}mathcal{L}(\pmb{\theta})\\]
* In machine learning fitting a model to the given data samples is a core problem which essentially means solving an optimization problem. That is, an ML model is optimized to produce the best prediction (smallest error). As we will see in the linear regression problems, we try to fit a linear model by optimizing its parameters such that the output model can be as close as possible to the observed samples.
* Here, we will assume that the parameter space given by \\(\pmb{\theta}\in \mathbb{\Theta}^p\\) is a continuous space; hence, we deal with continuous optimization. Later in this course, we talk about discrete optimization problems. 
