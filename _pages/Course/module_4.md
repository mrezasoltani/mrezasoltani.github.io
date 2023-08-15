---
title: "Module 4 -- Statistics"
classes: wide
---
## What Is Statistics?
* Informally speaking, statistics aims to infer the properties and characteristics of the underlying distribution of data given some set of samples/observations/measurements drawn from this unknown distribution. For example, suppose the target distribution is parametrized by parameter \\(\pmb{\theta}\\). In that case, the goal is to find the best \\(\pmb{\theta}\\) in some sense to agree with the set of the collected observations/samples. As a result, we have three main components of every statistic problem:
  - A probability distribution generating the data
  - Samples/Observations/Measurements. We are given \\(n\\) samples drawn from the probability distribution generating the data.
  - Properties of the distribution about which inferences are drawn
* In general, there are two approaches for statistical problems: parametric methods and non-parametric methods. parametric methods assume strong assumptions about data distribution of the data such as having a fixed number of parameters with respect to the number of samples, while non-parametric methods make fewer assumptions about the underlying distribution of the data (e.g., no assumption made for the form of the distribution) and assume that the number of parameters of the data distribution grows with the number of data samples.
* Statistic problems can be categorized into one of three areas of _point estimation_, _hypothesis testing_, and _interval (set) estimation_, where depending on the methods, we can approach these problems as frequentists or in a Bayesian way.
  - Point estimation produces a single estimate which is an approximation of the true parameter.  In set estimation, we produce a set of estimates which is supposed to include the ground-truth parameter with high probability.

### Point Estimation
* Here we focus on parametric methods and assume that our unknown underlying distribution is parametrized by parameter \\(\pmb{\theta}\\). The process of estimating \\(\pmb{\theta}\\) from the data set is called model fitting, or training, and is a central problem in machine learning (we state our results for the continuous random variables. The same thing holds for the discrete case).
  - Let \\(\mathbf{x}\in\mathcal{X}\\) be a realization from a random vector \\(\mathbf{X}\sim p(x)\\). An estimator of the parameter \\(\pmb{\theta}\\) is a function of the given data samples with size \\(n\\), i.e., parameter \\(\hat{\pmb{\theta}} _n(\mathbf{x}) = \hat{\pmb{\theta}}_n =  \hat{\pmb{\theta}} = \pmb{\theta}\\).
* In the point estimation problems, we want to estimate our parameters, and we don't quantify how confident we are about the estimation.
* Since \\(\hat{\pmb{\theta}}\\) is an estimate of the ground-truth parameter \\(\pmb{\theta}\\), there is going to be (inevitably) an error. That is, the error vector is given by \\(\pmb{e} = \pmb{\theta} - \hat{\pmb{\theta}}\\). The goal of statistical models is to minimize this error in some sense.
* To capture the notion of minimization of the error, we use the concept of a loss function, i.e., the loss incurred due to the error in estimating the ground truth parameter \\(\pmb{\theta}\\). This is denoted by \\(\mathcal{L}(\pmb{\theta}, \hat{\pmb{\theta}})\\). Depending on the nature of the problem, there are different loss functions. Two most common of these are \\(\ell_2 = \\| \pmb{\theta}- \hat{\pmb{\theta}} \\| _2^2\\) loss and \\(\ell_1 = \\| \pmb{\theta}- \hat{\pmb{\theta}} \\| _1\\) loss functions.
* Since the estimator \\(\hat{\pmb{\theta}}\\) is a function of the observed samples (a realization of the random samples drawn from the underlying distribution), it is a random variable. Hence, in statistical problems, the expectation of the loss function with respect to the data distribution is considered a measure of the goodness of a model. This is called _risk_, and is given by \\(\mathbb{E}\[\mathcal{L}(\pmb{\theta}, \hat{\pmb{\theta}} )\]\\).
* In principle, point estimation is guided by risk minimization, that is, by the search for estimators that minimize the risk.
* The common risk minimization are given the by _mean absolute error (MAE)_ risk by choosing the loss function as \\(\ell_1\\) norm, and the _mean squared error (MSE)_ by choosing the loss function as \\(\ell_2\\) norm. The square root of the mean squared error is called the _root mean squared error (RMSE)_.
* There are different ways to evaluate the estimators. Two common criteria for evaluating estimators are unbiasedness and consistency.
  - Unbiasedness. An estimator is an unbiasedness estimator iff \\(\mathbb{E}(\hat{\pmb{\theta(\mathrm{x})}}) = \pmb{\theta}\\).
  - Consistency. An estimator is a consistent estimator (or weakly consistent) iff the estimator converges to the ground-truth parameter \\(\pmb{\theta}\\) when the sample size goes to infinity, i.e., \\(\lim_{n \rightarrow \infty} \hat{\pmb{\theta}} = \pmb{\theta}\\) in probability. Here \\(n\\) is the sample size used to estimate parameter \\(\pmb{\theta}\\).
* The most common approach to parameter estimation is to choose the parameters that assign the highest probability to the observed data samples; this is called _**maximum likelihood estimation or MLE**_.
* 
