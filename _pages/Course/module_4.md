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
  - Let \\(\mathbf{x}\in\mathcal{X}\\) be a realization from a random vector \\(\mathbf{X}\sim p(\mathbf{x})\\). An estimator of the parameter \\(\pmb{\theta}\\) is a function of the given data samples with size \\(n\\) denoted by \\(\hat{\pmb{\theta}} _n(\mathbf{x}) = \hat{\pmb{\theta}}_n =  \hat{\pmb{\theta}}\\).
* In the point estimation problems, we want to estimate our parameters, and we don't quantify how confident we are about the estimation.
* Since \\(\hat{\pmb{\theta}}\\) is an estimate of the ground-truth parameter \\(\pmb{\theta}\\), there is going to be (inevitably) an error. That is, the error vector is given by \\(\pmb{e} = \pmb{\theta} - \hat{\pmb{\theta}}\\). The goal of statistical models is to minimize this error in some sense.
* To capture the notion of minimization of the error, we use the concept of a loss function, i.e., the loss incurred due to the error in estimating the ground truth parameter \\(\pmb{\theta}\\). This is denoted by \\(\mathcal{L}(\pmb{\theta}, \hat{\pmb{\theta}})\\). Depending on the nature of the problem, there are different loss functions. Two most common of these are \\(\ell_2 = \\| \pmb{\theta}- \hat{\pmb{\theta}} \\| _2^2\\) loss and \\(\ell_1 = \\| \pmb{\theta}- \hat{\pmb{\theta}} \\| _1\\) loss functions.
* Since the estimator \\(\hat{\pmb{\theta}}\\) is a function of the observed samples (a realization of the random samples drawn from the underlying distribution), it is a random variable. Hence, in statistical problems, the expectation of the loss function with respect to the data distribution is considered a measure of the goodness of a model. This is called _risk_, and is given by \\(\mathbb{E}\[\mathcal{L}(\pmb{\theta}, \hat{\pmb{\theta}} )\]\\).
* In principle, point estimation is guided by risk minimization, that is, by the search for estimators that minimize the risk.
* The common risk minimization are given the by _mean absolute error (MAE)_ risk by choosing the loss function as \\(\ell_1\\) norm, and the _mean squared error (MSE)_ by choosing the loss function as \\(\ell_2\\) norm. The square root of the mean squared error is called the _root mean squared error (RMSE)_.
* There are different ways to evaluate the estimators. Two common criteria for evaluating estimators are unbiasedness and consistency.
  - Unbiasedness. An estimator is an unbiasedness estimator iff \\(\mathbb{E}(\hat{\pmb{\theta}}(\mathbf{x})) = \pmb{\theta}\\).
  - Consistency. An estimator is a consistent estimator (or weakly consistent) iff the estimator converges to the ground-truth parameter \\(\pmb{\theta}\\) when the sample size goes to infinity, i.e., \\(\lim_{n \rightarrow \infty} \hat{\pmb{\theta}} = \pmb{\theta}\\) in probability. Here \\(n\\) is the sample size used to estimate parameter \\(\pmb{\theta}\\).
* The most common approach to parameter estimation is to choose the parameters that assign the highest probability to the observed data samples; this is called _**maximum likelihood estimation or MLE**_. So the problem can be formulated as follows: let \\(\mathbf{x} _i\stackrel{iid}{\sim} q(\mathbf{x})\\) for \\(i=1,2,\ldots,n\\). Here \\(q(\mathbf{x})\\) is the **unknown** underlying probability distribution that generates data samples, including the realization samples, \\(\mathbf{x} _i\\)'s.
* In parametric point estimation, we focus on a parametric family of probability distributions, and try to find a closest parametric distribution as the best proxy for the true distribution, \\(q(\mathbf{x})\\). We denote this parametric distribution by \\(p(\mathbf{x} \| \pmb{\theta})\\).
* In order to charecteristic "the closest", we need to define the followings:
  - There are many ways to formulate the distance/similarity between two probability distributions. Here we consider the most commom measure of simialrity known as _Kullback Leibler Divergence (KL)_. This measure is not quite a distance as it is not symmetric. We'll elaborate on this later. The KL between two probability distributions of \\(q(\mathbf{x})\\) and \\( p(\mathbf{x} \| \pmb{\theta})\\) is defined as follows:
 \\[D(q(\mathbf{x}) \\| p(\mathbf{x} \| \pmb{\theta})) = \mathbb{E}\[ \log \frac{q(\mathbf{x})}{p(\mathbf{x} \| \pmb{\theta})}\] = \int _{\mathbf{x} \in \mathcal{X}} \log\frac{q(\mathbf{x})}{p(\mathbf{x} \| \pmb{\theta})}q(\mathbf{x})d\mathbf{x}\\]
  - KL divergence is always non-negative, \\(D(q(\mathbf{x}) \\| p(\mathbf{x} \| \pmb{\theta}))\geq 0\\) with equality holds iff \\(q(\mathbf{x}) = p(\mathbf{x} \| \pmb{\theta})\\).
* The MLE denoted by \\(\hat{\pmb{\theta}}_n\\) is obtained by the following optimization problem:
  \\[\hat{\pmb{\theta}} _{mle} = \hat{\pmb{\theta}}_n = argmin _{\pmb{\theta}} -\log p(\mathbf{x} \| \pmb{\theta})\\]
  - Here, \\(-\log p(\mathbf{x} \| \pmb{\theta})\\) is denoted by \\(NLL(\pmb{\theta})\\) and called _negative log-likelihood or NLL_.
* It can be shown that the MLE is equivalent to the minimization of KL Divergence, \\(D(q(\mathbf{x}) \\| p(\mathbf{x} \| \pmb{\theta}))\geq 0\\).
* The log-likelihood of \\(n\\) i.i.d data samples is given by
  \\[NLL(\pmb{\theta}) = \log(\prod _{i=1}^n p(\mathbf{x_i} \| \pmb{\theta})) =-\sum _{i=1}^{n}\log p(\mathbf{x_i} \| \pmb{\theta}) \\]
* **Example.** MLE for the Bernoulli distribution
  - Consider tossing a coin experiment. Here, we don't know what the probability of landing head is; hence, we'd like to estimate it using \\(n\\) observations. That is, we toss the coin for \\(n\\) times and every time we record what has been landed. To formulate this mathematically, let \\(X\\) be a Bernoulli r.v. such that \\(X=1\\) is the event of landing a head and \\(X=0\\) be the event of landing tail. Let \\(P(X=1) = \theta\\); hence, \\(P(X=0) = 1-\theta\\). We now use MLE to estimate our parameter \\(\theta\\) which is the probability of seeing a head. As a result,
    \begin{equation}
        \begin{aligned}
          NLL(\theta) = -\log \prod _{i=1}^n P(X_i) = -\log \prod _{i=1}^n \theta^{ \mathbb{1} _{\\{X_i=1\\}} }(1-\theta)^{\mathbb{1} _{\\{X_i=0\\}}} \\\\\\\\
                      = -\sum _{i=1}^n \mathbb{1} _{\\{X_i=1\\}}\log \theta + \mathbb{1} _{\\{X_i=0\\}}\log(1-\theta)
        \end{aligned}
    \end{equation}
  - Now we are looking for a \\(\theta\\) to minimize this (\\(\hat{\theta} _{mle}\\)). From calculous, we can find this \\(\theta\\) by taking derivative of the NLL and equating it with zero:
    \\[\frac{d}{d\theta}NLL(\theta) = -\sum _{i=1}^n \mathbb{1} _{\\{X_i=1\\}}\frac{1}{\theta} + \mathbb{1} _{\\{X_i=0\\}}\frac{1}{\theta-1} = 0 \\]
    \begin{equation}
     \begin{aligned}
          \Longrightarrow \frac{\theta-1}{\theta} = \frac{\sum _{i=1}^n\mathbb{1} _{\\{X_i=0\\}}}{\sum _{i=1}^n\mathbb{1} _{\\{X_i=1\\}}}
        \end{aligned}
    \end{equation}
