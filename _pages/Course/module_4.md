---
title: "Module 4 -- Statistics"
classes: wide
---
## What Is Statistics?
* Informally speaking, statistics aims to infer the underlying distribution of data given some set of samples/observations/measurements drawn from this unknown distribution. For example, suppose the target distribution is parametrized by parameter \\(\pmb{\theta}\\). In that case, the goal is to find the best \\(\pmb{\theta}\\) in some sense to agree with the set of the collected observations/samples.
* In general, there are two approaches for statistical problems: parametric methods and non-parametric methods. parametric methods assume strong assumptions about data distribution of the data such as having a fixed number of parameters with respect to the number of samples, while non-parametric methods make fewer assumptions about the underlying distribution of the data (e.g., no assumption made for the form of the distribution) and assume that the number of parameters of the data distribution grows with the number of data samples.
* Statistic problems can be categorized into one of three areas of _point estimation_, _hypothesis testing_, and _interval estimation_, where depending on the methods, we can approach these problems as frequentists or in a Bayesian way.

### Point Estimation
* Here we focus on parametric methods and assume that our unknown underlying distribution is parametrized by parameter \\(\pmb{\theta}\\). The process of estimating \\(\pmb{\theta}\\) from the set of data is called model fitting, or training, and is a central problem in machine learning.
* In the point estimation problems, we want to estimate our parameters, and we don't quantify how confident we are about the estimation. 
* The most common approach to parameter estimation is to choose the parameters that assign the highest probability to the observed data samples; this is called __**maximum likelihood estimation or MLE**_.
* 
