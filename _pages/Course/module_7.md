---
title: "Regression Models"
classes: wide
---

## Introduction
* This is our first exposure to the Machine Leering methods. Recall that the three major components of a machine learning system are data, model, and learning algorithm. The goal of an ML algorithm is to learn the right model or hypothesis using the available data such that the model can predict the unseen data. The choice of the model depends on the nature of the problem. In this module, we start exploring a group of ML methods/models that deal with predicting a real-valued output(s) (also called the dependent variable or target). This class of methods is called regression methods. 
* Regression is a fundamental problem in machine learning, and regression problems appear in a diverse range of research areas and applications including time-series analysis, control and robotics (e.g., reinforcement learning) deep learning applications such as speech-to-text translation, and image recognition. 

* For a regression problem, we need to consider a variety of issues:
    - Choice of the right hypothesis (parametric/non-parametric, linear/non-linear models, etc)
    - Model selection (e.g., choosing the right hyper-parameters)
    - Choosing a learning algorithm
    - Overcoming the overfitting (if the model is memorizing the training data and cannot be generalized to the test data)
    - Choice of loss function (how to choose the loss function according to the underlying probabilistic model)
    - Uncertainty modeling (if the model parameters are deterministic or random)
    
* We'll talk about these issues later.
* Linear models (i.e., the output of the model is linear with respect to its parameters) are the simplest class of regression models. Here, we first discuss parametric regression methods and then overview some non-parametric models. Now let's start with a motivating example:
* Assume that we are given a set of \\(n\\) input-output pairs \\(\mathcal{D}=\\) \\(\\{(\mathbf{x_i}, y_i)\\}_{i=1}^n\\), where \\(\mathbf{x_i}\in \mathbb{R}^p\\) denotes the \\(i^{th}\\) sample which is usually a \\(p\\)-dimensional vector (also called features, independent variables, explanatory variables, or covariate). The goal is to learn a map from inputs, \\(\mathbf{x_i}\\)'s to outputs, \\(y_i\\)'s. For example, consider the following scatter plot, illustrating a dataset in 1-dimension. That is, each red point has one feature and the output is a scaler real number. For example, \\(x_i\\)'s can represent 50 different hours from 1 to 5, and \\(y_i\\)'s denotes the weather temperature in Celsius. We want to build a **Regression model** to predict the temperature for the future hours.

<details>
  <summary>Proof</summary>

    
        ```python
        import numpy as np
        import matplotlib.pyplot as plt
        
        n_samples = 50
        sigam = 0.5
        
        X = np.linspace(1, 5, n_samples)
        X = np.expand_dims(X, 1)
        y = (-np.sin(X) + sigam*np.random.randn(n_samples, 1))
        
        X_lin = np.hstack((np.ones((n_samples, 1)), X))
        w_lin = np.linalg.inv(np.matmul(X.T, X))*np.matmul(X.T, y)
        y_hat_lin = np.matmul(X, w_lin)
        
        
        X_quad = np.hstack((np.ones((n_samples, 1)), X, X**2))
        w_quad = np.matmul(np.linalg.inv(np.matmul(X_quad.T, X_quad)), np.matmul(X_quad.T, y))
        y_hat_quad = np.matmul(X_quad, w_quad)
        
        X_cub = np.hstack((np.ones((n_samples, 1)), X, X**2, X**3))
        w_cub = np.matmul(np.linalg.inv(np.matmul(X_cub.T, X_cub)), np.matmul(X_cub.T, y))
        y_hat_cub = np.matmul(X_cub, w_cub)
        
        
        plt.figure(figsize=(10,3))
        
        plt.subplot(131)
        plt.scatter(X, y, color="red", marker=".", s =200)
        plt.plot(X, y_hat_lin, color="blue", linewidth=2)
        plt.xlabel("Input")
        plt.ylabel("Response")
        plt.title("Linear Fitting")
        
        plt.subplot(132)
        plt.scatter(X, y, color="red", marker=".", s =200)
        plt.plot(X, y_hat_quad, color="blue", linewidth=2)
        plt.xlabel("Input")
        plt.ylabel("Response")
        plt.title("Quadratic Fitting")
        
        plt.subplot(133)
        plt.scatter(X, y, color="red", marker=".", s =200)
        plt.plot(X, y_hat_cub, color="blue", linewidth=2)
        plt.xlabel("Input")
        plt.ylabel("Response")
        plt.title("Cubic Fitting")
        
        plt.subplots_adjust(right=1.3)
        plt.show()
        ```
    
    
</details>

![results](/assets/images/output_2_0.png)

* In the above graph, we see three models used for predicting the weather temperature. The left panel shows a linear fitting (linear regression), the middle panel illustrates a quadratic prediction, and the right one shows a cubic curve fitting. We can see the limitedness of linear regression for this specific dataset as the trend of the data doesn't seem to be linear. 

## Probabilistic model

* Let's recall the goal of regression. We are given observation data, including a set of \\(n\\) i.i.d. input-output pairs (_training data_) \\(\mathcal{D}=\\) \\(\\{(\mathbf{x_i}, y_i)\\}_{i=1}^n\\) drawn from some probability distribution \\(p(\mathbf{x}, y)\\) such that \\(\mathbf{x}\in \mathcal{X}\\) and \\(y\in\mathcal{Y}\\), and we are asked to find a function \\(f:\mathcal{X}\rightarrow\mathcal{Y}\\) that maps any _test_ point \\(\mathbf{x}^{\*}\\) to the corresponding \\(y^{\*}\\) (please note that \\( (\mathbf{x}^{\*}, y^{\*})\sim p(\mathbf{x}, y)\\). What this means that we hope that we can learn a predictor (sometimes called an estimator)  \\(\hat{f} := \hat{f}_n(\mathcal{D}_n)\\) as a function of our training data  that generalizes well to the unseen data drawn from the same distribution \\(p\\). Please note that \\(\mathcal{D}_n\\) is a fixed realization set drawn from the distribution \\(p\\); hence, by changing our training set, the estimator \\(\hat{f}\\) will also be changed; as a result, \\(\hat{f}\\) is a random variable.

* Our approach to finding the estimator \\(\hat{f}\\) is to assume that the observation data has been corrupted by some additive noise (aka observation noise), and we are going to adopt a probabilistic approach and model the noise using a likelihood function. As a result, we can use the maximum likelihood principle to find \\(\hat{f}\\). 
    - For most of our problems, we consider a real-valued (scalar value) response (output). A similar approach usually works for the vector-valued outputs. 
* For regression problems, the observation noise is generally modeled as Gaussian noise; hence, we have the following likelihood function:

\\[p(y*\|\mathbf{x}^{\*}) =\mathcal{N}(y^{\*} \| f(\mathbf{x}^{\*}), \sigma^2)\\]

* This implies the following relationship between a generic input random vector \\(\mathbf{X}\\) and its random output \\(Y\\) with the joint distribution \\(p\\):
\\[Y = f(\mathbf{X}) + \epsilon\\]
    - \\(\epsilon\\) is the observation noise and is distributed as \\(\epsilon\sim\mathcal{N}(0, \sigma^2)\\).
    - For models we discuss here we assume that the observation noise is statistically independent of our data, \\(\mathbf{x}, y\\).

* So, finding the estimator \\(\hat{f}\\) boils down to estimating the mean of the Gaussian distribution using training data. To this end, we use the maximum likelihood approach. The negative log-likelihood (NLL) is given by
\\[\text{NLL}(f) = -\frac{1}{2}\log\sigma^2 + \frac{(Y - f(\mathbf{X}))^2}{2\sigma^2} + \text{cons.}\\]

* So, if we assume \\(\sigma^2\\) is knowm, minimizing the NLL is equivalent to minimizing \\(\frac{(Y - f(\mathbf{X}))^2}{2\sigma^2}\\). This is the squared loss. As we have discussed in the Statistics section ([link](https://mrezasoltani.github.io/_pages/Course/module_4/#what-is-statistics)), the measure of fitness is given by the expected risk, considering the expected performance of the algorithm (model) with respect to the chosen loss function. As a result, our estimator is the solution to the following optimization problem:
\\[f^{\*} = \text{argmin}_{\hat{f}}\mathbb{E} \big{(}Y - \hat{f}(\mathbf{X})\big{)}^2 = \text{argmin} _{\hat{f}}\mathbb{E} _{\mathcal{D}_n} \mathbb{E} _{\mathbf{X}, Y}\\]

    - In the above likelihood expression, please note that we have used \\(\hat{f}\\) instead of \\(f\\). This is because we have written the likelihood function using our training data which results in an estimator \\(\hat{f}\\) (not necessarily optimal one).
     \\[\mathbb{E} \big{(}Y - \hat{f}(\mathbf{X})\big{)}^2 = \int_{\mathcal{X}\times\mathcal{Y}}\big{(}y - \hat{f}(\mathbf{x})\big{)}^2 p(\mathbf{x}, y)d\mathbf{x}dy\\]
    - We note that the inner expectation (the risk) is a random variable as \\(\hat{f}\\) is a r.v.
    - it can be shown that the regression function which minimizes the above expected risk is given by \\(f^*= \hat{f}(\mathbf{x}) = \mathbb{E}\big{(}Y\|\mathbf{x}=\mathbf{X}\big{)}\\).
    
        <details>
          <summary>Proof</summary>
            * Now let's see the optimal solution for the above minimization problem. Using the Law of Iterated Expectations:
            \begin{equation}
                 \mathbb{E} \big{(}Y - \hat{f}(\mathbf{X})\big{)}^2  = \mathbb{E}_{\mathbf{X}}\Big{(}\mathbb{E}_{Y\|\mathbf{X}}\big{(}Y - \mathbb{E}\big{(}Y\|\mathbf{X}=\mathbf{x}\big{)} + \mathbb{E} 
                 \big{(}Y\|\mathbf{X}=\mathbf{x}\big{)} - \hat{f}(\mathbf{X})\big{)}^2 \| \mathbf{X}=\mathbf{x}\Big{)}  \\\\\\\\
                 \Longrightarrow \mathbb{E} \big{(}Y - \hat{f}(\mathbf{X})\big{)}^2 = 
                 \mathbb{E}_{\mathbf{X}}\Big{(}\mathbb{E}_{Y\|\mathbf{X}}\big{(}Y - \mathbb{E}\big{(}Y\|\mathbf{X}=\mathbf{x}\big{)}\|\mathbf{X}=\mathbf{x}\big{)}^2 
                 + 2\mathbb{E}_{Y\|\mathbf{X}}\big{(}\big{(}Y - \mathbb{E}\big{(}Y\|\mathbf{X} = \mathbf{x}\big{)}\big{)}\big{(}\mathbb{E}\big{(}Y\|\mathbf{X}=\mathbf{x}\big{)} -  
                 \hat{f}(\mathbf{X})\big{)}\|\mathbf{X} = \mathbf{x}\big{)}
                 + \mathbb{E}_{Y\|\mathbf{X}=\mathbf{x}}\big{(}\mathbb{E}\big{(}Y\|\mathbf{X}=\mathbf{x}\big{)} - \hat{f}(\mathbf{X})\big{)}^2\|\mathbf{X}=\mathbf{x}\Big{)} \\\\\\\\
                 \Longrightarrow \mathbb{E} \big{(}Y - \hat{f}(\mathbf{X})\big{)}^2 = 
                 \mathbb{E}_{\mathbf{X}}\Big{(}\mathbb{E}_{Y\|\mathbf{X}}\big{(}Y - \mathbb{E}\big{(}Y\|\mathbf{X}=\mathbf{x}\big{)}\|\mathbf{X}=\mathbf{x}\big{)}^2 
                 +2\mathbb{E}_{Y\|\mathbf{X}}\big{(}\mathbb{E}\big{(}Y\|\mathbf{X}=\mathbf{x}) -\hat{f}(\mathbf{X}\big{)}\big{)}\|\mathbf{X} = \mathbf{x}\big{)}\times 0
                  + \mathbb{E}_{Y\|\mathbf{X}=\mathbf{x}}\big{(}\mathbb{E}\big{(}Y\|\mathbf{X}=\mathbf{x}\big{)} - \hat{f}(\mathbf{X})\big{)}^2\|\mathbf{X}=\mathbf{x}\Big{)} \\\\\\\\
                  \Longrightarrow \mathbb{E} \big{(}Y - \hat{f}(\mathbf{X})\big{)}^2 \geq  \mathbb{E}\big{(}Y - \mathbb{E}\big{(}Y\|\mathbf{X}=\mathbf{x}\big{)}\big{)}^2 
            \end{equation}
            
       </details>

* Where the minimum in the last inequality is achieved if we choose
                \\(\hat{f}(\mathbf{x})=\mathbb{E}\big{(}Y\|\mathbf{X}=\mathbf{x}\big{)}\\).
  
* If we replace the minimizer, \\(\mathbb{E}\big{(}Y\|\mathbf{x}=\mathbf{X}\big{)}\\) in the expected risk expression, we find the following Bias-variance trade-off:
\\[\mathbb{E} \big{(}Y - \hat{f}(\mathbf{X})\big{)}^2 = \sigma^2 + \text{Bias}^2(\mathbf{X}) + \text{Var}(\mathbf{X})\\]

* where
    - \\(\text{Bias}(\mathbf{X}) = \mathbb{E}\big{(}\hat{f}(\mathbf{X})\big{)} - f(\mathbf{X})\\)
    - \\(\text{Var}(\mathbf{X}) = \mathbb{E}\big{(}\hat{f}(\mathbf{X}\big{)} - f(\mathbf{X})\big{)}^2\\)
    - \\(\sigma^2 = \mathbb{E}\big{(}Y-f(\mathbf{X})\big{)}\\)

## Parametric models

* One way to solve the above problem is to assume that our estimator \\(\hat{f}: \mathbb{R}^p\rightarrow \mathbb{R}\\) has a parametrix form, \\(\hat{f}(\mathbf{x}, \pmb{\theta})\\), where \\(\pmb{\theta}\in \mathbb{R}^k\\) denotes a set of parameters such that \\(k = o(n)\\), that is, \\(k\\) doesn't grow with the number of samples. If \\(k=O(p)\\), the model is called under-parametrized (i.e., we are in low-dimensional space), while models with \\(k >> p\\) are called over-parametrized (i.e., we are in high-dimensional space). One example of the over-parametrized models is Deep Neural Networks (DNNs).

### Linear Regression

* We first focus on the linear regression models; as a result, we may assume that the output is a linear function of the input. * We note that this is just an assumption, and it may not be valid or realistic, but certainly, the linear models are the simplest model we can start with. 
* Hence, \\(\hat{f}(\mathbf{x}, \pmb{\theta}) = b + \mathbf{w}^T\mathbf{x}\\), where we consider the case \\(k=p\\). Accordingly, the likelihood function is given by:
\\[p(y\|\mathbf{x}) =\mathcal{N}(y \| b + \mathbf{w}^T \mathbf{x}, \sigma^2)\\]
    - Here, \\(\pmb{\theta}(\mathbf{w}, b, \sigma^2)\\) are all the parameters of the model. 
    - The vector of parameters \\(w_1, w_2,\ldots, w_p\\) are known as the weights or regression coefficients. Each coefficient       \\(w_i\\) specifies the change in the output we expect if we change the corresponding input feature xd by one unit.

* There are fundamental assumptions to use a linear regression model.

* The key property of the model is that the expected value of the output is assumed to be a linear function of the input, E[yjx] = wTx, which makes the model easy to interpret, and easy to fit to data. We discuss nonlinear extensions later in this book.


```python

```
