---
title: "Module-2 -- Linear Algebra"
classes: wide
---

Linear algebra is the study of matrices and vectors. In this module, we start with some basic concepts in linear algebra, and toward the end of this part, we will see more advanced tools. 

* **Math Notations**. Throughout this course, we use the following conventions:

  |       Variable type       | Symbol             |
   | -------------| ---------------------- |
   | **Deterministic Scalar Variable**   | \\(x\\) |
   | **Random Scalar Variable** | \\(X\\)|
   | **Deterministic Vector** | \\(\mathbf{x}\\)|
   | **Random Vector** | \\(\mathbf{X}\\)|
   | **Deterministic Matrix/Tensor** | \\(\mathrm{X}\\)|
   | **Random Matrix/Tensor** | \\(\mathbfit{X}\\)|
   | **Graph and Operator** | \\(\mathcal{X}\\)|

* A vector \\(\mathbf{x} \in \mathbb{R}^n \\) is a collection of \\(n\\) numbers defined on real, \\(\mathbb{R}\\) or complex, \\(\mathbb{C}\\) field. In this course, we use a column vector to denote a vector \\(\mathbf{x}\\). Also, a matrix \\(\mathbf{X} \in \mathbb{R}^{m\times n}\\) is a 2-d array of \\(mn\\) numbers, arranged in \\(m\\) rows and \\(n\\) columns:

\begin{equation}
\begin{aligned}
  \mathbf{x} =  
  \begin{pmatrix}
    x_1 \\\\\\\\
    x_2 \\\\\\\\
    \vdots \\\\\\\\
    x_n
  \end{pmatrix}, & & & &
  \mathbf{X} = 
    \begin{pmatrix}
      x_{11} & x_{12} & \ldots & x_{1n} \\\\\\\\
      x_{21} & x_{22} & \ldots & x_{2n} \\\\\\\\
      \vdots & \vdots & \ddots & \vdots \\\\\\\\
      x_{m1} & x_{m2} & \ldots & x_{mn}
    \end{pmatrix} 
\end{aligned}
\end{equation}
