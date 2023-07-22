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
   | **A square, diagonal matrix with diagonal entries given by** \\(\mathbf{x}\\) | \\(diag(a)\\) |
   | **Real Set** | \\(\mathbb{R}\\)|
   | **Complex Set** | \\(\mathbb{C}\\)|
   | **Standard/Canonical Basis Vector with only 1 in** \\(i^{th}\\) **position** | \\(\e_i=(0, 0, \ldots, 0, 1, 0, \dots, 0 )\\)|
   | **Graph and Operator** | \\(\mathcal{X}\\)|
   | **Element \\(i\\) of vector** \\(x\\) | \\(x_i\\) |
   | **All elements of vector** vector** \\(x\\) **except for element** \\(i\\) of  | \\(x_{-i}\\) |
   | \\(i^{th}\\) **row (column) of a matrix** \\(\mathrm{X}\\) | \\(\mathrm{X_{i:}}\\)(\\(\mathrm{X_{:i}}\\)) |
   | **A probability distribution over a discrete variable (pmf)** | \\(P(x)\\) |
   | **A probability distribution over a continuous variable (pdf)** | \\(p(x)\\) |
   | \\(L^p\\)) **norm of** x | \\(||x||_p\\) |

* A vector \\(\mathbf{x} \in \mathbb{R}^n \\) is a collection of \\(n\\) numbers defined on real, \\(\mathbb{R}\\) or complex, \\(\mathbb{C}\\) field. In this course, we use a column vector to denote a vector \\(\mathbf{x}\\). Also, a matrix \\(\mathrm{X} \in \mathbb{R}^{m\times n}\\) is a 2-d array of \\(mn\\) numbers, arranged in \\(m\\) rows and \\(n\\) columns:

\begin{equation}
\begin{aligned}
  \mathbf{x} =  
  \begin{pmatrix}
    x_1 \\\\\\\\
    x_2 \\\\\\\\
    \vdots \\\\\\\\
    x_n
  \end{pmatrix}, & & & &
  \mathrm{X} = 
    \begin{pmatrix}
      x_{11} & x_{12} & \ldots & x_{1n} \\\\\\\\
      x_{21} & x_{22} & \ldots & x_{2n} \\\\\\\\
      \vdots & \vdots & \ddots & \vdots \\\\\\\\
      x_{m1} & x_{m2} & \ldots & x_{mn}
    \end{pmatrix} 
\end{aligned}
\end{equation}
