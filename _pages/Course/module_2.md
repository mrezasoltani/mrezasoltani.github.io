---
title: "Module-2 -- Linear Algebra"
classes: wide
---

Linear algebra is the study of matrices and vectors. In this module, we start with some basic concepts in linear algebra, and toward the end of this part, we will see more advanced tools. 

* **Math Notations**. Throughout this course, we use the following conventions:

  |       Variable type       | Symbol             |
   | -------------| ---------------------- |
   | **Deterministic scalar variable**   | \\(x\\) |
   | **Random scalar variable** | \\(X\\)|
   | **Deterministic vector** | \\(\mathbf{x}\\)|
   | **Random vector** | \\(\mathbf{X}\\)|
   | **Deterministic matrix/tensor** | \\(\mathrm{X}\\)|
   | **Random matrix/tensor** | \\(\mathbfit{X}\\)|
   | **Graph and operator** | \\(\mathcal{X}\\)|
   | **A square, diagonal matrix with diagonal entries given by** \\(\mathbf{x}\\) | \\(diag(a)\\) |
   | **Real set** | \\(\mathbb{R}\\)|
   | **Complex set** | \\(\mathbb{C}\\)|
   | **Standard/Canonical basis vector with only 1 in** \\(i^{th}\\) **position** | \\(e_i=[0, 0, \ldots, 0, 1, 0, \dots, 0 ]^T\\)|
   | **Element \\(i\\) of vector** \\(\mathbf{x}\\) | \\(x_i\\) |
   | **All elements of vector** \\(\mathbf{x}\\) **except for element** \\(i\\)  | \\(\mathbf{x_{-i}}\\) |
   | **Entry in** \\(i^{th}\\) **row and** \\(j^{th}\\) **column of a matrix** \\(\mathrm{X}\\) | \\(x_{ij}\)) |
   | \\(i^{th}\\) **row (column) of a matrix** \\(\mathrm{X}\\) | \\(\mathrm{X_{i:}}\\) (\\(\mathrm{X_{:i}}\\)) |
   | **A probability distribution over a discrete variable (pmf)** | \\(P(x)\\) |
   | **A probability distribution over a continuous variable (pdf)** | \\(p(x)\\) |
   | \\(L^p\\) **norm of vector** \\(\mathbf{x}\\) | \\(\\|\\|\mathbf{x}\\|\\|_p\\) |

* A vector \\(\mathbf{x} \in \mathbb{R}^n \\) is a collection of \\(n\\) numbers defined on real, \\(\mathbb{R}\\) or complex, \\(\mathbb{C}\\) field. In this course, we use a column vector to denote a vector \\(\mathbf{x}\\). In the above table, we have shown the Canonical basis vector, \\(e_i\\) with a superscript \\(T\\) to denote the transpose of a row vector is a column one. A matrix \\(\mathrm{X} \in \mathbb{R}^{m\times n}\\) is a 2-d array of \\(mn\\) numbers, arranged in \\(m\\) rows and \\(n\\) columns:

\begin{equation}
\begin{aligned}
  \mathbf{x} =  
  \begin{bmatrix}
    x_1 \\\\\\\\
    x_2 \\\\\\\\
    \vdots \\\\\\\\
    x_n
  \end{bmatrix}, & & & &
  \mathrm{X} = 
    \begin{bmatrix}
      x_{11} & x_{12} & \ldots & x_{1n} \\\\\\\\
      x_{21} & x_{22} & \ldots & x_{2n} \\\\\\\\
      \vdots & \vdots & \ddots & \vdots \\\\\\\\
      x_{m1} & x_{m2} & \ldots & x_{mn}
    \end{bmatrix} 
\end{aligned}
\end{equation}

* If \\(m=n\\), then the matrix is called a _square_ matrix. We can also show a matrix \\(\mathrm{X} \in \mathbb{R}^{m\times n}\\) either by its columns, or by rows:

\begin{equation}
\begin{aligned}
  \mathrm{X} =  
    \begin{bmatrix}
      | & | & \ldots & | \\\\\\\\
      \mathrm{X_{:1}} & \mathrm{X_{:2}} & \ldots & \mathrm{X_{:n}} \\\\\\\\
      | & | & \ldots & | \\\\\\\\
    \end{bmatrix}, & & & &
  \mathrm{X} =  
    \begin{bmatrix}
      | & | & \ldots & | \\\\\\\\
      \mathrm{X_{:1}} & \mathrm{X_{:2}} & \ldots & \mathrm{X_{:n}} \\\\\\\\
      | & | & \ldots & | \\\\\\\\
    \end{bmatrix}, & & & &
\end{aligned}
\end{equation}
