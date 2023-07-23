---
title: "Module 2 -- Linear Algebra"
classes: wide
---

Linear algebra is the study of matrices and vectors. In this module, we start with some basic concepts in linear algebra, and toward the end of this part, we will see more advanced tools. 

## Math Notations 
* Throughout this course, we use the following conventions:

  |       Variable Type       | Symbol             |
   | -------------| ---------------------- |
   | **Deterministic scalar variable**   | \\(x\\) |
   | **Random scalar variable** | \\(X\\)|
   | **Deterministic vector** | \\(\mathbf{x}\\)|
   | **Random vector** | \\(\mathbf{X}\\)|
   | **Deterministic matrix/tensor** | \\(\mathrm{X}\\)|
   | **Random matrix/tensor** | \\(\mathbfit{X}\\)|
   | **Graph and operator** | \\(\mathcal{X}\\)|
   | **A square, diagonal matrix with diagonal entries given by a vector** \\(\mathbf{x}\\) | \\(diag(\mathbf{x})\\) |
   | **Vector of a diagonal entries of a matrix** \\(\mathrm{X}\\) | \\(diag(\mathrm{X})\\) |
   | **Real set** | \\(\mathbb{R}\\)|
   | **Complex set** | \\(\mathbb{C}\\)|
   | **Standard/Canonical basis vector with only 1 in** \\(i^{th}\\) **position** | \\(e_i=[0, 0, \ldots, 0, 1, 0, \dots, 0 ]^T\\)|
   | **Element \\(i\\) of vector** \\(\mathbf{x}\\) | \\(x_i\\) |
   | **All elements of vector** \\(\mathbf{x}\\) **except for element** \\(i\\)  | \\(\mathbf{x_{-i}}\\) |
   | **Entry in** \\(i^{th}\\) **row and** \\(j^{th}\\) **column of a matrix** \\(\mathrm{X}\\) | \\(x_{ij}\\) |
   | \\(i^{th}\\) **row (column) of a matrix** \\(\mathrm{X}\\) | \\(\mathrm{X_{i:}}\\) (\\(\mathrm{X_{:i}}\\)) |
   | **A probability distribution over a discrete variable (pmf)** | \\(P(x)\\) |
   | **A probability distribution over a continuous variable (pdf)** | \\(p(x)\\) |
   | \\(L^p\\) **norm of vector** \\(\mathbf{x}\\) | \\(\\|\\|\mathbf{x}\\|\\|_p\\) |

## Vector and Matrix
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
* Column view of a matrix:

\begin{equation}
\begin{aligned}
  \mathrm{X} =  
    \begin{bmatrix}
      | & | & \ldots & | \\\\\\\\
      \mathrm{X_{:1}} & \mathrm{X_{:2}} & \ldots & \mathrm{X_{:n}} \\\\\\\\
      | & | & \ldots & |
    \end{bmatrix} =
    \begin{bmatrix}
      \mathrm{X_{:1}} & \mathrm{X_{:2}} & \ldots & \mathrm{X_{:n}}
    \end{bmatrix}
\end{aligned}
\end{equation}

* Row view of a matrix:

\begin{equation}
\begin{aligned}
  \mathrm{X} =  
    \begin{bmatrix}
      ---  & \mathrm{X_{1:}}^T & ---  \\\\\\\\
      ---  & \mathrm{X_{2:}}^T & ---  \\\\\\\\
       & \vdots & \\\\\\\\
      ---  & \mathrm{X_{m:}}^T & --- 
    \end{bmatrix} =
    \begin{bmatrix}
      \mathrm{X_{1:}} & \mathrm{X_{2:}} & \ldots & \mathrm{X_{m:}}
    \end{bmatrix}^T
\end{aligned}
\end{equation}

## Addition, Subtraction, and Scaling Vectors

* Addition, subtraction, and scaling vectors have been illustrated in the following figure.

    <p align="center">
    <img width="842" alt="Screenshot 2023-07-22 at 11 54 22 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/746371b3-d4d8-49b1-baa4-b441865919b6">
    <br>
      <em> Addition, subtraction, and scaling of vectors.</em>
    </p>
* Alegbriaclly, given two vectors of \\(\mathbf{x}\\), \\(\mathbf{y} \in \mathbb{R}^{n} \\), their addition is given by \\(\mathbf{x} + \mathbf{y} = [x_1+y_1 ~~ x_2+y_2 ~~ \ldots ~~ x_n+y_n]^T \in \mathbb{R}^{n} \\). Also, \\(\alpha\mathbf{x} = [\alpha x_1 ~~ \alpha x_2 ~~\dots ~~\alpha x_n]^T \in \mathbb{R}^{n} \\) for some \\(\alpha \in \mathbb{R}\\).
* Addition, subtraction, and scaling of matrices are simialr to the vectors (vector is a specila matrix). Alegbriaclly, given two matrices of \\(\mathrm{X}\\), \\(\mathrm{Y} \in \mathbb{R}^{m\times n} \\) and for any scalar \\(alpha \in \mathbb{R}\\):

\begin{equation}
\begin{aligned}
  \mathrm{X} \pm \mathrm{Y} =  
    \begin{bmatrix}
      x_{11}\pm x_{11} & x_{12}\pm y_{12} & \ldots & x_{1n}\pm y_{1n} \\\\\\\\
      x_{21}\pm y_{21} & x_{22}\pm y_{22} & \ldots & x_{2n}\pm y_{2n} \\\\\\\\
      \vdots & \vdots & \ddots & \vdots \\\\\\\\
      x_{m1}\pm y_{m1} & x_{m2}\pm y_{m2} & \ldots & x_{mn}\pm y_{mn}
    \end{bmatrix}, & & & &
  \mathrm{\alpha X} = 
    \begin{bmatrix}
      \alpha x_{11} & \alpha x_{12} & \ldots & \alpha x_{1n} \\\\\\\\
      \alpha x_{21} & \alpha x_{22} & \ldots & \alpha x_{2n} \\\\\\\\
      \vdots & \vdots & \ddots & \vdots \\\\\\\\
      \alpha x_{m1} & \alpha x_{m2} & \ldots & \alpha x_{mn}
    \end{bmatrix}
\end{aligned}
\end{equation}

* **Definition.** Consider a set of \\(p\\) matrices (vectors) \\(\\{\mathrm{X_1}, \mathrm{X_1}, \dots, \mathrm{X_p}\\}\\) with dimension \\(\mathbb{R}^{m\times n}\\)  \\((\\{\mathbf{x_1}, \mathbf{x_2}, \ldots, \mathbf{x_p}\\}\\) with dimension \\(\mathbb{R}^{p})\\). A matrix \\(\mathrm{Y} \in \mathbb{R}^{m\times n}\\) \\(( a vector \mathbf{y} \in  \mathbb{R}^{n})\\) is a _linear combination_ of \\(\\{\mathrm{X_1}, \mathrm{X_1}, \dots, \mathrm{X_p}\\}\\) ( \\(\\{\mathbf{x_1}, \mathbf{x_2}, \ldots, \mathbf{x_p}\\}\\) ) if and only if there exist \\(p\\) scalars coefficients \\(\\{\alpha_1, \alpha_1, \ldots, \alpha_1\\}\\) such that \\(\mathrm{Y} = \sum_{i=1}^{p} \alpha_i \mathrm{X_i}\\) \\((\mathbf{y} = \sum_{i=1}^{p} \alpha_i \mathbf{x_i})\\).

## Propoerties of Matrices
* **Aassociativity.** Multiplication of a matrix by a scalar is associative: for any matrix \\(mathrm{X}\\) and for any scalars \\(alpha\\) and \\(\beta\\), we have \\(\alpha (\beta \mathrm{X}) = (\alpha \beta)\mathrm{X}\\).
* **Distributivity w.r.t addition.** Multiplication of a matrix by a scalar is distributive with respect to matrix addition: for any matrices \\(\mathrm{X}\\) and \\(\mathrm{Y}\\) and for any scalar \\(alpha\\), we have \\(\alpha (\mathrm{X}+\mathrm{Y}) = \alpha \mathrm{X} + \alpha \mathrm{Y}\\).
* **Distributive property w.r.t multiplication.** Multiplication of a matrix by a scalar is distributive with respect to the addition of scalars: for any matrix \\(\mathrm{X}\\) and for any scalars \\(\alpha\\) and \\(\beta\\), we have \\((\alpha+\beta)\mathrm{X} = \alpha\mathrm{X} + \beta\mathrm{Y}\\).


## Diagonal and Off-diagonal Entries of a Matrix
* Let \\(\mathrm{X}\\) be a square matrix. The diagonal (or main diagonal of \\(\mathrm{X}\\)) is the set of all entries \\(\mathrm{X}_{i,j}\\) such that \\(i=j\\). The entries on the main diagonal are called diagonal entries, and all the other entries are called off-diagonal entries. In the following \\(3\times 3\\) square matrix, \\(daig(\mathrm{X})= \\{1, 5, 9\\}\\), and off-diagonal entries are given by \\(\\{2, 3, 4, 6, 7, 8\\}\\).

\begin{equation}
\begin{aligned}
  \mathrm{X} =  
    \begin{bmatrix}
      1 & 2 & 2  \\\\\\\\
      4 & 5 & 6  \\\\\\\\
      7 & 8 & 9  
    \end{bmatrix}
\end{aligned}
\end{equation}

## Identity Matrix
* An identity matrix denoted by \\(\mathrm{I_n}\\) is a square matrix where its all diagonal entries are equal to \\(1\\) and all its off-diagonal entries are equal to 0.

\begin{equation}
\begin{aligned}
  \mathrm{I_n} =  
    \begin{bmatrix}
      1  & 0 & 0 & \ldots & 0  \\\\\\\\
      0  & 1 & 0 & \ldots & 0  \\\\\\\\
      \vdots & \vdots & \ddots & \vdots & \vdots \\\\\\\\
      0  & 0 & 0 & \ldots & 1
    \end{bmatrix}
  \end{aligned}
\end{equation}

## Transpose of Matrix:
* By exchanging the rows and columns of a matrix, we obtain the transpose of the matrix. Given a matrix \\(\mathrm{X} \in \mathbb{R}^{m\times n}\\),
its transpose written \\(\mathrm{X}^T \in \mathbb{R}^{n\times m}\\).
* Properties of transposition operation.
  1. Given two matrices of \\(\mathrm{X} \in \mathbb{R}^{m\times n}\\) and \\(\mathrm{Y} \in \mathbb{R}^{m\times n}\\), we have
     \\((\mathrm{X} + \mathrm{Y})^T = \mathrm{X}^T + \mathrm{Y}^T\\).
  2. Given two matrices of \\(\mathrm{X} \in \mathbb{R}^{m\times n}\\) and \\(\mathrm{Y} \in \mathbb{R}^{n\times p}\\), we have
     \\((\mathrm{X}\mathrm{Y})^T = \mathrm{Y}^T\mathrm{X}^T \in \mathbb{R}^{p\times m}\\).
  3. Given a matrix \\(\mathrm{X} \in \mathbb{R}^{m\times n}\\), we have
     \\((\mathrm{X}^T)^T = \mathrm{X}\\).
  4. Given a complex matrix \\(\mathrm{X} \in \mathbb{C}^{m\times n}\\), the transpose operation is defined by exchanging the rows and columns and conjugating them, and denoted by \\(\mathrm{X}^H\\) (conjugate transpose).
  5. If for a real (complex) matrix \\(\mathrm{X} \in \mathbb{R}^{m\times n}\\) (\\(\in \mathbb{C}^{m\times n}\\)), we have \\(\mathrm{X}=(\mathrm{X})^T\\) (\\(\mathrm{X}=(\mathrm{X})^H\\)), then the matrix is called _Symmetric_ (_Hermitian_).
  6. The set of all symmetric (Hermitian) matrices is denoted as \\(\mathbb{S}^n\\) (\\(\mathbb{H}^n\\)).

## Tensor:
* In simple terms, a tensor is just a generalization of a 2-d array to more than 2 dimensions. We can think of a vector as a 1-d tensor and a matrix as a 2-d tensor. The following picture shows a scalar as 0 rank tensor, a vector as rank 1, a matrix as rank 2, and a 3-d tensor.

   <p align="center">
    <img width="700" alt="Screenshot 2023-07-22 at 11 21 29 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/f297d14b-115c-4489-b701-f2e7c5e7b76d">
    <br>
      <em> Four types of tensors. </em>
    </p>

* One common operator on tensors (as we will see in our programming module) is flattening a matrix or multi-dimensional tensor to a lower-dimensional tensor. For example, we can flatten a matrix to a vector. This flattening can be done in _row-major_ order or _column-major_ order. In row-major order (used by languages such as Python and C++)), we _vetorize_ a matrix by arranging rows of the matrix back-to-back in a vector; while columns of the matrix are placed in a vector in column-major order (used by languages such as Julia, Matlab, R, and Fortran). We use \\(vec(\mathrm{X})\\) for vectorizing operation.


## Linear independence, Spans, and Basis

* A set of vectors \\(\\{\mathbf{x_1}, \mathbf{x_2}, \ldots, \mathbf{x_n}\\}\\) is said to be linearly independent if \\(\sum_{i=1}^{n} \alpha_i \mathbf{x_i} = 0\\), then \\(\alpha_1 = \alpha_2 = \dots = \alpha_n = 0\\). 




