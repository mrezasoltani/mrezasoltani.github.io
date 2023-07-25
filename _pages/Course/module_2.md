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
   | **The** \\(i^{th}\\) **element of standard/Canonical basis vector with only 1 in** \\(i^{th}\\) **position and 0s in other positions** | \\(\mathbf{e}_i=[0, 0, \ldots, 0, 1, 0, \dots, 0 ]^T\\)|
   | **Element \\(i\\) of vector** \\(\mathbf{x}\\) | \\(x_i\\) |
   | **All elements of vector** \\(\mathbf{x}\\) **except for element** \\(i\\)  | \\(\mathbf{x_{-i}}\\) |
   | **Entry in** \\(i^{th}\\) **row and** \\(j^{th}\\) **column of a matrix** \\(\mathrm{X}\\) | \\(x_{ij}\\) |
   | \\(i^{th}\\) **row (column) of a matrix** \\(\mathrm{X}\\) | \\(\mathrm{X_{i:}}\\) (\\(\mathrm{X_{:i}}\\)) |
   | **A probability distribution over a discrete variable (pmf)** | \\(P(x)\\) |
   | **A probability distribution over a continuous variable (pdf)** | \\(p(x)\\) |
   | \\(L^p\\) **norm of vector** \\(\mathbf{x}\\) | \\(\\|\\|\mathbf{x}\\|\\|_p\\) |
   | \\(p\\) **-induced norm of a matrix** | \\(\\|\\|\mathrm{X}\\|\\|_p\\) |
   | **Dot product of two vectors** | \\(\mathbf{x}\cdot\mathbf{y}\\) |
   | **Hadamard (element-wise) product of two vectors (matrices)** | \\(\mathbf{x}\odot\mathbf{y} (\mathrm{X}\odot\mathrm{Y})\\) |

* Here, we mostly present results for the real field. We often give the corresponding results for the complex field.

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

* **Definition.** Consider a set of \\(p\\) matrices (vectors) \\(\\{\mathrm{X_1}, \mathrm{X_1}, \dots, \mathrm{X_p}\\}\\) with dimension \\(\mathbb{R}^{m\times n}\\)  \\((\\{\mathbf{x_1}, \mathbf{x_2}, \ldots, \mathbf{x_p}\\}\\) with dimension \\(\mathbb{R}^{p})\\). A matrix (vector) \\(\mathrm{Y} \in \mathbb{R}^{m\times n}\\) \\((\mathbf{y} \in  \mathbb{R}^{n})\\) is a _linear combination_ of \\(\\{\mathrm{X_1}, \mathrm{X_1}, \dots, \mathrm{X_p}\\}\\) ( \\(\\{\mathbf{x_1}, \mathbf{x_2}, \ldots, \mathbf{x_p}\\}\\) ) if and only if there exist \\(p\\) scalars coefficients \\(\\{\alpha_1, \alpha_1, \ldots, \alpha_1\\}\\) such that \\(\mathrm{Y} = \sum_{i=1}^{p} \alpha_i \mathrm{X_i}\\) \\((\mathbf{y} = \sum_{i=1}^{p} \alpha_i \mathbf{x_i})\\).

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

## Addition, Subtraction, and Scaling Vectors

* Addition, subtraction, and scaling vectors have been illustrated in the following figure.

    <p align="center">
    <img width="842" alt="Screenshot 2023-07-22 at 11 54 22 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/746371b3-d4d8-49b1-baa4-b441865919b6">
    <br>
      <em> Addition, subtraction, and scaling of vectors.</em>
    </p>
* Alegbriaclly, given two vectors of \\(\mathbf{x}\\), \\(\mathbf{y} \in \mathbb{R}^{n} \\), their addition is given by \\(\mathbf{x} + \mathbf{y} = [x_1+y_1 ~~ x_2+y_2 ~~ \ldots ~~ x_n+y_n]^T \in \mathbb{R}^{n} \\). Also, \\(\alpha\mathbf{x} = [\alpha x_1 ~~ \alpha x_2 ~~\dots ~~\alpha x_n]^T \in \mathbb{R}^{n} \\) for some \\(\alpha \in \mathbb{R}\\).
* Addition, subtraction, and scaling of matrices are similar to the vectors (vector is a special matrix). Alegbriaclly, given two matrices of \\(\mathrm{X}\\), \\(\mathrm{Y} \in \mathbb{R}^{m\times n} \\) and for any scalar \\(\alpha \in \mathbb{R}\\):

\begin{equation}
\begin{aligned}
  \mathrm{X} \pm \mathrm{Y} =  
    \begin{bmatrix}
      x_{11}\pm x_{11} & x_{12}\pm y_{12} & \ldots & x_{1n}\pm y_{1n} \\\\\\\\
      x_{21}\pm y_{21} & x_{22}\pm y_{22} & \ldots & x_{2n}\pm y_{2n} \\\\\\\\
      \vdots & \vdots & \ddots & \vdots \\\\\\\\
      x_{m1}\pm y_{m1} & x_{m2}\pm y_{m2} & \ldots & x_{mn}\pm y_{mn}
    \end{bmatrix}, &
  \mathrm{\alpha X} = 
    \begin{bmatrix}
      \alpha x_{11} & \alpha x_{12} & \ldots & \alpha x_{1n} \\\\\\\\
      \alpha x_{21} & \alpha x_{22} & \ldots & \alpha x_{2n} \\\\\\\\
      \vdots & \vdots & \ddots & \vdots \\\\\\\\
      \alpha x_{m1} & \alpha x_{m2} & \ldots & \alpha x_{mn}
    \end{bmatrix}
\end{aligned}
\end{equation}

## Inner Product of Two Vectors
* For any two vectors \\(\mathbf{x} \in \mathbb{R}^{n}\\) and \\(\mathbf{y} \in \mathbb{R}^{n}\\), their (inner) product is a scaler given by \\(\mathbf{x}.\mathbf{y} = \sum_{i=1}^{n}x_iy_i\\). Geometrically, the inner product is interpreted as the projection of one vector onto another one as depicted in the following figure. Accordingly, we also have another (geometric formula) for the inner product of two vectors: \\(\mathbf{x}.\mathbf{y} = \\|\mathbf{x}\\|_2\\|\mathbf{y}\\|_2\cos(\theta)\\), where \\(\theta\\) is the angle between two vectors.

  <p align="center">
    <img width="400" alt="Screenshot 2023-07-22 at 11 54 22 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/ff61f836-7b63-43ff-b11b-ced6e7b0cd54">
    <br>
    <em>Inner product as the projection of one vector on the other one.</em>
  </p>

## Matrix Multiplication and its Properties
* For any two matrices \\(\mathrm{X} \in \mathbb{R}^{m\times n}\\) and \\(\mathrm{Y} \in \mathbb{R}^{n\times p}\\), their (inner) product, belonging \\(\mathbb{R}^{m\times p}\\) is defined as \\((\mathrm{X}\mathrm{Y}) _{ij} = \sum _{k=1}^{n} x _{ik} y _{kj}\\). That is, the entry in \\(i^{th}\\) row and \\(j^{th}\\) column of the product is obtained by the inner product of \\(i^{th}\\) row of matrix \\(\mathrm{X}\\) and \\(j^{th}\\) column of matrix \\(\mathrm{Y}\\).
    
  * Propoerties of Matrices
    * **Associativity.** Multiplication of a matrix by a scalar is associative: for any matrix \\(mathrm{X}\\) and for any scalars \\(alpha\\) and \\(\beta\\), we have \\(\alpha (\beta \mathrm{X}) = (\alpha \beta)\mathrm{X}\\).
    * **Distributivity w.r.t. addition.** Multiplication of a matrix by a scalar is distributive with respect to matrix addition: for any matrices \\(\mathrm{X}\\) and \\(\mathrm{Y}\\) and for any scalar \\(alpha\\), we have \\(\alpha (\mathrm{X}+\mathrm{Y}) = \alpha \mathrm{X} + \alpha \mathrm{Y}\\).
    * **Distributive property w.r.t. multiplication.** Multiplication of a matrix by a scalar is distributive with respect to the addition of scalars: for any matrix \\(\mathrm{X}\\) and for any scalars \\(\alpha\\) and \\(\beta\\), we have \\((\alpha+\beta)\mathrm{X} = \alpha\mathrm{X} + \beta\mathrm{X}\\).

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

## Transpose of a Matrix:
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

## Trace of a Square Matrix
* The trace of a square matrix \\(\mathrm{X} \in \mathbb{R}^{n\times n}\\) denoted by \\(Tr(\mathrm{X})\\) is the sum of its diagonal entries. That is, \\(Tr(\mathrm{X}) = \sum_{i=1}^{n} x_{ii}\\).
* Properties of the trace (in all these properties \\(\mathrm{X}, ~\mathrm{Y}, ~\mathrm{Z} \in \mathbb{R}^{n\times n}, ~\mathbf{u} \in \mathbb{R}^n\\), and \\(\alpha \in \mathbb{R}\\)):
  1. \\(Tr(\mathrm{X}^{T}) = Tr(\mathrm{X})\\)
  2. \\(Tr(\mathrm{X} + \mathrm{Y}) = Tr(\mathrm{X}) + Tr(\mathrm{Y})\\)
  3. \\(Tr(\mathrm{\alpha X}) = \alpha Tr(\mathrm{X})\\)
  4. \\(Tr(\mathrm{X}\mathrm{Y}) = Tr(\mathrm{Y}\mathrm{X})\\)
  5. **Cyclic Permutation Property.** \\(Tr(\mathrm{X}\mathrm{Y}\mathrm{Z}) = Tr(\mathrm{Y}\mathrm{Z}\mathrm{X}) = Tr(\mathrm{Z}\mathrm{X}\mathrm{Y})\\)
  6. **Trace Trick.** \\(\mathbf{u}^T\mathrm{X}\mathbf{u} = Tr(\mathbf{u}^T\mathrm{X}\mathbf{u}) = Tr(\mathrm{X}\mathbf{u}\mathbf{u}^T)\\)

## Determinant of a Square Matrix
* Informally, the determinant of a square matrix, denoted by \\(det(A)\\) or \\(\|A\|\\), is a measure of how much it changes a unit volume when viewed as a linear transformation.
* Properties of the determinant (in all these properties \\(\mathrm{X}, ~\mathrm{Y}, ~\mathrm{Z} \in \mathbb{R}^{n\times n}, ~\mathbf{u} \in \mathbb{R}^n\\), and \\(\alpha \in \mathbb{R}\\)):
  1. \\(\|\mathrm{X}\| = \|\mathrm{X}^T\|\\)
  2. \\(\|\alpha\mathrm{X}\| = \alpha^n\|\mathrm{X}^T\|\\)
  3. \\(\|\mathrm{X}\mathrm{Y}\| = \|\mathrm{X}\|\|\mathrm{Y}\|\\)
  4. \\(\|\mathrm{X}\| = 0\\) iff \\(\mathrm{X}\\) is not-invetible (singular).
  5. \\(\|\mathrm{X}^{-1}\| = \dfrac{1}{\|\mathrm{X}\|}\\) if \\(\mathrm{X}\\) is invertible (non-singular)
  6.  \\(\|\mathrm{X}\| = \prod_{i=1}^{n}x_{ii}\\), if \\(\mathrm{X}\\) is a diagonal matrix.

## Tensor:
* In simple terms, a tensor is just a generalization of a \\(2\\)-d array to more than 2 dimensions. We can think of a vector as a \\(1\\)-d tensor and a matrix as a \\(2\\)-d tensor. The following picture shows a scalar as 0 rank tensor, a vector as rank 1, a matrix as rank 2, and a 3-d tensor.

   <p align="center">
    <img width="700" alt="Screenshot 2023-07-22 at 11 21 29 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/f297d14b-115c-4489-b701-f2e7c5e7b76d">
    <br>
      <em> Four types of tensors. </em>
    </p>

* One common operator on tensors (as we will see in our programming module) is flattening a matrix or multi-dimensional tensor to a lower-dimensional tensor. For example, we can flatten a matrix to a vector. This flattening can be done in _row-major_ order or _column-major_ order. In row-major order (used by languages such as Python and C++)), we _vetorize_ a matrix by arranging rows of the matrix back-to-back in a vector; while columns of the matrix are placed in a vector in column-major order (used by languages such as Julia, Matlab, R, and Fortran). We use \\(vec(\mathrm{X})\\) for vectorizing operation.

## Linear spaces
* Informally, a given set \\(S\\) is a linear space (ana vector space) if its elements can be multiplied by scalars and added together, and the results of these operations belong to \\(S\\). In order to have a more formal definition, we need the following definition of _field_.
* **Definition.** Consider a set \\(F\\) together with two binary operations, the addition, denoted by \\(+\\), and the multiplication, denoted by \\(\cdot\\). The set \\(F\\) is said to be a field if and only if, for any \\(\alpha, ~\beta,  ~\gamma \in F\\), all the following properties hold:
  1. Associativity of addition: \\(\alpha + (\beta+\gamma) = (\alpha + \beta)+\gamma\\)
  2. Commutativity of addition: \\(\alpha+\beta = \beta+\alpha\\)
  3. Additive identity: there exist an element in \\(F\\) denoted by \\(0\\) such that \\(\alpha + 0 =\alpha\\)
  4. Additive inverse: \\(\forall \alpha \in F\\), there exist an element in \\(F\\) denoted by \\(-\alpha\\) such that \\(\alpha +(-\alpha) = 0\\)
  5. Associativity of multiplication: \\(\alpha \cdot (\beta\cdot\gamma) = (\alpha \cdot \beta)\cdot\gamma\\)
  6. Commutativity of multiplication: \\(\alpha\cdot\beta = \beta\cdot\alpha\\)
  7. Multiplicative identity: there exist an element in \\(F\\) denoted by \\(1\\) such that \\(\alpha \cdot 1 =\alpha\\)
  8. Multiplicative inverse: \\(\forall \alpha\neq 0 \in F\\), there exist an element in \\(F\\) denoted by \\(\alpha^{-1}\\) such that \\(\alpha\cdot\alpha^{-1} = 1\\)
  9. Distributive property: \\(\alpha \cdot(\beta+\gamma) = \alpha \cdot \beta + \alpha \cdot \gamma\\)
      
* **Definition.** Let \\(F\\) be a field and \\(S\\) be a set equipped with a vector addition defined on \\(S\times S\rightarrow S\\) denoted by \\(+\\), and a scalar multiplication another operation defined on \\(F\times S\rightarrow S\\) denoted by \\(\cdot\\). The set \\(S\\) is said to be a linear space (or vector space) over field \\(F\\) if and only if, for any \\(\mathbf{x}, ~\mathbf{y}, ~\mathbf{z} \in S\\) and any \\(\alpha, ~\beta \in F\\), the following properties hold:
  1. Associativity of vector addition: \\(\mathbf{x} + (\mathbf{y} + \mathbf{z}) = (\mathbf{x} + \mathbf{y}) + \mathbf{z}\\)
  2. Commutativity of vector addition: \\(\mathbf{x} + \mathbf{y} = \mathbf{y} + \mathbf{x}\\)
  3. Additive identity: there exists a vector \\(0\in S\\), such that \\(\mathbf{x}+0=\mathbf{x}\\)
  4. Additive inverse: \\(\forall \mathbf{x}\in S\\), there exists an element of \\(S\\), denoted by \\(-\mathbf{x}\\), such that \\(\mathbf{x}+(-\mathbf{x})=0\\)
  5. Compatibility of multiplications: \\(\mathbf{x} \cdot (\mathbf{y}\cdot \mathbf{z}) = (\mathbf{x}\cdot \mathbf{y}) \cdot \mathbf{z}\\)
  6. Multiplicative identity: Let \\(1 \in F\\) be the multiplicative identity in \\(F\\), then \\(1\cdot \mathbf{x}=\mathbf{x}\\)
  7. Distributive property w.r.t. _vector_ addition: \\(\mathbf{x} \cdot(\mathbf{y}+\mathbf{z}) = \mathbf{x} \cdot \mathbf{y} + \mathbf{x} \cdot \mathbf{z}\\)
  8. Distributive property w.r.t. _field_ addition: \\((\alpha +\beta)\cdot\mathbf{x} = \alpha\cdot\mathbf{x} + \beta\cdot\mathbf{x}\\)
  
* Given the above definition, the elements of a vector space \\(S\\) and its associated field \\(F\\) are called vectors and scalars, respectively.

* **Definition.** A (linear) subspace of a linear space \\(S\\) is a set which is subset of \\(S\\) and itself is a linear space.

## Linear Independence, Span, and Basis

* A set of vectors \\(\\{\mathbf{x_1}, \mathbf{x_2}, \ldots, \mathbf{x_n}\\}\\) is said to be linearly independent if \\(\sum_{i=1}^{n} \alpha_i \mathbf{x_i} = 0\\), then \\(\alpha_1 = \alpha_2 = \dots = \alpha_n = 0\\). This means no vectors in set \\(\\{\mathbf{x_1}, \mathbf{x_2}, \ldots, \mathbf{x_n}\\}\\) can be represented as a linear combination of the remaining vectors. Conversely, a vector representing a linear combination of the remaining vectors is said to be linearly dependent.
* The span of a set of vectors  \\(\\{\mathbf{x_1}, \mathbf{x_2}, \ldots, \mathbf{x_n}\\}\\) is the set of all vectors that can be expressed as a linear combination of  \\(\\{\mathbf{x_1}, \mathbf{x_2}, \ldots, \mathbf{x_n}\\}\\). That is,
\\[span(\\{\mathbf{x_1}, \mathbf{x_2}, \ldots, \mathbf{x_n}\\}) = \\{\sum_{i=1}^{n} \alpha_i \mathbf{x_i} | \alpha_i \in \mathbb{R}\\}\\]
* Let \\(S\\) be a linear space. Consider a set of \\(n\\) linearly independent vectors \\(\\{\mathbf{x_1}, \mathbf{x_2}, \ldots, \mathbf{x_n}\\}\\). Then, this set is a **_basis_** for \\(S\\) if and only if, for any \\(\mathbf{x} \in S\\), there exist \\(n\\) scalars \\(\alpha_1,\alpha_2, \dots, \alpha_n\\) such that \\(\mathbf{x} = \sum_{i=1}^{n} \alpha_i \mathbf{x_i}\\). It is sometime said that the basis spans the set \\(S\\). In this case, \\(span(\\{\mathbf{x_1}, \mathbf{x_2}, \ldots, \mathbf{x_n}\\}) = \mathbb{R}^n\\).
* Every finite-dimensional linear space has a basis. This can be seen from the _Steinitz exchange lemma_.
* **Proposition.** Let \\(U=\\{\mathbf{x_1}, \mathbf{x_2}, \ldots, \mathbf{x_k}\\}\\) be a set of \\(k\\) linearly independent vectors belonging to a linear space \\(S\\). Also, consider a finite set of vectors \\(V=\\{\mathbf{x_1}, \mathbf{x_2}, \ldots, \mathbf{x_p}\\}\\) that span \\(S\\). If these \\(k\\) linearly independent vectors do not form a basis for \\(S\\), then one can form a basis by adding some elements of \\(V\\) to set \\(U\\).
* A linear space is a finite-dimensional space if it has a finite spanning set.
* **Proposition.** Let \\(S\\) be a finite-dimensional linear space. Then,  \\(S\\) possesses at least one basis.
* **Dimension theorem.** Let \\(S\\) be a finite-dimensional linear space. Consider two set of vectors given by \\(\\{\mathbf{x_1}, \mathbf{x_2}, \ldots, \mathbf{x_p}\\}\\) and \\(\\{\mathbf{x_1}, \mathbf{x_2}, \ldots, \mathbf{x_k}\\}\\). Then, \\(p=k\\).
* **Definition.** Let \\(S\\) be a finite-dimensional linear space, and \\(n\\) be the number of elements of any its bases (\\(n\\) is called cardinality of that bases). Then, \\(n\\) is called the dimension of \\(S\\).
* **Definition.** Let \\(S\\) be the space of all \\(K\\)-dimensional vectors. Then, the set of \\(K\\) vectors given by \\(\\{\mathbf{e_1}, \mathbf{e_2}, \ldots, \mathbf{e_K}\\}\\) is called the standard/canonical basis of \\(S\\) (recall our notion for the canonical basis vector)
* **Proposition.** The standard basis is a basis of the space of all \\(K\\)-dimensional vectors.
* The rows and columns of \\(\mathrm{I}_K\\), a \\(K\times K\\) identity matrix are standard basis of the space of all \\(K\\)-dimensional vectors.

## Four Fundamental Spaces of a Matrix
1. **Column Space** It is also called _range_ of matrix \\(\mathrm{X} \in \mathbb{R}^{m\times n}\\) and it is defined as the span of the columns of \\(\mathrm{X}\\):
\\[\mathcal{R}(\mathrm{X}) = \\{\mathrm{X}u \in \mathbb{R}^m \| u \in\mathbb{R}^n \\}\\]
2. **Null Space** It is also called _kernel_ of matrix \\(\mathrm{X} \in \mathbb{R}^{m\times n}\\) and it is defined as follows:
\\[\mathcal{N}(\mathrm{X}) = \\{u \in \mathbb{R}^n \| \mathrm{X}u=0 \\}\\]
3. **Row space** The third fundamental subspace is the range of matrix \\(\mathrm{X}^T\\) defined as follows:
\\[\mathcal{R}(\mathrm{X}^T) = \\{\mathrm{X}^Tu \in \mathbb{R}^n \| u \in\mathbb{R}^m \\}\\]
4. **Left Null Space** The fourth fundamental space is the kernel of of matrix \\(\mathrm{X}^T\\) defined as follows:
\\[\mathcal{N}(\mathrm{X}^T) = \\{u \in \mathbb{R}^m \| \mathrm{X}^Tu=0 \\}\\]

## Rank of a Matrix:
* The column rank of a matrix \\(\mathrm{X}\\) is the dimension of its column space (\\(\mathcal{R}(\mathrm{X})\\) (i.e., the space spanned by its columns), and the row rank is the dimension of its row space (\\(\mathcal{R}(\mathrm{X}^T)\\) (i.e., the space spanned by its rows). It can be shown that for any matrix \\(\mathrm{X}\\), **_column rank equals row rank_**. This quantity is referred to as the **_rank of_** \\(\mathrm{X}\\), denoted as \\(rank(\mathrm{X})\\).
  Properties of rank:
    Consider matrices \\(\mathrm{X}, ~ \mathrm{Y} \in \mathbb{R}^{m\times n}\\).
     1. \\(rank(\mathrm{X}) \leq \min(m, n)\\). If \\(rank(\mathrm{X}) = \min(m, n)\\), then \\(\mathrm{X}\\) is called full rank. Otherwise it is called rank deficient.
     2. \\(rank(\mathrm{X}) = rank(\mathrm{X}^T) = rank(\mathrm{X}^T\mathrm{X}) = rank(\mathrm{X}\mathrm{X}^T)\\).
     3. \\(rank(\mathrm{X}\mathrm{Y}) \leq  \min(rank(\mathrm{X}), rank(\mathrm{Y}))\\)
     4. \\(rank(\mathrm{X} + \mathrm{Y}) \leq  rank(\mathrm{X}) + rank(\mathrm{Y}) \\).

## Norm
* Let \\(\mathbb{V}\\) be a vector space. Informally, the norm is a measure of the _length_ of the vector. To be more specific, function \\(f : \mathbb{V} \rightarrow \mathbb{R}\\) is called a norm on \\(\mathbb{V}\\) if it satisfies the following 3 properties:
  1. **Positivity.** \\(f(\mathbf{x}) \geq 0\\, ~ f(\mathbf{x}) = 0\\) iff \\(\mathbf{x}=0\\)
  2. **Homogeneity.** \\(f(\alpha\mathbf{x}) = \|\alpha\|f(\mathbf{x}), ~ \forall \alpha \in \mathbb{R}\\)
  3. **Triangle Inequality.** \\(f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{x}) \\) 
* Norm function is denotd by \\(\\|.\\|\\) notation.

## Inverse of a Matrix:
* Let \\(\mathrm{X}\\) be a \\(n \times	n\\) matrix. The inverse of \\(\mathrm{X}\\), if it exists, is another \\(n \times	n\\) matrix such that \\(\mathrm{X}^{-1}\mathrm{X} = \mathrm{X}\mathrm{X}^{-1} = \mathrm{I_n}\\). If \\(\mathrm{X}^{-1}\\) exists, then matrix \\(\mathrm{X}\\) is called _invertible_.
* **Proposition.** If the inverse of a \\(n \times	n\\) matrix exists, then it is unique.
* **Proposition.** Let \\(\mathrm{X} \in \mathbb{R}^{n\times	n}\\) be a matrix. Then \\(\mathrm{X}\\) is an invertible matrix if and only if it is full-rank.
* Properties:
  1. \\((\mathrm{X}^{-1}) ^-1 = \mathrm{X}\\)
  2. \\((\mathrm{X}\mathrm{Y}) ^{-1}) = \mathrm{Y}^{-1}\mathrm{X}^{-1}\\)
  3. \\((\mathrm{X}^{T}) ^{-1} = (\mathrm{X}^{-T})\\)
  4. Let \\(\mathrm{X}\\) be a \\(2\times 2\\) invertible matrix, then
  \begin{equation}
  \begin{aligned}
    \mathrm{X} ^{-1} =  \frac{1}{det(\mathrm{X})}
     \begin{bmatrix}
        x_{22} & -x_{12}  \\\\\\\\
        -x_{21} & x_{11} 
      \end{bmatrix}
  \end{aligned}
  \end{equation}
  6. For a block diagonal matrix, the inverse is obtained by inverting each block separately:
  \begin{equation}
  \begin{aligned}
    \mathrm{X} ^{-1} =  
      \begin{bmatrix}
        \mathrm{A} & \mathrm{0}  \\\\\\\\
        \mathrm{0} & \mathrm{B} 
      \end{bmatrix} =
       \begin{bmatrix}
        \mathrm{A} ^{-1} & \mathrm{0}  \\\\\\\\
        \mathrm{0} & \mathrm{B} ^{-1} 
      \end{bmatrix}
  \end{aligned}
  \end{equation}

## **The (Moore-Penrose) Pseudo-Inverse** 
* In some case, we can generalize the concept of inverse matrix to the rectangular matrices. In particular, _Pseudo-Inverse_ of a matrix \\(\mathrm{X}\\) denoted by \\(\mathrm{X}^{\dagger}\\) is defined as the unique matrix that satisfies the following 4 properties:
  1. \\[\mathrm{X}\mathrm{X}^{\dagger}\mathrm{X} = \mathrm{X}\\]
  2. \\[\mathrm{X}^{\dagger}\mathrm{X}\mathrm{X}^{\dagger} =  \mathrm{X}^{\dagger}\\]
  3. \\[ (\mathrm{X}\mathrm{X}^{\dagger})^T = \mathrm{X}\mathrm{X} ^{\dagger} \\]
  4. \\[ (\mathrm{X}^{\dagger}\mathrm{X})^T = \mathrm{X} ^{\dagger}\mathrm{X}\\]
* If \\(\mathrm{X}\\) is a square \\(n\times n\\) full-rank matrix, then Pseudo-Inverse is given by standard inverse of a matrix, i.e., \\(\mathrm{X}^{\dagger} = \mathrm{X}^{-1}\\).
* If \\(\mathrm{X}\\) is a \\(m\times n\\) full-column rank matrix where \\(m > n\\) (a thin/tall/skinny matrix), then the Pseudo-Inverse is given by, left inverse \\(\mathrm{X}^{\dagger} = (\mathrm{X}^T\mathrm{X})^{-1}\mathrm{X}^T\\).
* If \\(\mathrm{X}\\) is a \\(m\times n\\) full-row rank matrix where \\(m < n\\) (a fat/short matrix), then the Pseudo-Inverse is given by right inverse, \\(\mathrm{X}^{\dagger} = \mathrm{X}^T(\mathrm{X}\mathrm{X}^T)^{-1}\\).

### Vector Norm
* Consider a vector \\(\mathbf{x} \in \mathbb{R}^n\\). We have the following different norm functions.
  
1. \\(p\\)-norm (\\(\ell_p\\)-norm): \\(\\|\mathbf{x}\\|_p = (\sum _{i=1}^n \|x_i\|^p)^{\dfrac{1}{p}} \\), \\(\forall p\geq 1\\)
2. \\(2\\)-norm (\\(\ell_2\\)-norm): \\(\\|\mathbf{x}\\|_2 = \sqrt{\sum _{i=1}^n \|x_i\|^2} = \sqrt{\mathbf{x}^T\mathbf{x}}\\)
3. \\(1\\)-norm (\\(\ell_1\\)-norm): \\(\\|\mathbf{x}\\|_1 = \sum _{i=1}^n \|x_i\|\\)
4. \\(\infty\\)-norm (\\(\ell_{\infty}\\)-norm): \\(\\|\mathbf{x}\\|_{\infty} = \max _{i\geq 1}\|x_i\|\\)
5. \\(0\\)-norm (\\(\ell_{0}\\)-norm): \\(\sum _{i=1}^n \mathbb{1} _{x_i\neq 0}\\)
   This is a pseudo norm since it does not satisfy the homogeneity property. It counts the number of non-zero elements in \\(\mathbf{x}\\).
* Here, \\(\mathbb{1} _{x \in \mathbb{A}}: \mathbb{A}\rightarrow \\{0,1\\}\\) denotes the indicator function defined as follows:

\begin{equation}
\begin{aligned}
  \mathbb{1} _{x \in \mathbb{A}} =
    \begin{cases}
      1, & x\in \mathbb{A}  \\\\\\\\
      & \\\\\\\\
      0, & x\notin \mathbb{A} 
    \end{cases}
\end{aligned}
\end{equation}

### Matrix Norm
* We can think of a matrix as a vector; hence, defining the matrix norm in terms of a vector norm. \\(\\|\mathrm{X}\\| = \\|vec(A)\\|\\). If the vector norm is the 2-norm, the corresponding matrix norm is called the _Frobenius_ norm:
  \\[ \\|\mathrm{X}\\|_F = \sqrt{\sum _{i=1}^m\sum _{j=1}^{n}  x _{ij}^2} = \sqrt{Tr(\mathrm{X}^T\mathrm{X})} = \\|vec(\mathrm{X})\\|_2 \\]
* We can also define norms for matrices. In particular, we can define the _induced norm_ of matrix \\(\mathrm{X}\\) as the maximum amount by which any unit-norm input can be lengthened:
\\[ \\|\mathrm{X}\\|_p = \max _{\mathbf{u}\neq}\dfrac{\\|\mathrm{X}\mathbf{u}\\|_p}{\\|\mathbf{u}\\|_p} = \max _{\\|\mathbf{u}\\|_p=1}\\|\mathrm{X}\mathbf{u}\\|_p \\]
* Consider a matrix \\(\mathrm{X} \in \mathbb{R}^{m\times n}\\) with rank \\(r\\). We can define _Schatten_ \\(p\\)-norm as follows:
\\[ \\|\mathrm{X}\\|_p = (\sum _{i=1}^r \sigma _i^p(\mathrm{X}))^{\dfrac{1}{p}} \\]
* We have two important cases:
  1. Spectral norm (induced \\(2\\)-norm). If \\(p=2\\), then we have:
     \\[\\|\mathrm{X}\\|_2 = \sqrt{\lambda _{max}(\mathrm{X}^T\mathrm{X})} = \max _i \sigma_i \\]
  2. Nuclear (Trace) norm (induced \\(1\\)-norm). If \\(p=1\\), then we have:
     \\[ \\|\mathrm{X}\\|_* = Tr(\sqrt{\mathrm{X}^T\mathrm{X}}) = \sum _{i=1}^r \sigma_i = \\|\mathbf{\sigma}\\|_1\\]

## Eigenvalues and Eigenvectors of a Matrix
* Given a square matrix \\(\mathrm{X} \in \mathbb{R}^{n\times n}\\), \\(\lambda\\) is called an eigenvalue of \\(\mathrm{X}\\) and \\(\mathbf{u} \in \mathbb{R}^n\\) is called the corresponding eigenvector if \\(\mathbf{X}\mathbf{u} = \lambda\mathbf{u}, ~ \mathbf{u}\neq 0\\).
  The above equation means that eigenvector of a matrix is the direction if matrix \\(\mathrm{X}\\) is multiplied by the vector \\(\mathbf{u}\\) results in a new vector that with the same direction as \\(\mathbf{u}\\) and scaled by the eigenvalue, \\(\lambda\\).
* From the relation of \\(\alpha\mathbf{X}\mathbf{u} = \mathbf{X}(\alpha\mathbf{u}) = \lambda(\alpha\mathbf{u}) \\), we see that if \\(\mathbf{u}\\) is the eigenvector corresponding to the eigenvalue \\(\lambda\\), then \\(\alpha\mathbf{u}\\) is also another eigenvector corresponding to the eigenvalue \\(\lambda\\). As a result, we have eigenvectors corresponding to a eigenvalue. This motivates us to talk about the normalized eigenvector, where \\(\\|\mathbf{u}\\|_2=1\\). However, this does not completely remove the non-uniqueness of the eigenvectors since both \\(\mathbf{u}\\) and \\(\mathbf{u}\\) have unit norm, but they are pointing to the opposite direction.

### Characteristic Equation of a Matrix
* Consider a matrix \\(\mathbf{X} \in \mathbb{R}^{n\times n}\\) with rank \\(r\\). If we re-write the eigenvector-eigenvalue equation as \\(\mathbf{X}\mathbf{u} -\lambda\mathbf{u} = 0, ~ \mathbf{u}\neq 0\\), it follows that in order to have a non-zero (non-trivial) solution, we need that the matrix \\(\mathbf{X}\mathbf{u} - \lambda\mathbf{u}\\\) to be non-singular. That is, \\(det(\mathbf{X} - \lambda\mathrm{I_n})=0\\).
* This resulting equation from \\(det(\mathbf{X} - \lambda\mathrm{I_n})=0\\) is a polynoimia equation in \\(\lambda\\) and is called the _characteristic equation_ of \\(\mathbf{X}\\). From algebra, we know that this polynomial equation has \\(n\\) possibly complex-valued solutions, which are eigenvalues of \\(\mathbf{X}\\), denoted by \\(\lambda_i\\)'s, and \\(\mathbf{u}_i\\)'s are the corresponding eigenvectors.
* Typically, all eigenvectors are sorted in order of their eigenvalues, with the largest magnitude ones first.
  Properties:
    1. The rank of matrix \\(\mathbf{X}\\) is equal to the number of non-zero eigenvalues of \\(\mathbf{X}\\).
    2. The trace of a matrix is equal to the sum of its eigenvalues, \\(Tr(\mathrm{X}) = \sum _{i=1}^r \lambda_i\\)
    3. The determinant of matrix \\(\mathbf{X}\\) is equal to the product of its eigenvalues, \\(det(\mathrm{X}) = \prod _{i=1}^r \lambda_i\\)

### Diagonalization
* If we stack all eigenvalues and eigenvectors of a matrix \\(\mathrm{X}\\), in the matrix notation, we can write:
\\[\mathrm{X} \mathrm{U} = \mathrm{U}\mathrm{\Lambda}\\]
  Here, we group all eigenvectors of \\(\mathrm{X}\\) as the columns of matrix \\(\mathrm{U}\\), and corresponding eigenvalues in a diagonal matrix \\(\mathrm{\Lambda}\\), that is
  \begin{equation}
\begin{aligned}
  \mathrm{U} =  
    \begin{bmatrix}
      | & | & \ldots & | \\\\\\\\
      \mathrm{U_1} & \mathrm{U_2} & \ldots & \mathrm{U_n} \\\\\\\\
      | & | & \ldots & |
    \end{bmatrix},
  \mathrm{\Lambda} = 
    \begin{bmatrix}
        \lambda_1 & 0 & \ldots & 0 \\\\\\\\
        0 & \lambda_2 & \ldots & 0 \\\\\\\\
        \vdots & \vdots & \ddots & \vdots \\\\\\\\
        0 & 0 & \ldots & \lambda_n
    \end{bmatrix}
\end{aligned}
\end{equation}
* From above, we immidiately see that if matrix \\(\mathrm{U}\\) to be invertible, or eigenvectors, \\(\mathbf{u_i}\\)'s to be linearly independent, then we can write \\(\mathrm{X}\\) as follows:
\\[\mathrm{X} = \mathrm{U}\mathrm{\Lambda}\mathrm{U}^{-1}\\]
* A matrix that can be written in the above form is called _diagonalizable_.

## Eigen Value Decomposition (EVD)
* Let the matrix \\(\mathrm{X}\\) be a diagonalizable matrix, then we can decompose matrix \\(\mathrm{X} = \mathrm{U}\mathrm{\Lambda}\mathrm{U}^{T}\\).
  Here, we have used the fact that matrix \\(mathrm{U}\\ is a Orthogonal (Hermitian) matrix, i.e., \\(\\(mathrm{U}^T = \\(mathrm{U}^{-1} \\)

\begin{equation}
\begin{aligned}
  \mathrm{X} = \mathrm{U}\mathrm{\Lambda}\mathrm{U}^{T}=
    \begin{bmatrix}
      | & | & \ldots & | \\\\\\\\
      \mathrm{U_1} & \mathrm{U_2} & \ldots & \mathrm{U_n} \\\\\\\\
      | & | & \ldots & |
    \end{bmatrix}
    \begin{bmatrix}
        \lambda_1 & 0 & \ldots & 0 \\\\\\\\
        0 & \lambda_2 & \ldots & 0 \\\\\\\\
        \vdots & \vdots & \ddots & \vdots \\\\\\\\
        0 & 0 & \ldots & \lambda_n
    \end{bmatrix}
    \begin{bmatrix}
      ---  & \mathrm{U_1}^T & ---  \\\\\\\\
      ---  & \mathrm{U_2}^T & ---  \\\\\\\\
       & \vdots & \\\\\\\\
      ---  & \mathrm{U_n}^T & --- 
    \end{bmatrix} =
\end{aligned}
\end{equation}
