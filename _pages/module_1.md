# Module 1 - Introduction:

* We start by introducing what machine learning is, and why we need to know about it. Next, we review the required mathematical background for learning Machine Learning (ML) and Deep Learning (DL).
* We cover the following fundamental topics:
1. Linear Algebra
2. Probability
3. Statistics
4. Optimization
   
* Next we do a crash course in python and Numpy.
## What is machine learning?
* With the deluge of data, we need to find ways to discover what is in the data. ML is a set of algorithms/methods that help us learn and recognize the hidden patterns in data. ML is not a new topic. In fact, learning from data has been explored and used by many disciplines such as Statistics, Signal Processing, Control Theory, etc. What makes ML special is to provide a common formalism to the problems and algorithms. With the help of ML techniques, one can predict future data, or perform other kinds of decision-making under uncertainty.
* There are different types of ML. Two common types of categorizing ML methods:
1. Supervised and unsupervised
2. Discriminative and Generative
### Supervised Learning
* In supervised methods, we are given a set of $N$ input-output pairs $\mathcal{D}=$ $`\{(x_i, y_i)\}`$$_{i=1}^N$, and the goal is to learn a map from inputs, $x_i$'s to outputs, $y_i$'s. Input variables have different names like **features**, **attributes**, or **covariates**. These input variables are typically a $p$-dimentional vector, denoting for example heights and weights of different persons (in this case, $p=2$. That is, $x_i$ is a 2-dimensional real vector corresponding to the $i^{th}$ person.   
However, input variables can be a very complex structured object, such as an image, s speech signal, a sentence, an email message, a time series, a graph, etc. On the other hand, output variables known as **response variable** can be anything, but most methods assume that $y_i$'s are categorical or nominal variables from some finite set, i.e., $y_i$ $\in$ $`\{1,2,\dots,C\}`$ in a classification problem, for example.
* Supervised problems come in two flavors:
  1. **Regression:** In regression problems, the output variables are continuous, i.e., $y_i \in \mathbb{R}$ or $y_i \in \mathbb{C}$ for $i=1,2,\dots, N$.
  2. **Classification:** In classification problems, the output variables are discrete, and they belong to a finite set (i.e., a set with a finite number of elements). That is, $y_i$ $\in$ $`\{1,2,\dots,C\}`$ for $i=1,2,\dots, N$.
* Some examples of supervised learning:
* Regression example: Face detection. The input, $x$ is an image, where $p$ is the number of pixels in the image. The output, $y_i$ is the location of faces in the figure (a real vale).
<img width="831" alt="Screenshot 2023-07-10 at 7 21 57 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/38d8dfc0-7825-49f7-9993-09db19733f41">
(a) Input image (Murphy family, photo taken 5 August 2010). (b) The output of the classifier, which detected 5 faces at different poses.
Classification example: Hand-written digit recognition. The input, $x$ is an image, where $p$ is the number of pixels in the image. The output, $y_i$ is one the numbers in the set $`{0,1,2,\dots,9\}`$ (a discrete value).

* In supervised methods, we
