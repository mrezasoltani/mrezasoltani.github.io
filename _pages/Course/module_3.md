---
title: "Module 3 -- Probability"
classes: wide
---
## What Is Probability?
* Probability is a critical tool for modern data analysis. It arises in dealing with uncertainty, randomized algorithms, and Bayesian analysis. Generally, we can interpret probability in two ways; the **_Fequentist_** interpretation and **_Bayesian_** interpretation.
* In the frequentist view, probabilities represent long-run frequencies of events that can happen multiple times. For example, when we say a fair coin has a \\(50\\%\\) chance of turning tail, we mean that if we toss the coin many times, we expect that it lands tail half of the time.
* In Bayesian interpretation, probability is used to quantify our ignorance or uncertainty about something. This viewpoint is more related to information rather than repeated trials. In the above flipping a fair coin, Bayesian interpretation states our belief that the coin is equally likely to land heads or tails on the next toss.
* Bayesian interpretation can be used to model our uncertainty about events that do not have long-term frequencies [K. Murthy, 2022].

### Two Types of Uncertainty
  1. **Epistemic (Model) Uncertainty.**: This uncertainty is due to our ignorance and uncertainty about the mechanism of generating the data.
  2. **Aleatoric (Data) Uncertainty.** The uncertainty is due to the intrinsic variability in the data and cannot be reduced even more collection of data. This is derived from the Latin word for “dice”.

## Informal Review of Some Concepts
Here, we review the most important concepts in probability theory without mathematical rigor. Later, we make these concepts more formal by defining them using  the language of Measure Theory.
* **Sample Space.** Set of all possible outcomes in a random experiment is called _sample space_ and denoted by \\(\Omega\\). For example, in rolling a die, there are \\(6\\) possible outcomes, so \\(\Omega = \\{1,2,3,4,5,6\\}\\). An outcome is an element of a sample space, e.g., \\(\omega = 3\\). This example shows a discrete sample space. A sample space can also be a continuous space. For example, the waiting time for arriving at a bus is a random experiment. An outcome of this random experiment is any non-negative real number. 
* **Event.** A subset of the sample space is an _event_, i.e., \\(A\subseteq \Omega\\). For example, in the rolling of a die, an event can be defined as _facing an odd number_, i.e., \\(A=\\{1,3,5\\}\\). For a sample space with \\(n\\) outcomes, we can have \\(2^n\\) events.
* **Probability.** Informally, a probability is a measure from the set of events to the real numbers in \\(\[0,1\]\\) such that (axioms of probability, Kolmogorov axioms):
  1. For any event \\(A\subseteq \Omega\\), \\(~ 0 \leq P(A) \leq 1\\)
  2. \\(p(\Omega) = 1\\)
  3. For any sequence \\(A_1, A_2, \ldots\\), such that \\(\forall i,j, ~ A_i \cap A_j = 0 \\) (i.e, pairwise disjoint sets):
     \\[P(\cup _{i=1}^{\infty} A_i) = \sum _{i=1}^{\infty} P(A_i)\\]
     - The disjoin sets assumption in the third property means that events \\(A\\) and \\(B\\) cannot happen at the same time, that is, they are mutually exclusive. For example, in flipping a coin, the event of facing with a tail does not have any intersection with the event that the coin lands the head.
     - **Probability of union of two events (addition rule).** The probability of event A or B happening is given by \\(P(A\cup B) = P(A) + P(B) - P(A\cap B)\\).
* **Joint Probability.** A joint probability of happening two events \\(A\\) and \\(B\\) is defined as \\(P(A, B) = P(A\cap B)\\).
* **Independency.** If two events \\(A\\) and \\(B\\) are independent, then their join probability is given by their product, i.e., \\(P(A\cap B) = P(A)\cdot P(B)\\). This means that occurring the event \\(A\\) does not have any effect on the happening or not happening of the event \\(B\\). For example, consider a random experiment defined as follows: choosing two numbers uniformly at random from a set \\(X\\{1,2,3,4\\}\\). Let \\(A\\) be an event that the first number belongs to the set \\(\\{3\\}\\), and \\(B\\) be an event that the second number belongs to the set \\(\\{1, 2, 4\\}\\), then these two events are independent of each other; hence, \\(P(A\cap B) = P(A)\cdot P(B) = \frac{1}{4}\cdot\frac{2}{4} = \frac{1}{8}\\).
  - It is very important to note that in independency of two events does not imply their mutually exclusive and vice versa.
  - **Mutual Independence.** Let \\(A_{1},A_2,\dots,A_{n}\\) be \\(n\\) events. \\(A_{1},A_2,\dots,A_{n}\\) are mutually independent (jointly independent ) if and only if for any sub-collection of \\(k\\) events (\\(k\leq n\\)) \\(A_{1},A_2,\dots,A_{k}\\), \\(P(\cap _{i=1}^{k} A_i) = \prod _{i=1}^{k} P(A_i)\\).
  - **Pairwise Independence.** For \\(n \geq 3\\), events \\(A_{1},A_2,\dots,A_{n}\\) are pairwise independent if \\(A_i \cap A_j = 0, ~\forall 1\leq i,j\leq n\\).
    - Please note that Pairwise Independence does not imply Mutual Independence.
* **Conditional Probability.** The conditional probability of happening an event \\(A\\) given that another event \\(B\\) has occurred is given by \\(P(A\|B) = \frac{P(A, B)}{P(B)}, ~ P(B)\neq 0\\).
  - **Product Rule.** From the above relation, we can wrtie \\(p(A, B) = P(A\|B)\cdot P(B)\\). This is called the product rule.
* **Conditional Independency.** If two events \\(A\\) and \\(B\\) are independent given another event \\(C\\), we say that \\(A\\) and \\(B\\) are _conditionally independent_ given \\(C\\), this is denoted by \\(A \perp B \| C \\). In this case, \\(P(A, B\|C) = P(A\|C)\cdot P(B\|C)\\). Many events are dependent on each other, but it is possible when relevant information is known, the two events become independent.
* **Law of Total Probability.** Let \\(A_{1},A_2,\dots,A_{n}\\) be a partition of sample space \\(\Omega\\). That is, \\(\cup _{i=1}^n A_i = \Omega, ~ A_i\cap A_j =0, ~ \forall ~ 0\leq i,j\leq n\\). Then for any set \\(B\subseteq\Omega\\), we have:
  \\[P(B) = \sum _{i=1}^{n} P(B\|A_i)\cdot P(A_i)\\]

  The relation between sets \\(\Omega\\), \\(B\\), \\(A_i\\)'s in the Law of Total Probability is illustrated in the following figure:
    <p align="center">
            <img width="500" alt="Screenshot 2023-07-30 at 7 21 57 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/7738d6f3-301a-4d38-acdd-7d4bf15b0147">
    <br>
            <em>A Venn diagram to show the Law of Total Probability.</em>
    </p>
    
  <details>
  <summary>Proof</summary>
    
      We have: 
      \begin{equation}
        B = B\cap\Omega = B\cap(\cup _{i=1}^n A_i ) = \cup _{i=1}^n(B\cap A_i)
      \end{equation}
  
      We note that \(B\cap A_i, ~i=1,2,\ldots,n\) are disjoint events by the assumption on \(A_1, A_2,\ldots, A_{n}\) being a partition of the sample space. Hence, applying the third axiom of probability, we have:
  
      \begin{equation}
      P(B\cap A_i)= \sum _{i=1}^{n} P(B\cap A_i) = \sum _{i=1}^{n} P(B|A_i)\cdot P(A_i)
      \end{equation}
  
      For the last equality, we have used the conditional probability rule. \(\blacksquare\)
   
  </details>
  
* **Bayes Rule (Bayes’ Theorem).** Let \\(A\\) and \\(B\\) be two events with positive probabilities, i.e., \\(P(A)>0\\) and \\(P(B) >0\\). From the definition of conditional probability, we can derive Bayes’ rule:
  \\[P(A\|B) = \frac{P(B\|A)\cdot P(A)}{P(B)}\\]
  
* **Random Variables.** A random variable (r.v.) \\(X: \Omega\rightarrow \mathbb{R}\\) is a function from the sample space to the real line.
  - **Discrete random variables.** If the sample space \\(\Omega\\) over which a random variable \\(X\\) is defined is finite or countably infinite, then \\(X\\) is called a discrete random variable. Any possible realization of \\X\\) is an event, hence, the probability of the event that \\(X\\) has value \\(x\\) is given by \\(P(X=x)\\) (Most of the time for simplicity of notation, we write \\(P(x)\\)).
    - \\(P(x)\\) is called **probability mass function (pmf)** and satisfies the following conditions:
     - \\(0 \leq P(x) \leq 1\\)
     - \\(\sum _{x\in \Omega}P(x) = 1\\)
    - **Cumulative distribution function (cdf).** We define a cdf of a r.v. as \\(F(x) = Pr(x) = Pr_X(x) =  Pr(X\leq x) = \sum _{\alpha \in \Omega, \alpha\leq x} P(\alpha)\\). From this definition, we immediately see that \\(Pr(\alpha \leq X \leq \beta) = Pr(\beta) - Pr(\alpha)\\).
    - The cdf is a monotonically increasing (not necessarily  strictly) function and is always continuous **_only_** from the right.
    - Assume that a r.v. \\(X\\) can have \\(K\\) values as \\(\\{1,2,\dots,K\\}\\). Then the Bayes rule is given by:
      \\[P(X=i\|B) = \frac{P(B\|X=i)P(X=i)}{P(B)} = \frac{P(B\|X=i)P(X=i)}{\sum _{k=1}^{K} P(B\|X=k)P(X=k)}\\]
    - \\(P(X=i\|B)\\) is called the _**Posterior Probability**_. \\(P(B\|X=i)\\) is called the _**Likelihood**_,\\(P(X=i)\\) is the _**Prior Probability**_, and \\(P(B)\\) is the _**Normalization Constant, Evidence, or Marginal Likelihood**_.
     - The following figures show the pmf and cdf of a  categorical random variable with the following pmf:
   
       \begin{equation}
        \begin{aligned}
          P(x) =
            \begin{cases}
              ~ 0.23, & x=1  \\\\\\\\
              0.23, & x=2  \\\\\\\\
              0.25, & x=3  \\\\\\\\
              0.29, & x=4  \\\\\\\\
              0.0, & o.w
            \end{cases}
        \end{aligned}
        \end{equation}
    
        <p align="center">
            <img width="1000" alt="Screenshot 2023-07-30 at 7 21 57 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/ff9d69f0-70fa-4895-be7d-d3cab0fd4e61">

        <br>
          <em>PMF and CDF of a categorical random variable.</em>
        </p>

  - **Continuous random variables.** If the sample space \\(\Omega\\) over which a random variable \\(X\\) is defined is infinite (e.g., \\(\mathbb{R}\\)), then \\(X\\) is called a Continuous random variable. Here, we cannot consider a set of finite/countable values for \\(X\\). However, we can choose a set of intervals in \\(\mathbb{R}\\), for example, and define the probability of \\(X\\) taking on a specific value by an infinitesimal interval containing that value.
  
    - **Probability density function (pdf)** For the a continuous r.v., we can define the pdf as \\(p(x)=\frac{dPr(x)}{dx}\\), assuming this derivative exists. 
    - Similar to the discrete case, the cdf function for a continuous r.v. is defined by \\(F_X(x) = Pr(x) = Pr_X(x) =  Pr(X\leq x) = \int _{u \in \Omega, u\leq x} p(u)du\\), where \\(p(u)\\) is the probability density function (pdf). If \\(\Omega = \mathbb{R}\\), then \\(Pr(x) = \int _{-\infty}^{x}p(u)du\\). From this, we can compute the probability of a continuous variable being in a finite interval \\(\[\alpha, \beta\]\\) for some \\(\alpha,\beta \in \mathbb{R}\\) as follows:
      \\[Pr(\alpha \leq X \leq \beta) = Pr(\beta) - Pr(\alpha) = \int _{\alpha}^{\beta}p(u)du \\]
    - If the above interval is an infinitesimal interval, from calculus, \\(Pr(x \leq X \leq x+\Delta x) \approx p(x)dx\\). So, the product of the density at \\(x\\) and the width of the interval gives the probability of \\(X\\) being in a small interval around \\(x\\).
    - The cdf is a monotonically increasing (not necessarily  strictly) function and for a continuous r.v. is always a continuous function.
    - The Bayes rule is given by
      \\[p(X=x\|B) = \frac{p(B\|X=x)p(X=x)}{p(B)} = \frac{p(B\|X=x)p(X=x)}{\int _{x\in \Omega} p(B\|X=x)p(X=x)dx}\\]
    - The following figures show the pdf and cdf of a standard Gaussian random variable, \\(\mathcal{N}(0,1)\\).
        <p align="center">
            <img width="500" alt="Screenshot 2023-07-30 at 7 21 57 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/75016f02-6145-4625-8172-235bd2dbf811">
        <br>
          <em>PDf and CDF of a standard Gaussian random variable.</em>
        </p>
  - **Multivariate Random variables.** A cdf of a real-valued random vector \\(\mathbf{X} \in \\mathbb{R}^{p}\\) (aka multivariate real-valued random variable) is given by:
    \\[F_{\mathbf{X}}(x) = Pr(\mathbf{x}) = Pr_{\mathbf{X}}(\mathbf{x}) = Pr(X_1\leq x_1, X_2\leq x_2, \ldots, X_p\leq x_p)\\]
    - For the continuous case, the cdf can be expressed as the integral of the pdf \\(p(\mathbf{x})\\):
      \\[Pr(\mathbf{x}) = \int _{\mathbf{x} \in \mathbb{R}^p}p(\mathbf{x})d\mathbf{x} = \int _{-\infty}^{x_p} \int _{-\infty}^{x _{p-1}}\ldots \int _{-\infty}^{x_1}p(x_1, x_2, \ldots, x_p) dx_1 dx_2 \ldots dx_p \\]
* **Support.** Support of a r.v. \\(X\\) is a set \\(\mathcal{X}\subseteq \Omega\\) such that \\(p(x)\neq 0 ~\text{or} ~ P(x)\neq 0, ~ \forall x\in \mathcal{X}\\).
* **Independent and Identically Distributed (IID) Random Variables.** A set of random variables is said to be iid if they are mutually independent and drawn from the same probability distribution. We denoted \\(n\\) iid random variables drawn from a distribution \\(p\\) as \\(x_1, x_2, \ldots, x_n \stackrel{iid}{\sim} p\\).
* **Marginalization.** Given a joint pdf (pmf) of \\(n\\) random variables, we can obtain pdf (pmf) of one or any number of variables through marginalization. That is,
    \begin{equation}
        \begin{aligned}
        p(x_i) = \int _{\mathbf{x} _{-i}} p(x_1, x_2, \ldots, x_n)d\mathbf{x _{-i}} ~~ \text{or} ~~ p(x_i) = \sum _{\mathbf{x} _{-i}} p(x_1, x_2, \ldots, x_n)
        \end{aligned}
    \end{equation}
  - This is also called _**sum rule**_ or the _**rule of total probability**_.
  - Please note that in the above integral, we take integral over \\(n-1\\) variables (all but \\(x_i\\)) denoted by vector \\(\mathbf{x_{-i}}\\). So, this is a multiple integral problem.
  
* **Chain Rule.** A very important result in probability is _Chain Rule_:
  \\[p(x_1, x_2, \ldots, x_n) = p(x_1)p(x_2 \| x_1)p(x_3 \| x_1, x_2)\ldots p(x_n\|x_1,x_2,\ldots,x_{n-1})\\]

* **Exchangability.** Consider a sequence of random variables \\(x_1, x_2, \ldots, x_n\\). If for any \\(n\\), the joint probability of these r.v.'s is invariant to permutation of indices, then the sequence is said to be _infinitely exchangeable_. That is, \\(p(x_1, x_2, \ldots, x_n) = p(x_{\pi_1}, x_{\pi_2}, \ldots, x_{\pi_n})~\\), where \\(\pi\\) is a permutation of the index set \\(\\{1,2,\ldots,n\\}\\). The following theorem states the condition for exchangeability (here we state the theorem for continuous r.v.'s, but the same is true for discrete cases).
   - **De Finetti’s Theorem.** A sequence of random variables \\(x_1, x_2, \ldots, x_n\\) is infinitely exchangeable if and only if for all \\(n\\), we have
   \\[p(x_1, x_2, \ldots, x_n) = \int \prod _{i=1}^n p(x_i\|\pmb{\theta})p(\pmb{\theta})d\pmb{\theta}\\]
     - where \\(\pmb{\theta}\\) is some hidden random variable (possibly infinite-dimensional) which is common to all variables. This means that is, \\(x_1, x_2, \ldots, x_n\\) are iid r.v.'s conditional on \\(\pmb{\theta}\\).

* **Quantiles.**
  - If the cdf is strictly monotonically increasing, then it is invertible and its inverse is called the _**inverse cdf, or percent point function (ppf), or quantile function**_.
  - The \\(q^{th}\\) quantile of the cdf of a random variable \\(X\\) denoted by \\(x_q\\) and is defined by \\(P^{-1}(q)\\) where \\( Pr(X \leq x_q) = q \\).
    - The value \\(P^{-1}(0.5)\\) is _the median of a distribution_. The median is the point where half of the probability mass is on the left and half on the right.
    - The values \\(P^{-1}(0.25)\\) and \\(P^{-1}(0.75)\\) are the lower and upper quartiles.

- **Expectation (Mean).** Mean, or expected value of a random variable is the first moment of distribution (see below for more details), often denoted by \\(\mu_x\\) or \\(\mu_1\\). For a continuous r.v. with the support \\(\mathcal{\Omega}\\), the mean is defined as follows (if the integral is not finite, the mean is not defined):
  \\[\mu_x = \mu_1 = \mathbb{E}(X) = \int _{x\in \mathcal{\Omega}} x p(x)dx\\]
  And for a discrete r.v. with the support \\(\mathcal{\Omega}\\):
  \\[\mu_x = \mu_1 = \mathbb{E}(X) = \sum _{x\in \mathcal{\Omega}} x P(x)\\]
  - Expectation is a linear operator, i.e., let \\(X\\) be a random variable (discrete/continous) and \\(a\\) and \\(b\\) be two constants, \\(E(aX+b) = aE(X) + b\\). More generally, \\(\mathbb{E}\[\sum _{i-1}^n X_i\] = \sum _{i=1}^n \mathbb{E}X_i\\), where \\(X_1, X_2, \ldots X_n\\) are \\(n\\) random variables (discrete/continous).
  - Let \\(X_1, X_2, \ldots X_n\\) be independent random variables. Then, \\(\mathbb{E}\[\prod _{i=1}^nX_i\] = \prod _{i=1}^n\mathbb{E}X_i\\).
  - **Law of Iterated Expectations (Law of Total Expectation**
    
- **Variance.** Variance of a random variable is the second moment of distribution (see below for more details), often denoted by \\(\\sigma_x^2\\), \\(m_2\\), \\(var(X)\\). For a continuous r.v. with the support \\(\mathcal{\Omega}\\), the variance is defined as follows (if the integral is not finite, the variance is not defined):
  \\[\sigma_x^2 = m_2 = var(X) = \mathbb{E}\[(X-\mu_x)^2 \]=\int _{x\in \mathcal{\Omega}} (x-\mu_x)^2 p(x)dx\\]
   - We can simplify the above relation as follows;
     \\[\sigma_x^2 = \mathbb{E}(X^2) + \mu^2 -2\mathbb{E}(X)\mu_x = \mathbb{E}(X^2) - \mu^2 = \mathbb{E}(X^2) -\mathbb{E}(X)^2\\]
   - Since \\(var(X)\geq 0\\), we immiadiately see that \\(\mathbb{E}(X^2) \geq \mathbb{E}(X)^2\\).
   - The square root of variance is called _standard deviation_, \\(std(X) = \sigma_x = \sqrt{var(x)}\\).
   - Let \\(\alpha\\) and \\(\beta~\\) be two constants (determonistic variables), then \\(var(\alpha X + \beta) =\alpha^2var(X)\\).
   - Let \\(X_1, X_2, \ldots X_n\\) be independent random variables. Then, the variance of their sum is given by \\(var\[\[\prod _{i=1}^nX_i\] = \prod _{i=1}^n var(X_i)\\).
   - Let \\(X_1, X_2, \ldots X_n\\) be independent random variables. Then, the variance of their product is given by
     \begin{equation}
        \begin{aligned}
           var(\prod _{i=1}^n X_i) = \mathbb{E}\[(\prod _{i=1}^n X_i)^2\] - (\mathbb{E}\[\prod _{i=1}^n X_i\])^2 \\\\\\\\
           \prod _{i=1}^n\mathbb{E}(X_i^2) - (\prod _{i=1}^n\mathbb{E}X_i)^2 \\\\\\\\
           \prod _{i=1}^n(\sigma_x^2 + \mu_x^2) - \prod _{i=1}^n\mu_x^2
         \end{aligned}
      \end{equation}
    - **Law of Total Variance (Conditional Variance Formula)**
  
- **Mode.** The mode of a distribution is the value in which the probability mass function or probability density function are maximized:
  \\[x^* = argmax_x p(x)\\]
  - If the distribution is multimodal, the solution to the above maximization problem may not be unique.
  
* **Moments of a Distribution**
  * Moment in probability and momentum in physics are similar concepts. In fact, both of these words come from the Latin word "movimentum", meaning to move, set in motion, or change [G. Gundersen, 2020]. Since momentum in physics is concerned with torque, and force, it is inherently related to the distribution of mass in objects. Similarly, in probability, moments of distribution help us to understand how the mass of data is distributed. In particular, moments reveal a distribution's _location_, _scale_, and _shape_. The location is captured by the first moment or mean of the distribution and it tells us how far away from the origin the center of mass is. The scale is captured by the second moment and shows how spread out a distribution is. The shape of a distribution is revealed by a higher moment.
  * Generally, \\(k^{th}\\) moment can be defined in different ways (here, we only state the formulas for the continuous case. The discrete case has the same formula by replacing "integral" with "sigma"):
  * Consider a random variable \\(X\\) with support \\(\mathcal{X}\\).
    1. \\(k^{th}\\) Raw Moment: \\(\mu_k = \mathbb{E}(X^k) = \int _{x\in \mathcal{X}} x^k p(x)dx\\)
       - With \\(k=1\\), we obtain the expectation or the mean of a random variable. 
    2. \\(k^{th}\\) Central Moment: \\(m_k = \mathbb{E}\[(X-\mu_k)^K\] = \int _{x\in \mathcal{X}} (x-\mu_k)^k p(x)dx\\)
       - With \\(k=1\\), we obtain the variance of a random variable.
    3. \\(k^{th}\\) Standardized Moment: \\(\overline{m} _k = \mathbb{E}\[(\frac{X-\mu_k}{\sigma_x})^k\] = \int _{x\in \mathcal{X}} (\frac{x-\mu_k}{\sigma_x})^k p(x)dx\\).
       - \\(Z=\frac{X-\mu_k}{\sigma_x}\\) is called the standard score or z-score.
       - With \\(k=1\\), we obtain the _skeness of a random variable.
       - The skewness is positive for longer right tails (right-skewed or positively skewed distribution) and negative for longer left tails (left-skewed or negatively skewed distribution).
       - A symmetric random variable has a zero skewness, but the opposite is not always true.
       - With \\(k=4\\), we obtain the _kurtosis_ of a distribution. Kurtosis is a measure, describing how the tails of a distribution differ from the tails of a normal distribution. There are three types of kurtosis: mesokurtic (Distributions with medium tails), leptokurtic (distributions with fat tails), and platykurtic (distributions with thin tails).
    5. \\(k^{th}\\) Sample Moment about \\(C\\): \\(\tilde{m} _k = \frac{1}{N}\sum _{i=1}^{n}(X_i-C)^k\\)
  
## Common Discrete Random Variables
* **Bernoulli.** A Bernoulli random variable with parameter \\(0 \leq \theta \leq 1\\) is a binary discrete r.v. with the following pmf:
  \begin{equation}
        \begin{aligned}
          Ber(x|\theta) =
            \begin{cases}
              ~ \theta, & x=1  \\\\\\\\
              1-\theta, & x=0
            \end{cases} = \theta^{x}\theta^{1-x}
        \end{aligned}
        \end{equation}

* **Multinomial (Categorical).** 
* **Binomial.** Consiider \\(n\\) random variables \\(X_i\stackrel{iid}\sim Ber(x|\theta), ~ i=1,2,\ldots,n\\). The sum of these random variables, \\(Y=\sum _{i=1}^n X_i\\) is called Binomial r.v. with parameters \\((n, \theta)\\), and is defined as follows:
  \\[Bin(y\|n, \theta) = \binom ni \theta^{i}(1-\theta)^{n-i}\\]
    - \\(\binom ni = \frac{n!}{(n-i)!i!}\\) is the number of ways to choose \\(i\\) items from \\(n\\) (aka binomial coefficient).
* **Multinomial.**
* **Poisson.**

## Common Continous Random Variables
* **Gaussian (Normal).**
* **Exponential.**
* **Student-t.**
* **Gamma.**
* **Chi-Squared.**
* **Beta.**
* **Wishart.**
