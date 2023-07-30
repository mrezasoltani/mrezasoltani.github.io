---
title: "Module 2 -- Probability"
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
Here, we review most important concepts in the probability theory without mathematical rigorous. Later, we make these concepts more formal by defining them using  the language of Measure Theory.
* **Sample Space.** Set of all possible outcomes in a random experiment is called _sample space_ and denoted by \\(\Omega\\). For example, in rolling a die, there are \\(6\\) possible outcomes, so \\(\Omega = \\{1,2,3,4,5,6\\}\\). An outcome is an element of a sample space, e.g., \\(\omega = 3\\). This example shows a discrete sample space. A sample space can also be a continous space. For example, consider waiting time for arrving a bus is a random experiment. An outcome is any non-negative real number. 
* **Event.** A subset of the sample space is an _event_, i.e., \\(A\subseteq \Omega\\). For example, in the rolling of a die, an event can be defined as _facing an odd number_, i.e., \\(A=\\{1,3,5\\}\\). For a sample space with \\(n\\) outcomes, we can have \\(2^n\\) events.
* **Probability.** Informally, a probability is a measure from the set of events to the real numbers in \\(\[0,1\]\\) such that (axioms of probability, Kolmogorov axioms):
  1. For any event \\(A\subseteq \Omega\\), \\(~ 0 \leq P(A) \leq 1\\)
  2. \\(p(\Omega) = 1\\)
  3. For any sequence \\(A_1, A_2, \ldots\\), such that \\(\forall i,j, ~ A_i \cap A_j = 0 \\) (i.e, pairwise disjoint sets):
     \\[P(\cup _{i=1}^{\infty} A_i) = \sum _{i=1}^{\infty} P(A_i)\\]
     - The disjoin sets assumption in the third property means that events \\(A\\) and \\(B\\) cannot happen at the same time, that is, they are mutually exclusive. For example, in flipping a coin, the event of facing with a tail does not have any intersection with the event that the coin lands the head.
     - **Probability of union of two events (addition rule).** The probability of event A or B happening is given by \\(P(A\cup B) = P(A) + P(B) - P(A\cap B)\\).
* **Joint Probability.** A joint probability of happening two events \\(A\\) and \\(B\\) is defined as \\(P(A, B) = P(A\cap B)\\).
* **Independency.** If two events \\(A\\) and \\(B\\) are independent, then their join probability is given by their product, i.e., \\(P(A\cap B) = P(A)\cdot P(B)\\). This means that occurring the event \\(A\\) does not have any effect in happening or not happening of the event \\(B\\). For example, consider a random experiment defined as follows: choosing two numbers uniformly at random from a set \\(X\\{1,2,3,4\\}\\). Let \\(A\\) be an event that the first number belongs to the set \\(\\{3\\}\\), and \\(B\\) be an event that the second number belongs to the set \\(\\{1, 2, 4\\}\\), then these two event are independent from each other; hence, \\(P(A\cap B) = P(A)\cdot P(B) = \frac{1}{4}\cdot\frac{2}{4} = \frac{1}{8}\\).
  - It is very important to note that in independency of two events does not imply their mutually exclusive and vice versa.
  - **Mutual Independence.** Let \\(A_{1},A_2,\dots,A_{n}\\) be \\(n\\) events. \\(A_{1},A_2,\dots,A_{n}\\) are mutually independent (jointly independent ) if and only if for any sub-collection of \\(k\\) events (\\(k\leq n\\)) \\(A_{1},A_2,\dots,A_{k}\\), \\(P(\cap _{i=1}^{k} A_i) = \prod _{i=1}^{k} P(A_i)\\).
  - **Pairwise Independence.** For \\(n \geq 3\\), events \\(A_{1},A_2,\dots,A_{n}\\) are pairwise independent if \\(A_i \cap A_j = 0, ~\forall 1\leq i,j\leq n\\).
    - Please note that Pairwise Independence does not imply Mutual Independence.
* **Conditional Probability.** The conditional probability of happening an event \\(A\\) given that another event \\(B\\) has occurred is given by \\(P(A\|B) = \frac{p(A, B)}{p(B)}, ~ P(B)\neq 0\\).
* **Conditional Independency.** If two events \\(A\\) and \\(B\\) are independent given another event \\(C\\), we say that \\(A\\) and \\(B\\) are _conditionally independent_ given \\(C\\), this is denoted by \\(A \perp B \| C \\). In this case, \\(P(A, B\|C) = P(A\|C)\cdot P(B\|C)\\). Many events are dependent on each other, but it is possible when a relevant information is known, then two event become independent.
* **Law of Total Probability.** Let \\(A_{1},A_2,\dots,A_{n}\\) be a partition of sample space \\(\Omega\\). That is, \\(\cup _{i=1}^n A_i = \Omega, ~ A_i\cap A_j =0, ~ \forall ~ 0\leq i,j\leq n\\). Then for any set \\(B\subseteq\Omega\\), we have:
  \\[P(B) = \sum _{i=1}^{n} P(B\|A_i)P(A_i)\\]
  <details>
  <summary>Proof</summary>
  \begin{equation}
  B = B\cap\Omega = B\cap(\cup _{i=1}^n A_i ) \\\\
    = \cup _{i=1}^n(B\cap A_i)
  \end{equation}
    We note that \\(B\cap A_i\\) are disjoint matrices by the auumption on \\(A_{1},A_2,\dots,A_{n}\\) being a partition of sample space. Hence, applying the third axioms of probability, we have:
    \\[P(B\cap A_i) = \sum _{i=1}^{n} P(B\cap A_i) = \sum _{i=1}^{n} P(B\|A_i)P(A_i)\\]
    For the last eqaulity we have used the conditional probabilit rule. 
  </details>
* **Bayes Rule.**
* **Random Variables.** A random variable (r.v.) \\(X: \Omega\rightarrow \mathbb{R}\\) is a function from the sample space to the real line.
  - **Discrete random variables.** If the sample space \\(\Omega\\) over which a random variable \\(X\\) is defined is finite or countably infinite, then \\(X\\) is called a discrete random variable. Any possible realization of \\X\\) is an event, hence, the probability of the event that \\(X\\) has value \\(x\\) is given by \\(P(X=x)\\) (Most of the time for simplicity of notation, we write \\(P(x)\\)).
    - \\(P(x)\\) is called **probability mass function (pmf)** and satisfies the following conditions:
     - \\(0 \leq P(x) \leq 1\\)
     - \\(\sum _{x\in \Omega}P(x) = 1\\)
    - **Cumulative distribution function (cdf).** We define a cdf of a r.v. as \\(Pr(x) = Pr_X(x) =  Pr(X\leq x) = \sum _{\alpha \in \Omega, \alpha\leq x} P(\alpha)\\). From this definition, we immediately see that \\(Pr(\alpha \leq X \leq \beta) = Pr(\beta) - Pr(\alpha)\\).
    - The cdf is a monotonically increasing function, and is always continuous **_only_** from the right.
  - **Continuous random variables.** If the sample space \\(\Omega\\) over which a random variable \\(X\\) is defined is infinite (e.g., \\(\mathbb{R}\\)), then \\(X\\) is called a Continuous random variable. Here, we cannot consider a set of finite/countable values for \\(X\\). However, we can choose a set of intervals in \\(\mathbb{R}\\), for example, and define the probability of \\(X\\) taking on a specific value by infinitesimal interval containing that value.
    - **Probability density function (pdf)** For the a continuous r.v., we can define the pdf as \\(p(x)=\frac{dPr(x)}{dx}\\), assuming this derivative exists. 
    - Similar to the discrete case, the cdf function for a continuous r.v. is defined by \\(Pr(x) = Pr_X(x) =  Pr(X\leq x) = \int _{u \in \Omega, u\leq x} p(u)du\\), where \\(p(u)\\) is the probability density function (pdf). If \\(\Omega = \mathbb{R}\\), then \\(Pr(x) = \int _{-\infty}^{x}p(u)du\\). From this, we can compute the probability of a continuous variable being in a finite interval \\(\[\alpha, \beta\]\\) as follows:
      \\[Pr(\alpha \leq X \leq \beta) = Pr(\beta) - Pr(\alpha) = \int _{\alpha}^{\beta}p(u)du \\]
    - If the above interval is an infinitesimal interval, from calculus, \\(Pr(x \leq X \leq x+\Delta x) \approx p(x)dx\\). So, the product of the density at \\(x\\) and the width of the intervalt gives he probability of \\(X\\) being in a small interval around \\(x\\).
    - The cdf is a monotonically increasing function and for a continuous r.v. is always a continuous function.
* **Quantiles.**
* **Moments of a Distribution**
  - * **Expectation (Mean).**
