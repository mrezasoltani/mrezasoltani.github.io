---
title: "Module 2 -- Probability and Statistics"
classes: wide
---
## What Is Probability?
* Probability is a critical tool for modern data analysis. It arises in dealing with uncertainty, randomized algorithms, and Bayesian analysis. Generally, we can interpret probability in two ways; the **_Fequentist_** interpretation and **_Bayesian_** interpretation.
* In the frequentist view, probabilities represent long-run frequencies of events that can happen multiple times. For example, when we say a fair coin has a \\(50\\%\\) chance of turning tail, we mean that if we toss the coin many times, we expect that it lands tail half of the time.
* In Bayesian interpretation, probability is used to quantify our ignorance or uncertainty about something. This viewpoint is more related to information rather than repeated trials. In the above flipping a fair coin, Bayesian interpretation states our belief that the coin is equally likely to land heads or tails on the next toss.
* Bayesian interpretation can be used to model our uncertainty about events that do not have long-term frequencies [K. Murthy, 2022].

### Type of Uncertainty
  1. **Epistemic (Model) Uncertainty.**: This uncertainty is due to our ignorance and uncertainty about the mechanism of generating the data.
  2. **Aleatoric (Data) Uncertainty.** The uncertainty is due to the intrinsic variability in the data and cannot be reduced even more collection of data. This is derived from the Latin word for “dice”.

## Informal Review of Some Concepts
* **Sample Space.** Set of all possible outcomes in a random experiment is called _sample space_ and denoted by \\(\Omega\\). For example, in rolling a die, there are \\(6\\) possible outcomes, so \\(\Omega = \\{1,2,3,4,5,6\\}\\). An outcome is an element of a sample space, e.g., \\(\omega = 3\\). 
* **Event.** A subset of the sample space is an _event_, i.e., \\(A\subseteq \Omega\\). For example, in the rolling of a die, an event can be defined as _facing an odd number_, i.e., \\(A=\\{1,3,5\\}\\). For a sample space with \\(n\\) outcomes, we can have \\(2^n\\) events.
* **Probability.** Informally, a probability is a measure from the set of events to the real numbers in \\(\[0,1\]\\) such that:
  1. For any event \\(A ~\\), \\(0 \leq (P(A) \leq 1\\)
  2. \\(p(\Omega) = 1\\)
  3. For any sequence \\(A_1, A_2, \ldots\\), such that \\(\forall i,j, ~ A_i \cap A_j = 0 \\) (i.e, pairwise disjoint sets):
     \\[P(\cup _{i=1}^{\infty} A_i) = \sum _{i=1}^{\infty} P(A_i)\\] 
