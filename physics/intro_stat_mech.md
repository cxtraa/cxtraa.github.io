---
title: 'An introduction to statistical mechanics'
layout: post
parent: Physics
nav-order: 1
scholar:
  style: apa
  locale: en

  sort_by: none
  order: ascending

  source: ./physics
  bibliography: intro_stat_mech_ref.bib
  bibliography_template: "{{reference}}"

  replace_strings: true
  join_strings:    true

  details_dir:    bibliography
  details_layout: bibtex.html
  details_link:   Details

  query: "@*"
---

# An introduction to statistical mechanics
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

In classical mechanics, Newton's laws can be used to predict the displacement of a particle over time. However, for systems with $$ N $$ particles, it is intractable to integrate Newton's second law for every particle individually. Statistical mechanics deals with the macroscopic behaviour of systems where $$ N $$ is very large, usually on the order of $$ 10^{23} $$, through quantities such as the entropy $$ S $$, and the internal energy $$ U $$. Entropy appears in statistical mechanics, Bayesian inference, and information theory, suggesting it is a universal statistical property, not just a thermodynamic property of a system.

## Microstates and macrostates

We will begin our discussion of statistical mechanics by considering a set of discrete energy levels, where $$ E_j $$ denotes the energy of the $$ j^{th} $$ energy level. $$ E_0 $$ denotes the ground state which typically has zero energy. A particle can exist in one of these discrete $$ E_j $$. Now, suppose we have three particles A, B, and C; particle A is at $$ E_0 $$, particle B is at $$ E_2 $$, and particle C is also at $$ E_2 $$. This specific *permutation* of particles is called a *microstate*:

{: .definition }
> A microstate is a **permutation** of particles arranged into energy levels, subject to the following constraints:
>
> $$ \sum_{j=0}^{\infty} N_j = N $$
>
> $$ \sum_{j=0}^{\infty} E_j \cdot N_j = U $$
>
> where $$ N_j $$ denotes the number of particles in energy level $$ E_j $$, $$ N $$ is the total number of particles, and $$ U $$ is the internal energy of the system.

The total number of microstates available to a system is denoted by $$ \Omega_{N, U} $$. The subscript of $$ N, U $$ indicates that the total number of particles is $$ N $$, and their energies sum to $$ U $$. Sticking with our original example, suppose that we swapped particles $$ A $$ and $$ B $$. This system is different to the one we had before, but looks qualitatively the same - we have one particle in the ground state $$ E_0 $$, and two particles in $$ E_2 $$. This leads us to the definition of a *macrostate*:

{: .definition}
> A macrostate is a **combination** of particles arranged into energy levels. It is defined by solely by the distribution of $$ N_j $$.

The *multiplicity* of a macrostate is denoted by $$ W $$, which refers to the *number of microstates for a given macrostate*. For example, $$ W = 3 $$ for our example, because there are three microstates that give one particle in $$ E_0 $$, and two particles in $$ E_2 $$.

## The statistical definition of entropy

In classical thermodynamics, the change in entropy across some reversible path is given by:

$$ \Delta S = \int \frac{dQ}{T} $$

In statistical mechanics, we present the following formula for entropy:

$$ S_B = k_B \ln(\Omega_{N, U}) $$

Entropy is a measure of our uncertainty about a system. The greater the number of available microstates of a system, the more uncertain we are about which microstate is actually the microstate the system is in at that instant in time, and hence the entropy should intuitively be larger, which is exact what the statistical entropy predicts. Recall the Second Law of Thermodynamics:

{: .definition }
> The Second Law states that the entropy of an isolated system must either increase over time or remain constant. When the entropy of a system is maximised, the system is said to be in **thermodynamic equilibrium**. 

One of the key postulates of statistical mechanics is that at equilbrium (when $$ S $$ is maximised), *all microstates are equally probable*. This means that the microstate where all of the internal energy $$ U $$ is stored in a single particle, and the rest of the particle sit dead in $$ E_0 $$ is equally probable to the one where the energy is evenly distributed between the particles. However, there are many more permutations where the energy is *almost* evenly distributed between the particles, so that is the microstate we tend to observe.

It follows that the probability of observing a microstate $$ i $$ is:

$$ P_i = \frac{1}{\Omega_{N, U}} $$

i.e. a uniform distribution. And the probability that we observe a particular macrostate $$ i $$ is:

$$ P_i = \frac{W_i}{\Omega_{N, U}} $$

The multiplicity $$ W $$ can be described by the following formula:

$$ W = \frac{N!}{N_0! \, N_1! \, \ldots \, N_{\infty}!} $$

To provide intutition for this expression, consider if each energy level had either 0 or 1 particle in it. Then, the number of ways of arranging these particles would just be $$ N! $$. However, when we have multiple particles in the *same energy level*, we do not care about their ordering. Hence, we must divide by the number of ways we can order the particles in the $$ j^{th} $$ energy level, which is $$ N_j ! $$. We do this for every energy level, yielding the expression we wanted for $$ W $$.

We will also briefly describe what an *ensemble* is:

{: .definition}
> An **ensemble** is a special type of macrostate. It is the set of all microstates that satisfy the ensemble's conditions.

There are three main types of ensemble:
- **Microcanonical Ensemble** - a system with fixed $$ U $$, $$ N $$, and $$ V $$, representing an isolated system.
- **Canonical Ensemble** - a system with fixed $$ N $$, $$ V $$, and $$ T $$, but fluctuant $$ U $$. This represents a system in thermal equilbrium with a heat bath.
- **Grand Canonical Ensemble** - a system with fixed chemical potential $$ \mu $$, $$ V $$, and $$ T $$, but $$ N $$ and $$ U $$ are fluctuant. It represents a system in equilbrium with a heat bath and particle reservoir.

## The Gibbs entropy

Previously, we introduced the statistical entropy $$ S_B = k_B \ln(\Omega_{N, U}) $$. $$ \Omega_{N, U} $$ is extremely difficult to measure, so instead, we make the following assumption:

$$ \Omega_{N, U} \approx W_{max} $$

where $$ W_{max} $$ denotes the multiplicity of the macrostate with the most microstates. This approximation is valid for large $$ N $$, which is well-justified in the case of a gas where $$ N $$ is on the order of $$ 10^{23} $$! Hence:

$$ S_B = k_B \ln(W_{max}) $$

Now, let us take the natural logarithm of $$ W $$:

$$
\begin{align}
\ln(W) &= \ln(\frac{N!}{N_0! \, N_1! \, \ldots \, N_{\infty}!}) \\
&= \ln(N!) - \ln(N_0!) - \ln(N_1!) - ...
\end{align}
$$

Recall Stirling's approximation for $$ \ln(N!) $$:

$$ \ln(N!) \approx N\ln(N) - N $$

Hence, letting $$ p_j $$ be the probability of finding a particle in energy level $$ E_j $$, such that $$ p_j = \frac{N_j}{N} $$:

$$
\begin{align}
ln(W) &\approx N\ln(N) - N - \sum_{j=0}^{\infty} \left( N_j \ln(N_j) - N_j \right) \\
&= N\ln(N) - \sum_{j=0}^{\infty} \left( N_j \ln(N_j) \right) \\
&= N\ln(N) - \sum_{j=0}^{\infty} \left( p_j N \ln(p_j N) \right) \\
&= N\ln(N) - N\sum_{j=0}^{\infty} \left( p_j \ln(p_j) + p_j \ln(N) \right) \\
&= -N\sum_{j=0}^{\infty} \left(p_j \ln(p_j) \right)
\end{align}
$$

From this, we can substitute into the equation $$ S_B = k_B \ln(W_{max}) $$ to obtain the Gibbs entropy.

{: .definition}
> The **Gibbs entropy** is a statistical approximation to the Boltzmann entropy, and is closely related to the Shannon information entropy:
>
> $$ S_g = -N k_B \sum_{j=0}^{\infty} p_j ln(p_j) $$

Notice that this formula is very similar to the information entropy of a probability distribution, $$ H(X) = - \sum_{i} p_i \log_2{p_i} $$. The Gibbs entropy, much like the information entropy, is maximised when the distribution of $$ p_j $$ is uniform. Therefore, the entropy is maximised when all microstates are equally probable, which is the postulate we proposed earlier. Heat will flow from areas of higher temperature to lower temperature until the system is at a uniform temperature and at thermal equilibrium, and the number of possible microstates is at its maximum, and so the entropy is at its maximum.

## The Boltzmann distribution

Recall that in thermodynamic equilibrium, the entropy is maximised. We wish to find the probability distribution $$ p(E_j) $$ that tells us the fraction of particles in energy level $$ E_j $$. This distribution $$ p(E_j) $$ must maximise the entropy.

We will do this using the method of *Lagrange multipliers*, subject to the following two constraints:

$$ \sum_{i=0}^{\infty} p_j = 1 $$

$$ \sum_{i=0}^{\infty} p_j E_j = U $$

We can express our problem using the Lagrange multipliers $$ \lambda $$ and $$ \mu $$:

$$
S_g = -\sum_{j=0}^{\infty} p_j \ln(p_i) - \lambda \left( \sum_{i=0}^{\infty} p_j - 1 \right) - \beta \left( \sum_{j=0}^{\infty} E_j p_j - \bar{U} \right)
$$

Differentiating with respect to $$ p_j $$ and setting $$ \frac{dS_g}{dp_j} = 0 $$, we obtain:

$$
\frac{\partial S_g}{\partial p_j} = 0 = -(1 + \ln p_j) - \lambda - \beta E_i
$$

Hence, we find a preliminary expression for p_i:

$$ p_i = e^{-1 - \lambda - \beta E_i} $$

Using the normalisation constraint, we obtain:

$$ e^{-1-\lambda} = \frac{1}{\sum_{i=0}^{\infty} e^{-\beta E_i}} $$

Hence, we can re-express $$ p_i $$ as:

$$ p_i = \frac{e^{-\beta E_i}}{\sum_{i=0}^{\infty} e^{-\beta E_i}} = \frac{e^{-\beta E_i}}{Z} $$

where $$ Z $$ is an important quantity in statistical mechanics known as the **partition function**, and $$ \beta = \frac{1}{k_B T} $$. This is the *Boltzmann distribution*. There are a couple of interesting things to note about this distribution:

- $$ p_i $$ decays exponentially. We are more likely to find particles in lower energy states than higher ones.
- $$ \beta $$ plays the role of an "inverse temperature". For high $$ \beta $$ (low T), the proportion of particles at low energy levels increases, and for low $$ \beta $$ (high T), the fraction of particles at high energy levels increases.

{: .definition}
> The **Boltzmann distribution** gives the probability $$ p_i $$ of finding a particle at energy level $$ E_i $$:
>
> $$ p_i = \frac{e^{-\beta E_i}}{Z} $$

## The magic box relations

$$ Z $$, the partition function seems very unassuming at the moment; it appears to only be a normalisation constant. It turns out that $$ Z $$ is a "magic box" that can give us all the major macroscopic quantities of a system, such as the internal energy $$ U $$ and the Helmholtz free energy $$ F $$. Let us begin to derive these "magic box relations".

We begin by substituting the Boltzmann distribution for $$ p_i $$ in the Gibbs entropy equation:

$$
\begin{align}
S_g &= -k_B N \sum_{j} p_j \ln(p_j) \\
&= -k_B N \sum_{j} \frac{e^{-\beta E_j}}{Z} \ln\left(\frac{e^{-\beta E_j}}{Z}\right) \\
&= k_B N \sum_{j} \frac{e^{-\beta E_j}}{Z} \left(\beta E_j + \ln(Z)\right) \\
&= k_B N \beta \sum_{j} p_j E_j + k_B N \ln(Z) \sum_{j} p_j \\
&= k_B N \beta U + k_B N \ln(Z) \\
F &= U - TS_g \\
&= U + k_B N T \left( -\beta U - \ln(Z) \right) \\
&= U - k_B N T \beta U - k_B N T \ln(Z) \\
&= -k_B N T \ln(Z) \\
&= -\frac{N}{\beta} \ln(Z)
\end{align}
$$

Hence, the magic box of $$ Z $$ has given us the Helmholtz free energy F! $$ U $$, the internal energy, can be found through similar arguments. This is the second magic box relation:

$$ U = -N \frac{\partial{\ln(Z)}}{\partial{\beta}} $$

{: .definition }
> Using the **magic box relations**, we can obtain macroscopic quantities such as $$ U $$ and $$ F $$ of a thermodynamic system, from the partition function $$ Z $$:
>
> $$ U = -N \frac{\partial{\ln(Z)}}{\partial{\beta}} $$
> 
> $$ F = -\frac{N}{\beta} \ln(Z) $$

## State degeneracy and phase space

There have been a couple of issues with our analysis so far. Let's discuss each of them in turn.

Firstly, it is possible to have multiple states with the same energy level. This is known as **degeneracy**.

{: .definition}
> **Degeneracy** is when multiple states have the same energy level $$ E_j $$.

Our formula for the partition function, $$ Z = \sum_{j=0}^{\infty} e^{-\beta E_j} $$ summed over each energy level $$ E_j $$ just once, neglecting the possibility of state degeneracy. We can solve this problem by introducing a quantity known as the **density of state**, denoted by $$ g(E_j) $$. It tells us the number of states at energy level $$ E_j $$. The modified formula for the partition function is thus:

$$ Z = \sum_{i=0}^{\infty} g(E_j) e^{-\beta E_j} $$

This is not too bad for a system with discrete energy levels, such as an electron in an atom, but what about systems where the energy can be continuous? In such cases, the partition function transitions from a sum to an integral:

$$ Z = \int_{0}^{\infty} g(E) e^{-\beta E} \, dE $$

In the continuous case, $$ g(E) = \frac{dN}{dE} $$, i.e. the number of particles N per unit energy.

References
----------
{% bibliography %}































