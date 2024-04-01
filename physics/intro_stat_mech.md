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

{: .definition }
> The **Boltzmann entropy** of a system is:
> 
> $$ S_B = k_B \ln(\Omega_{N, U}) $$

Entropy is a measure of our uncertainty about a system. The greater the number of available microstates of a system, the more uncertain we are about which microstate is actually the microstate the system is in at that instant in time, and hence the entropy should intuitively be larger, which is exact what the statistical entropy predicts. Recall the Second Law of Thermodynamics:

{: .definition }
> The **Second Law** states that the entropy of an isolated system must either increase over time or remain constant. When the entropy of a system is maximised, the system is said to be in **thermodynamic equilibrium**. 

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
\ln(W) &= \ln\left(\frac{N!}{N_0! \, N_1! \, \ldots \, N_{\infty}!}\right) \\
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

We begin by substituting the Boltzmann distribution for $$ p_i $$ in the Gibbs entropy equation for one particle:

$$
\begin{align}
S_g &= -k_B \sum_{j} p_j \ln(p_j) \\
&= -k_B \sum_{j} \frac{e^{-\beta E_j}}{Z} \ln\left(\frac{e^{-\beta E_j}}{Z}\right) \\
&= k_B \sum_{j} \frac{e^{-\beta E_j}}{Z} \left(\beta E_j + \ln(Z)\right) \\
&= k_B \beta \sum_{j} p_j E_j + k_B \ln(Z) \sum_{j} p_j \\
&= k_B \beta U + k_B \ln(Z) \\
F &= U - TS_g \\
&= U + k_B T \left( -\beta U - \ln(Z) \right) \\
&= U - k_B T \beta U - k_B T \ln(Z) \\
&= -k_B T \ln(Z) \\
&= -\frac{1}{\beta} \ln(Z)
\end{align}
$$

Hence, the magic box of $$ Z $$ has given us the Helmholtz free energy F! $$ U $$, the internal energy, can be found through similar arguments. This is the second magic box relation:

$$ U = - \frac{\partial{\ln(Z)}}{\partial{\beta}} $$

{: .definition }
> Using the **magic box relations**, we can obtain macroscopic quantities such as $$ U $$ and $$ F $$ of a thermodynamic system, from the partition function $$ Z $$:
>
> $$ U = - \frac{\partial{\ln(Z)}}{\partial{\beta}} $$
> 
> $$ F = -\frac{1}{\beta} \ln(Z) $$

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

We will now make a slight detour and talk about **phase space**.

{: .definition}
> **Phase space** is a plot of momentum $$ \boldsymbol{p} $$ versus position $$ \boldsymbol{x} $$. It represents all possible states of a particle.

For each particle, we can specify it with 6 coordinates $$(x, y, z, p_x, p_y, p_z)$$. Importantly, each coordinate is *unique* - it is impossible to have state degneracy in phase space because no two particles can occupy the same coordinate. This removes the need for $$ g(E) $$ which considerably simplifies things. Unfortunately, this means we're going to have to reformulate $$ Z $$, the partition function, yet *again*. Let's start by doing this for one particle in an ideal gas. The Hamiltonian of the particle is expressed as:

$$ H = E_K + E_P $$

where $$ E_K $$ is the kinetic energy, and $$ E_P $$ is the potential energy. Noting that $$ E_K = \frac{\|\boldsymbol{p}\|^2}{2m} $$, and that in an ideal gas, there are no intermolecular forces, we can say the following:

$$
\begin{align}
H &= \frac{\|\boldsymbol{p}\|^2}{2m} \\
&= \frac{p_x^2 + p_y^2 + p_z^2}{2m}
\end{align}
$$

The Hamiltionian is just the energy of the particle, and so we can substitute it into the Boltzmann distribution in $$ Z $$, expressing $$ Z $$ as a hextuple integral over $$ \boldsymbol{x} $$ and $$ \boldsymbol{p} $$:

$$
Z = \frac{1}{h^3} \int_{V} \int_{\mathbb{R}^3} e^{-\beta \frac{p_x^2 + p_y^2 + p_z^2}{2m}} \, dp_x \, dp_y \, dp_z \, dx \, dy \, dz
$$

(There are actually six integrals here but I've written them as two for brevity.) Noting that the function we are integrating does not depend on $$ \boldsymbol{x} $$, we can rewrite our problem as a triple integral:

$$
\begin{align}
Z &= \frac{V}{h^3} \int_{-\infty}^{+\infty} \int_{-\infty}^{+\infty} \int_{-\infty}^{+\infty} e^{-\beta \frac{p_x^2 + p_y^2 + p_z^2}{2m}} \, dp_x \, dp_y \, dp_z\ \\
&= \frac{V}{h^3} \left( \int_{-\infty}^{+\infty} e^{-\beta \frac{p_x^2}{2m}} \, dp_x \right) \left( \int_{-\infty}^{+\infty} e^{-\beta \frac{p_y^2}{2m}} \, dp_y \right) \left( \int_{-\infty}^{+\infty} e^{-\beta \frac{p_z^2}{2m}} \, dp_z \right) \\
&= \frac{V}{h^3} \left( \frac{2 \pi m}{\beta} \right)^{\frac{3}{2}}
\end{align}
$$

where the final result was derived by using the standard result that $$ \int_{-\infty}^{+\infty} e^{-ax^2} \, dx = \sqrt{\frac{\pi}{a}} $$, which is the famous Gaussian integral.

How, from this, can we obtain the partition function for $$ N $$ particles, $$ Z_n $$? Recall that the partition function sums over states. If there are $$ N_1 $$ states available for 1 particle, then for $$ N $$ particles, we have $$ \frac{N_1^N}{N!} $$ available microstates of the system. The expression $$ \frac{1}{N!} $$ is known as the *Gibbs factor*, and accounts for the fact that the particles are **indistinguishable** from each other; we must therefore divide by the number of permutations of $$ N $$ particles, which is $$ N! $$. It follows that:

$$ Z_N = \frac{Z_N^N}{N!} $$

{: .definition}
> The **N-particle partition** function for an ideal gas is:
>
> $$ Z_N = \frac{1}{N!} \left( \frac{V}{h^3} \left( \frac{2 \pi m}{\beta} \right)^{\frac{3}{2}} \right)^N $$

The "magic box" relations we derived earlier are still valid for our shiny new $$ Z_N $$ - so let's use them! Recall that:

$$ U = -\frac{\partial \ln Z}{\partial \beta} $$

I won't go through all the algebra here, but the result we obtain is:

$$ U = \frac{3}{2} N k_B T $$

which is a familiar result from thermodynamics. Finally, after building up from concepts like microstates and entropy, we have reached the same predictions as classical thermodynamics, under a completely different theory.

## The equipartition theorem and the Dulong-Petit Law

It's probably about time we discussed what a **degree of freedom** is.

{: .definition}
> For a Hamiltonian with $$ f $$ quadratic terms, the number of **degrees of freedom** is also $$ f $$. 

Informally, we can think about degrees of freedom as the number of directions in which particles are free to move in.

- Ideal monatomic gas: $$ f = 3 $$ (three spatial degrees)
- Ideal diatomic gas: $$ f = 5 $$ (three spatial, two rotational)

Note that for a diatomic gas we do not count rotation about the axis passing through the two atoms, because the moment of inertia about this axis is negligible.

Armed with our new knowledge of degrees of freedom, let's derive the famous **equipartition theorem**. We'll begin by defining a very general Hamiltonian:

$$ H(\boldsymbol{x}, \boldsymbol{p}) = ax^2 + by^2 + cz^2 + rp_x^2 + sp_y^2 + tp_z^2 $$

For an ideal gas, $$ a = b = c = 0 $$, and $$ r = s = t = 1 $$, which is why the ideal gas has three degrees of freedom - it has three quadratic terms in its Hamiltonian. Let's substitute this into our integral for $$ Z_1 $$. We'll derive the general partition function for the canonical ensemble $$ Z_N $$, in much the same way as we did for the ideal gas case.

$$ 
\begin{aligned}
Z_1 &= \frac{1}{h^3} \int_{-\infty}^{+\infty} \int_{-\infty}^{+\infty} \int_{-\infty}^{+\infty} \int_{-\infty}^{+\infty} \int_{-\infty}^{+\infty} \int_{-\infty}^{+\infty} e^{-\frac{\beta}{2m} \left( ax^2 + by^2 + cz^2 + rp_x^2 + sp_y^2 + tp_z^2 \right)} \, dx \, dy \, dz \, dp_x \, dp_y \, dp_z \\
&= \left( \frac{\pi}{\beta h} \right)^3 \frac{1}{abc \, rst}
\end{aligned}
$$

The integration is performed very similarly to before. We use the same relation $$ Z_N = \frac{Z_1^N}{N!} $$, and substitute into the equation $$ U = -\frac{\partial \ln Z}{\partial \beta} $$, to obtain the following result:

$$ U = \frac{1}{2} \cdot 6 \cdot N k_B T $$

Pay close attention to the $$ 6 $$. It came from the fact that we had 6 quadratic terms in our Hamiltonian. It follows that if we have $$ f $$ quadratic terms, the internal energy would be $$ \frac{1}{2} f N k_B T $$. In other words, for each extra degree of freedom $$ f $$, we obtain $$ \frac{1}{2} k_B T $$ energy per particle on average. This is the **equipartition theorem**.

{: .definition}
> The **equipartition function** states that for a system with $$ f $$ degrees of freedom, the total internal energy $$ U $$ is given by:
>
> $$ U = \frac{1}{2} f N k_B T $$

- Ideal monatomic gas: $$ U = \frac{3}{2} N k_B T $$
- Ideal diatomic gas: $$ U = \frac{5}{2} N k_B T $$

We will end with a brief discussion on the **Dulong-Petit Law**. The Dulong-Petit law concerns metallic lattices. In a metallic lattice, we have positively charged ions surrounded by a sea of delocalised electrons which can move freely. We can model the electrons as a gas, with six degrees of freedom (there are six terms in the Hamiltonian, $$x, y, z, p_x, p_y, p_z$$, as there are certainly forces acting on the electrons, unlike an ideal gas). Using the equipartition theorem:

$$ U = 3 N k_B T $$

Noting that $$ C_v $$, the heat capacity at constant volume, is just $$ \frac{\partial{U}}{\partial{T}} $$:

$$ C_v = \frac{\partial{U}}{\partial{T}} = 3 N k_B $$

And the molar heat capacity at constant volume, $$ c_v = \frac{C_v}{n} $$ is:

$$ c_v = \frac{3 N k_B}{n} = \frac{3 n R}{n} = 3R \approx 24.9 \: \text{J} \: \text{mol}^{-1} \: \text{K}^{-1} $$

This suggests that all metals should have the same molar heat capacity at constant volume, which is the Dulong-Petit Law. Unfortunately, the Dulong-Petit Law is generally incorrect, and is only asymptotically true in the limit as $$ T \to \infty $$.

{: .definition}
> The **Dulong-Petit Law** (erroneously) postulates that all metals have $$ c_v = 3R $$.

References
----------
{% bibliography %}































