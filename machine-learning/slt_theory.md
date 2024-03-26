---
title: 'Singular learning theory'
layout: post
parent: Machine learning
nav-order: 4
scholar:
  style: apa
  locale: en

  sort_by: none
  order: ascending

  source: ./machine-learning
  bibliography: slt_theory_ref.bib
  bibliography_template: "{{reference}}"

  replace_strings: true
  join_strings:    true

  details_dir:    bibliography
  details_layout: bibtex.html
  details_link:   Details

  query: "@*"
---

# A tour of singular learning theory (SLT)
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

Singular learning theory is a novel field of neural network research based on work by [Sumio Watanabe](https://scholar.google.com/citations?user=_KUAdxcAAAAJ&hl=en) on singular statistical models. It provides tools for analysing the effective dimensionality of models, proposing that optimisers such as SGD converge towards points called *degenerate singularities*, where the effective dimensionality is reduced. The quantity of key interest in SLT is the RLCT, denoted by $$ \lambda $$, which is a measure of the model effective dimensionality at a minima converged to by an optimiser.

## An introduction to the Bayesian setting

We begin our exploration of SLT in the Bayesian paradigm, where we have some set of samples $$ D_n $$ that are all independent of each other:

$$D_n = \{(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)\}$$

We also suppose there is some true distribution $$ q(y, x) = q(x) q(y \vert x) $$, such that we know $$ q(x) $$, the data generating process, but not $$ q(y \vert x) $$, the probability of an output $$y$$ given some input data $$x$$.

We produce a *model* for the true distribution, called $$ p(y \vert x, w) $$ defined by a vector $$\textbf{w}$$ of parameters belonging to the space $$W \subseteq \mathbb{R}^d$$.

In the Bayesian setting, we have some initial assumption about the probability distribution over weights, $$ \phi(w) $$, known as the *prior distribution*.

We take the cross entropy loss as our loss function, such that the loss is:

$$\mathcal{L}(w) = -\int_x\int_y q(y,x)\log p(y \vert x,w)\,dy\,dx$$

Given that $$ q(y, x) = 1 $$ for the true mapping, and $$ 0 $$ otherwise, the empirical loss reduces to the *negative log likelihood (NLL)* of $$ p(y \vert x, w) $$:

$$\mathcal{L}(w) = -\sum_{i=1}^{N} \log p(y_i \vert x_i,w)$$

It follows that the model likelihood, $$ p(D_n  \vert  w) $$ is given by:

$$\prod_{i=1}^{n} p(y_i \vert x_i,w) = e^{-nL_n(w)}$$

Using Bayes' theorem:

$$
\begin{aligned}
p(w \vert D_n) &= \frac{p(D_n \vert w)p(w)}{p(D_n)} \\
&= \frac{e^{-nL_n(w)}\phi(w)}{\int \phi(w)e^{-nL_n(w)}dw}
\end{aligned}
$$

The denominator is given a special name, $$ Z_n $$, and is known as the *partition function*. It is essentially a normalising constant, and is equal to $$ p(D_n) $$, the probability of observing the set of data, given $$ q(x) $$.

We now borrow a concept from physics known as the free energy of a system, which in thermodynamics involves a balance between maximising entropy, and minimising enthalpy. The Gibbs free energy {% cite Gibbs1878OnTE %} is given by the following equation:

$$G = H - TS$$

Where:
- $$G$$ is the Gibbs free energy
- $$H$$ is the enthalpy of the system, given by:
  $$H = U + PV$$
  - $$U$$ is the internal energy of the system
  - $$P$$ is the pressure
  - $$V$$ is the volume
- $$T$$ is the absolute temperature
- $$S$$ is the entropy of the system

For a thermodynamic system to be in equilbrium with its surroundings, its Gibbs free energy must be minimised. The free energy can be minimised either by reducing the system's enthalpy, or by maximising its entropy.

Moving back to our statistical setting, we define the statistical free energy:

$$ F_n = - \log(Z_n) $$

Consider a region of space in $$W$$ that has a high model likelihood, $$p(D_n  \vert  w)$$. Then we say that the *posterior density* is large, because we obtain a lot of information about which weights minimise the loss most in this region. If the likelihood is large in this region, then $$Z_n$$ will also be large. Hence, *by minimising the free energy $$F_n$$*, we maximise $$Z_n$$, which increases our information about which weights are most optimal. {% cite LessWrongDSLT1 %}

We will now briefly define the *Kullback-Leibler divergence* $$K(w)$$, which measures the error between the true distribution $$q(y, x)$$, and our model $$p(y \vert x, w)$$:

$$\begin{aligned}
D_\text{KL}(q(y, x) \ \vert  p(y \vert x, w)) &= \int_{x}\int_{y} q(y, x) \log\left(\frac{q(y, x)}{p(y \vert x, w)}\right) dy dx \\
&= \int_{x}\int_{y} q(y, x) \log\left(\frac{q(y \vert x)q(x)}{p(y \vert x, w)}\right) dy dx \\
&= \int_{x} q(x) \int_{y} q(y \vert x) \log\left(\frac{q(y \vert x)}{p(y \vert x, w)}\right) dy dx \\
&= \mathbb{E}_{(x, y) \sim q(x, y)}\left[\log\left(\frac{q(y \vert x)}{p(y \vert x, w)}\right)\right]
\end{aligned}$$

The KL divergence is thus the expected logarithmic "distance" between the truth and the model. It is the metric we will use to study the loss landscape in the Bayesian setting.

We define the set $$W_0$$, the set of *true parameters*, where $$K(w) = 0$$, or equivalently, $$p(y \vert x, w) = q(y \vert x)$$. Furthermore, we say that the model is *realisable*, if the set $$W_0$$ is non-empty.

## Singularity of models

One of the main ideas explored in SLT is how we can find a better metric for measuring the effective dimensionality of a statistical model. A naive way of doing this would just be to count the number of parameters, but numerous studies have shown that models do not use all of their parameters meaningfully - only a handful of parameters are important. The answer to this question lies in the *Fisher Information Matrix*, denoted as $$I(w)$$.

$$\begin{aligned}
{I}(w) = \mathbb{E}_{(x, y) \sim q(x, y)}\left[\nabla_w \log p(y \vert x, w) \nabla_w \log p(y \vert x, w)^\top\right] \\
\end{aligned}$$

A model is singular if the set of points for which $${I(w)}$$ is non-invertable (i.e. $$\det(I(w)) = 0$$), is non-empty. Otherwise, the model is regular. 

Points at which the Fisher Information Matrix has zero determinant are referred to as *degenerate singularities*. These are points which have a lower effective dimension because not all the parameters are being meaningfully utilised. At a true parameter (parameter for which $$K(w) = 0$$), it can be shown that $$I(w) = H(w)$$, where $$H(w)$$ is the Hessian of the model:

$$H(w) = \nabla_w^2 L_n(w)$$

The Hessian is the matrix of all second derivatives of the loss function $$L_n$$ with respect to the model parameters $$w$$. Intuitively, the elements of the Hessian represent the "curvature" of the loss landscape with respect to those parameters. Suppose we are at a degenerate singularity. Then $$\det(I(w)) = 0$$, implying that $$\text{rank}(I(w)) < d$$, where d is the number of model parameters, which implies that some of the eigenvalues of $$I(w)$$, and thus $$H(w)$$ at a singularity, are equal to zero. This means that the model loss does not "curve" or "bend" with respect to some of the parameters, implying that the minima is more "flat" compared to a regular minima (as the loss is much less sensitive to change in some of the dimensions!). {% cite LessWrongDSLT1 %}

## The Real Log Canoncial Threshold (RLCT)

The RLCT is in many ways the diamond of SLT - it provides us a metric for the effective dimensionality of a model, and allows us to make claims about the complexity of models, and how this relates to a model's ability to generalise.

The RLCT is best understood from the perspective of *volume scaling*. Suppose we are at a minima somewhere in the loss landscape. Now, suppose that we cut off a section of a minima (so we have a cup shape of sorts), such that $$ K(w) \le \epsilon $$ everywhere in this region, for some $$ \epsilon $$. Then, we can define a volume scaling function $$ V(\epsilon) $$:

$$ V(\epsilon) \propto \epsilon^{\lambda} $$

where $$ \lambda $$ is the RLCT. {% cite Watanabe2018SPD %}

Because $$ \epsilon $$ is typically much smaller than one, a low $$ \lambda $$ implies that $$ V(\epsilon) $$ grows quickly with increasing $$ \lambda $$. Conversely, a large $$ \lambda $$ implies that $$ V(\epsilon) $$ grows slowly. Intuitively, the former case corresponds to loss functions where $$ K(w) \propto w^4 $$, whereas the latter case corresponds to cases where $$ K(w) \propto w^2 $$, say. $$ K(w) \propto w^4 $$ has a volume that rises more steeply near zero, because it is more "flat" than $$ K(w) \propto w^2 $$. The following figure from *"Estimating the Local Learning Coefficient at Scale"*, by Furman and Lau, illustrates this principle of "available volume" very well:

![volume-scaling](https://i.ibb.co/Cz6ZC19/volume-scaling.png)

The blue liquid shows how the volume scales as you move up the minima loss landscape. For the upper model, which is paraboloid, the volume increases slowly. The lower model is much more singular, and therefore has a volume that increases at a much faster rate.

Recall that a degenerate singularity has a non-invertable $$ I(w) $$. The RLCT, $$\lambda$$, measures the degree to which a model is singular - a low RLCT implies high volume scaling, which implies a more "flat" minima, so the model is more singular. 

The three key cases for the RLCT are as follows:

- **regular** : $$ \lambda = \frac{d}{2} $$
- **minimally singular** : $$ \lambda = \frac{\text{rank}(H(w))}{2} $$
- **singular** : $$ \lambda \ge \frac{\text{rank}(H(w))}{2} $$

Notice the interesting case of a *minimally singular model*. This is a model that has converged to a degenerate singularity simply by virtue of not using all of its parameters. This is the most obvious kind of way a model can reduce its dimensonality, by having $$ r < d $$ dimensions be non-free parameters, and the rest of the $$ d - r $$ dimensions be free parameters.

For a minimally singular model, the following can be shown:

$$ L(\textbf{w}) = \sum_{i}^{r} w_i^2, \text{ where } r < d  \text{  (num. of parameters)  } $$

In other words, if the loss can be written as a sum of squares of some of the parameters in the local region around the minima, but not all the parameters, then the model is minimally singular. 

## Free energy and the information criterion

We previously introduced $$ F_n = -\log(Z_n) $$, the free energy. For a regular model, it can be shown that:

$$ F_n = \text{BIC} = n L_n(w) + \frac{d}{2} \log(n) $$

It turns out that this is also the expression for the Bayesian Information Criterion (BIC) {% cite Watanabe2018IC %}. This is a model selection criterion used in statistics, and models with a lower BIC are preferred over those with a higher BIC. Minimising the BIC is equivalent to minimising the loss, whilst also trying to minimise the parameter count $$ d $$. For singular statistical models, it can be shown that the expression for free energy is the following {% cite Watanabe2018IC %}:

$$ F_n = \text{WBIC} = n L_n(w) + \lambda \log(n) $$

This is the expression for Watanabe's Widely Applicable Bayesian Information Criterion (WBIC), which uses the RLCT in place of $$ \frac{d}{2} $$, as we previously established that for singular models, the correct measure of model dimensonality is $$ \lambda $$. Note that for a regular model, $$ \lambda = \frac{d}{2} $$, and so the effective dimensionality of a model is given by $$ 2 \lambda $$.

Supposing we have some optimiser that tries to *minimise $$ F_n $$*, then we observe that there is an accuracy-complexity tradeoff occurring here - the optimiser wants to minimise the loss, but also minimise $$ \lambda $$, the model complexity. This tradeoff has profound consequences, because it implies that there can be phase transitions between different minima. The model might initially converge towards one minima, and then jump to a different minima because it has a lower value of $$ \lambda $$.

It's also worth noting that in the limit as $$n \to \infty$$, the $$ \lambda \log(n) $$ term becomes dominated by the $$ n L_n(w) $$ term, so for large n, accuracy is always preferred over complexity. Phase transitions are therefore more common in the low $$ n $$ regime, where the $$ \log(n) $$ term is more dominant. 

We can also define the *generalisation loss*, $$ G(n) $$, which is calculated by taking the derivative of the WBIC with respect to n:

$$ G(n) = L_n(w) + \frac{\lambda}{n} $$

This is a better measure of model generlisation than just loss alone, because it takes into account overfitting of the model. {% cite Watanabe2018IC %}

## Conclusion

In this article, we discussed the theoretical basis of SLT, which has the following key results:

- Stochastic gradient descent converges towards degenerate singularities, where $$ \det(I(w)) = 0 $$.
- The RLCT $$ \lambda $$ measures the local complexity of a minima, and is defined using the volume scaling function $$ V(\epsilon) \propto \epsilon^{\lambda} $$, measuring the flatness of the minima.
- A point that is more singular has a flatter minima and thus a lower RLCT.
- SLT hypothesises that SGD implicitly minimises the free energy, which has an accuracy ($$ n L_n(w) $$) complexity ($$ \lambda \log(n) $$) tradeoff. This implies that phase transitions can occur between minima with different local complexities.

Singular learning theory is the framework in which *developmental interpretability* research is conducted. This is a field of research which analyses how and why neural networks generalise, and provides explanations in terms of the RLCT $$ \lambda $$, for phenomenon such as grokking, and polysemanticity. To get started with developmental interpretability research, see the [devinterp](https://devinterp.com/) webpage, which contains a list of unstarted research projects along with their difficulty levels. Also see the [devinterp Python library](https://github.com/timaeus-research/devinterp) which provides useful tools for estimating the RLCT and free energy of a model.

References
----------
{% bibliography %}


















