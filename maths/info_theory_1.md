---
title: 'Information and entropy'
layout: post
parent: Mathematics
nav-order: 1
scholar:
  style: apa
  locale: en

  sort_by: none
  order: ascending

  source: ./maths
  bibliography: info_theory_1_ref.bib
  bibliography_template: "{{reference}}"

  replace_strings: true
  join_strings:    true

  details_dir:    bibliography
  details_layout: bibtex.html
  details_link:   Details

  query: "@*"
---

# Information and entropy
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

This series of articles discusses the mathematical field of information theory and Bayesian inference. Information theory is the study of how data can be encoded and decoded, and compressed and decompressed. The primary goal of information theory, as set out by Claude Shannon, is *reliable communication across an unreliable channel*. Throughout these articles, we will discuss the minimum length to which data can be compressed, as well as the minimum amount of redundancy we need to add to data to encode it in such a way that it can be decoded with no error.

*Note: throughout these articles, $$ \log(x) $$ refers to $$ \log_2(x) $$.*

## Information

Suppose we have an event $$ x_i $$ belonging to a probability distribution $$ P(X) $$. Then the **Shannon information content** of that event is:

$$ h(x_i) = \log\left(\frac{1}{p(x_i)}\right) $$

For example, suppose that the weather is rainy with probability $$ 0.5 $$, and one day you step outside and you find that it is raining. Then the information gain from this event is: $$ \log_2{2} = 1 \text{bit}$$. We gain one bit of information each time we halve our sample space of all possible sequences of events. If it rained with probability $$ 0.25 $$, then the information gain would be $$ 2 $$ bits, because we have cut our sample space into half twice. Intuitively, the information gain of an event $$ x_i $$ measures our *level of surprise* at that event happening. If we knew with certainty it was going to rain, then the information gain would be $$ \log_2{1} = 0 \text{bits} $$, because we were completely unsurprised - we gained no new information from seeing it rain because we knew it was going to rain anyway!

## Entropy

The entropy of a probability distribution, $$ H(X) $$, is just the expected information gain:

$$ H(X) = \sum_i{p(x_i) \log\left(\frac{1}{p(x_i)}\right)} $$

You can think of the entropy as our "average surprise" when we sample from this distribution, or alternatively, you could say it is the average amount of information we lack (if the expected information gain is high, we must not have known much about the distribution before). If one event has probability $$ 1 $$, whilst the others all occur with probability $$ 0 $$, then we have no information to gain, because we know exactly what will happen. In this case, the entropy is $$ 0 $$.

The entropy is maximised when $$ X $$ is distributed uniformly. This corresponds to us having maximum ignorance / maximum lack of information about the distribution, because we have no reason to believe a priori that one event will happen over another. The maximum entropy is:

$$ H_{max}(X) = \log(N) $$

where N is the number of events in the distribution.

We can also define the entropy of a joint distribution $$ p(X, Y) $$:

$$
H(X, Y) = -\sum_{x \in X} \sum_{y \in Y} p(x, y) \log p(x, y)
$$

Using the Bayes' rule:

$$
\begin{align*}
P(X, Y) &= P(X | Y) P(Y) \\\\
-\log P(X, Y) &= -\log(P(X|Y)) - \log P(Y) \\\\
\end{align*}
$$

Taking expectations on both sides,

$$
H(X, Y) = H(X | Y) + H(Y)
$$

We could find that $$ H(X, Y) = H(Y \vert X) + H(X) $$ by similar reasoning.

## Data compression with the bent coin

So far, we have presented formulae for the information content and entropy of a distribution, without showing why they are useful or indeed quantities we should care about in the first place. Let's take a look at a toy problem: the bent coin. This is a coin with the following distribution:

$$
\begin{aligned}
&P(\text{Head (1)}) = p \\
&P(\text{Tail (0)}) = 1 - p
\end{aligned}
$$

For concreteness, suppose $$ p = 0.1 $$, so that $$ 1 - p = 0.9 $$. The bent coin is flipped $$ N = 1000 $$ times, to generate an N-digit lottery number. We would like to know the answer to the following question: *how many lottery tickets must we buy to be 99% sure of winning?*

We might begin by asking ourselves: what is the most probable lottery ticket? And the answer to that would be simple - it would be the ticket with all 0s (all tails), which occurs with probability $$ 0.9^N = 1.7 \times 10^{-46}$$. However, if I asked you, *what is the expected number of 1s in the ticket*, then you would say $$ p N $$ 1s, which gives an expected 100 1s.

A sensible strategy would be to first buy all the tickets containing all 1s, then those containing one 1, then two 1s, and so on, until we reach the tickets containing 100 1s. We might also wish to buy tickets containing 101 1s, just to be safe. Because the distribution of this coin is binomial, and we have a large $$ N $$, we can use a normal approximation to this distribution, such that:

$$
X \sim \mathcal{N}(Np, Np(1-p))
$$

Knowing that the standard deviation of this distribution is $$ \sqrt{Np(1-p)} $$, we could buy tickets with up to $$ \mu + 2\sigma $$ 1s, which would give us approximately $$ 99\% $$ of the distribution. In our case, $$ \mu = 100 $$, and $$ \sigma \approx 10 $$, so we should buy tickets with up to $$ 120 $$ 1s. This quantity can be expressed as:

$$\sum_{k=0}^{120} \binom{1000}{k}$$

The dominant term in this sum is $$ \binom{1000}{120} $$. We can apply Stirling's approximation:

$$ \binom{N}{r} \approx 2^{N H_2(\frac{N}{r})} $$

where $$ H_2(X) $$ is the binary entropy of the distribution of X:

$$ H_2(p) = p \log \left(\frac{1}{p} \right) + (1-p) \log \left(\frac{1}{1-p}\right) $$

Hence the number of tickets we need to buy, $$ \binom{1000}{120} $$, is approximately:

$$ n = \binom{1000}{120} \approx 2^{N H(X)} \approx 2^{470} $$

These $$ n = 2^{470} $$ tickets form what is known as the **typical set** of this distribution - these are the tickets we "expect" to see when we sample from the bent coin. $$ 2^{470} $$ tickets implies that we can encode each ticket using only $$ 470 $$ bits instead of $$ N = 1000 $$ like we originally had. This provides some evidence that the entropy of a distribution is related to the minimum number of bits a sequence can be compressed to. In particular, for a sequence of N outcomes, being sampled from a probability distribution $$ X $$, Shannon proved that the compression limit is:

$$ \text{compression limit} = N H(X) \text{bits} $$

## Encoding and redundancy

When sending data over a channel, some noise $$ \mathbf{n} $$ is added to that data, which distorts it. We would like to formulate an encoding and decoding scheme, to minimise the probability of error in transmission. During encoding, we will add **redundancy** to the data, and during decoding, we will exploit this redundancy to detect where errors have occurred in transmission and correct them.

Let's start by considering a very simple channel: the **binary symmetric channel**. This is a channel that flips a bit with probability $$ p $$, and keeps the bit the same with probability $$ 1-p $$.

Suppose we generated the following sequence $$ \mathbf{s} = 011 $$, and sent this to our friend with no encoding scheme. Suppose that due to noise $$ \mathbf{n} $$, the received message $$ \mathbf{r} = 111 $$, because one of the bits were flipped. Our friend will get the wrong message! One way to circumvent this is to use a **repetition code**. This repeats each bit $$ N $$ times, so if $$ N = 3 $$, our message $$ \mathbf{s} = 000111111 $$. Suppose it is received as $$ \mathbf{r} = 010110011 $$. By splitting the string into groups of $$ N $$, we get $$ 010 110 011 $$, and by taking the majority vote in each block, we decode the message as $$ 011 $$, which was our original message. 

Repetition codes can reduce our value of $$ f $$ (probability of error in a string), but they come at the cost of increased redundancy. We define the **capacity** of the channel as the number of bits of useful information sent per bit. For example, our repetition code sends 1 bit of information for every 3 bits sent, so the capacity $$ C = \frac{1}{3} $$.

Shannon proved in his **noise channel coding theorem** that you can achieve an arbitrarily low error $$ f $$ at a finite capacity $$ C $$. This is a remarkable result, because common sense tells us that you should need your capacity to tend towards zero for the error to tend towards zero, because you would need some infinitely huge repetition code to guarantee no error. But Shannon tells us that you can get zero error with a non-infinite $$ C $$! We will come back to this later and prove this result in a later article.

## Conclusion

This article introduced the concepts of information and entropy of a probability distribution. We explored the limits of data compression and channel capacity, bringing us closer towards the goal of *reliable communication along an unreliable channel*. Here's a summary of the key points:

- The information content of an event $$ i $$ is $$ \log(\frac{1}{p_i}) $$.
- The entropy $$ H(X) $$ is the expected information content of a distribution.
- The minimum number of bits to which a sequence can be compressed is $$ N H(X) $$.
- Redundancy can be added in the form of repetition codes to reduce the chance of error in transmission.
- There exists a finite channel capacity $$ C $$ at which $$ f = 0 $$.

References
----------
{% bibliography %}































