# Stochastic Approximation and SGD

## Robbins-Monro Algorithm

The Robbins–Monro algorithm is the **original stochastic approximation method**.

Suppose you want to find the mean $\mu$ of a random variable $X$, but you can only draw samples $X_1, X_2, ...$  .

You could use RM to estimate it:

$$
\mu_{t+1}=\mu_{t}+\alpha_t(X_t-\mu_t)
$$

This is exactly the incremental mean update rule:

$$
\mu_{t+1}=\mu_{t}+\frac{1}{t}(X_t-\mu_t)
$$

## Stochastic Gradient Descent

SGD is an optimization algorithm used to minimize a loss function (like ML or DL)

It’s a method for finding the minimum of a function when you can only estimate its gradient from **random samples** of data

We want to minimize a function:

$$
J(\theta)=E_{data}[L(\theta;x)]
$$

Where:

- $J(\theta)$ = expected loss function
- $L(\theta;x)$ = loss on one sample x
- $\theta$ = model parameters

We want to find parameters $\theta^*$ that make $J(\theta)$  as smal as possible

### Step-by-step intuition

Imagine you’re standing on a noisy surface (like a hilly terrain with random bumps) and want to find the lowest point.

- Each gradient step tells you **“which direction goes downhill”**, but it’s based on one random sample — so it’s noisy.
- Over many steps, those noisy gradients **average out**, and you move toward the true minimum.

That’s why it’s *stochastic* (random) but still converges with enough samples.