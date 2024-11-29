# Linear regression
In statistics, linear regression is a model that estimates the linear relationship between a scalar response (dependent variable) and one or more explanatory variables (regressor or independent variable).

## Simple linear regression
Simple linear regression is a linear regression model with a single explanatory variable.

### Formulation
We consider the following linear model

$$ y = b_0 + b_1x + e $$

where
- $m$ is the slope of the line
- $c$ is the y intercept
- $e$ is the error term (residuals)

Our goal is to find estimated values $\hat{b_0}$ and $\hat{b_1}$ for the parameters $b_0$ and $b_1$ which would provide the best fit for the data points.

Suppose we observe $n$ data pairs and call them $\\{(xi, yi), i=1, \dots, n\\}$. We can describe the underlying relationship between $y_i$ and $x_i$ involving this error term $e_i$ by:

$$ y_i = b_0 + b_1x_i + e_i $$

The coefficients $b_0$ and $b_1$ are estimated according to the Ordinary Least Squares (OLS) method to obtain a line that minimizes the sum of squared residuals.

$$ d(b_0,b_1) = \sum_{i=1}^n e_i^2 = \sum_{i=1}^n {(y_i-b_0-b_1x_i)^2} $$

Estimates are obtained by solving:

$$ \arg\min_{b_0,b_1} d(b_0,b_1) =
  \begin{cases}
    \frac{\partial}{\partial b_0}d(b_0,b_1) = 0\\
    \frac{\partial}{\partial b_1}d(b_0,b_1) = 0 
  \end{cases} = 
  \begin{cases}
    -2 \sum_{i=1}^n {y_i-b_0-b_1x_i} = 0\\
    -2 \sum_{i=1}^n {y_i-b_0-b_1x_i}x_i = 0 
  \end{cases} $$

from which the following solutions are derived:

$$ \hat{b_1} = \frac{\sigma_{xy}}{\sigma_x^2} \qquad \hat{b_0} = \mu_y-b_1\mu_x$$

In order to confirm the statistical significance of the choice of coefficients, it is necessary to make some hypotheses known as Gauss–Markov assumptions:
- $\\{e_1, \dots, e_n\\}$ and $\\{x_1, \dots, x_n\\}$ are independent
- $\text{E}[e_i]=0$
- $\text{Var}(e_i)=\sigma^2 \lt \infty \quad \forall i$ (homoscedastic)
- $\text{Cov}(e_i,e_j)=0 \quad \forall i \neq j$
