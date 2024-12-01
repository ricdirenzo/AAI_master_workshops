# Linear regression
In statistics, linear regression is a model that estimates the linear relationship between a scalar response (dependent variable) and one or more explanatory variables (regressor or independent variable).

## Simple linear regression
Simple linear regression is a linear regression model with a single explanatory variable.

### Formulation
We consider the following linear model

$$ y = b_0 + b_1x + e $$

where
- $b_1$ is the slope of the line
- $b_0$ is the y intercept
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
  \end{cases} $$
  
$$ = \begin{cases}
    -2 \sum_{i=1}^n {y_i-b_0-b_1x_i} = 0\\
    -2 \sum_{i=1}^n {(y_i-b_0-b_1x_i)x_i} = 0 
  \end{cases} $$

from which the following solutions are derived:

$$ \hat{b_1} = \frac{ \sum_{i=1}^n {x_iy_i} - \frac{1}{n} \sum_{i=1}^n {x_i}\sum_{i=1}^n {y_i} }{ \sum_{i=1}^n {x_i^2} - \frac{1}{n} (\sum_{i=1}^n {x_i})^2 } \qquad \hat{b_0} = \frac{1}{n} \sum_{i=1}^n {y_i} - \frac{\hat{b_1}}{n} \sum_{i=1}^n {x_i}$$

i.e. 

$$ \hat{b_1} = \frac{ \sigma_{x,y} }{ \sigma_{x}^2 } \qquad \hat{b_0} = \mu_y - \hat{b_1} \mu_x $$

In order to confirm the statistical significance of the choice of coefficients, it is necessary to make some hypotheses known as Gauss–Markov assumptions:
- $\\{e_1, \dots, e_n\\}$ and $\\{x_1, \dots, x_n\\}$ are independent
- $\text{E}[e_i]=0$
- $\text{Var}(e_i)=\sigma^2 \lt \infty \quad \forall i$ (homoscedastic)
- $\text{Cov}(e_i,e_j)=0 \quad \forall i \neq j$

Under these conditions, the ordinary least squares (OLS) estimator of the coefficients of a linear regression model is the best linear unbiased estimator (_BLUE_).

Often this formulation are written in matrix notation as:

$$ \textbf{y} = \textbf{X}b + e $$

where:
- $\textbf{y}$ is a vector of observed values $y_i (i=1, \dots, n)$ of the variable called the target variable or dependent variable
- $\textbf{X}$ is a matrix of row-vectors $\textbf{x}_i$ or of $n$-dimensional column-vectors $\textbf{x}_j$ which are known as regressors, explanatory variables or independent variables
- $b$ is a $(p+1)$-dimensional dimensional parameter vector, where $b_0$ is the intercept term and, in simple linear regression, $p=1$, $b_1$ is the regression slope
- $e$ is a vector of values $e_i$ or residuals.

The vector of residuals $e$ is given by:

$$ e = \textbf{y} - \textbf{X}b $$

We can write the sum of squared residuals as:

$$ e^Te = (\textbf{y} - \textbf{X}b)^T(\textbf{y} - \textbf{X}b) = \textbf{y}^T\textbf{y} - b^T\textbf{X}^T\textbf{y} - \textbf{y}^T\textbf{X}b + b^T\textbf{X}^T\textbf{X}b $$

This development uses the fact that $\textbf{y}^T\textbf{X}b = (\textbf{y}^T\textbf{X}b)^T = b^T\textbf{X}^T\textbf{y}$

$$ e^Te = \textbf{y}^T\textbf{y} - 2b^T\textbf{X}^T\textbf{y} + b^T\textbf{X}^T\textbf{X}b $$

To find the coefficients that minimizes the sum of squared residuals, we need to take the derivative of $e^Te$ with respect to $b$

$$ \arg\min_{b} e^Te = \frac{\partial}{\partial b} (\textbf{y}^T\textbf{y} - 2b^T\textbf{X}^T\textbf{y} + b^T\textbf{X}^T\textbf{X}b) = 0 $$

$$ 2\textbf{X}^T\textbf{X}b -2\textbf{X}^T\textbf{y} = 0 $$

$$ \textbf{X}^T\textbf{X}b = \textbf{X}^T\textbf{y} $$

If the inverse of $\textbf{X}^T\textbf{X}$ exists, then pre-multiplyng both sides by the inverse gives us the following equation:

$$ (\textbf{X}^T\textbf{X})^{-1}\textbf{X}^T\textbf{X}b = (\textbf{X}^T\textbf{X})^{-1}\textbf{X}^T\textbf{y} $$

We know that $(\textbf{X}^T\textbf{X})^{-1} \textbf{X}^T\textbf{X} = \textbf{I}$ identity matrix. This gives us:

$$ \hat{b} = (\textbf{X}^T\textbf{X})^{-1}\textbf{X}^T\textbf{y} $$
