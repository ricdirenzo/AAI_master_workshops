# Linear regression
In statistics, linear regression is a model that estimates the linear relationship between a scalar response (dependent variable) and one or more explanatory variables (regressor or independent variable).

Simple linear regression is a linear regression model with a single explanatory variable.

## OLS Formulation
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

&nbsp;

### Evaluating model performance
The coefficient of determination, knows as "_R squared_" and denoted $R^2$, is a measure of the goodness of fit of a model, i.e. of how well the regression predictions approximate the real data points. An $R^2$ of 1 indicates that the regression predictions perfectly fit the data. It is defined as the proportion of the variation in the dependent variable that is predictable from the independent variable(s).

$$ R^2 = 1 - \frac{ \sum_{i=1}^{n} {e_i^2} }{ \sum_{i=1}^n {(y_i - \mu_y)^2} } = 1 - \frac{ SS_{\text{res}} }{ SS_{\text{tot}} }$$

where:
- $SS_\text{res}$ is called residual sum of squares
- $SS_\text{tot}$ is called total sum of squares (proportional to the variance of the data)

&nbsp;

### Gauss–Markov assumptions
In order to confirm the statistical significance of the choice of coefficients, it is necessary to make some hypotheses known as Gauss–Markov assumptions:
- $\mathbf{Cov}(X_i,e_j)=0$ (exogeneity of the regressor)
- $\mathbf{E}[e_i]=0$
- $\mathbf{Var}(e_i)=\sigma^2 \lt \infty \quad \forall i$ (homoscedastic)
- $\mathbf{Cov}(e_i,e_j)=0 \quad \forall i \neq j$ (no autocorrelation)

Under these conditions, the ordinary least squares (OLS) estimator of the coefficients of a linear regression model is the best linear unbiased estimator (_BLUE_).

&nbsp;

Often this formulation are written in matrix notation as:

$$ \mathbf{y} = \mathbf{X}b + e $$

where:
- $\mathbf{y}$ is a vector of observed values $y_i (i=1, \dots, n)$ of the variable called the target variable or dependent variable
- $\mathbf{X}$ is a matrix of row-vectors $\mathbf{x}_i$ or of $n$-dimensional column-vectors $\mathbf{x}_j$ which are known as regressors, explanatory variables or independent variables
- $b$ is a $(p+1)$-dimensional dimensional parameter vector, where $b_0$ is the intercept term and, in simple linear regression, $p=1$, $b_1$ is the regression slope
- $e$ is a vector of values $e_i$ or residuals.

The vector of residuals $e$ is given by:

$$ e = \mathbf{y} - \mathbf{X}b $$

We can write the sum of squared residuals as:

$$ e^\top e = (\mathbf{y} - \mathbf{X}b)^\top(\mathbf{y} - \mathbf{X}b) = \mathbf{y}^\top\mathbf{y} - b^\top\mathbf{X}^\top\mathbf{y} - \mathbf{y}^\top\mathbf{X}b + b^\top\mathbf{X}^\top\mathbf{X}b $$

This development uses the fact that $\mathbf{y}^\top\mathbf{X}b = (\mathbf{y}^\top\mathbf{X}b)^\top = b^\top\mathbf{X}^\top\mathbf{y}$

$$ e^\top e = \mathbf{y}^\top\mathbf{y} - 2b^\top\mathbf{X}^\top\mathbf{y} + b^\top\mathbf{X}^\top\mathbf{X}b $$

To find the coefficients that minimizes the sum of squared residuals, we need to take the derivative of $e^\top e$ with respect to $b$

$$ \arg\min_{b} e^\top e = \frac{\partial}{\partial b} (\mathbf{y}^\top\mathbf{y} - 2b^\top\mathbf{X}^\top\mathbf{y} + b^\top\mathbf{X}^\top\mathbf{X}b) = 0 $$

$$ 2\mathbf{X}^\top\mathbf{X}b -2\mathbf{X}^\top\mathbf{y} = 0 $$

$$ \mathbf{X}^\top\mathbf{X}b = \mathbf{X}^\top\mathbf{y} $$

If the inverse of $\mathbf{X}^\top\mathbf{X}$ exists, then pre-multiplyng both sides by the inverse gives us the following equation:

$$ (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{X}b = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y} $$

We know that $(\mathbf{X}^\top\mathbf{X})^{-1} \mathbf{X}^\top\mathbf{X} = \mathbf{I}$ identity matrix. This gives us:

$$ \hat{b}_{OLS} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y} $$

&nbsp;

## Gradient descent algorithm:
The objective is to minimize the cost function (Mean Squared Error) using gradient descent. The cost function is given by:

$$ L(b_0,b_1) = \frac{1}{n} \sum_{i=1}^n {(y^{(i)} - h_b(x^{(i)}))^2} $$

where:
- $h_b(x^{(i)}) = b_0 + b_1x^{(i)}$ is the hypothesis function
- $n$ is the number of training examples.

The gradients for $b_0$ and $b_1$ are computed as follows:

$$ \frac{\partial}{\partial b_0}L(b_0,b_1) = -\frac{2}{n} \sum_{i=1}^n {(h_b(x^{(i)}) - y^{(i)})} $$
$$ \frac{\partial}{\partial b_1}L(b_0,b_1) = -\frac{2}{n} \sum_{i=1}^n {(h_b(x^{(i)}) - y^{(i)})}x^{(i)} $$

At each step, we update $b_0$ and $b_1$ using the following chain rules:

$$ b_0 = b_0 - a\frac{\partial}{\partial b_0}L(b_0,b_1) $$
$$ b_1 = b_1 - a\frac{\partial}{\partial b_1}L(b_0,b_1) $$

where a is the learning rate.
