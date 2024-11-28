# Regression task
Regression is a supervised machine learning method where the model tries to predict the state of an outcome variable with the help of other correlated independent variables. A regression task, unlike the classification task, outputs continuous values within a given range.

## Linear approach
In statistics, linear regression tries to model the relationship between a dependent variable and one or more independent variables. Let X be the independent variable and Y be the dependent variable. The linear relationship between these two variables can be defined as follows:

$$Y ~ mX + c + u$$

where
- $m$ is the slope of the line
- $c$ is the y intercept
- $u$ is the error term

We will use this equation to train our model with a given dataset and predict the value of $Y$ for any given value of $X$. Our task is to determine the value of $m$ and $c$, such that the line corresponding to those values is the best fitting line or gives the minimum error.

&nbsp;

## Loss Function
The loss is the error in our predicted value of $m$ and $c$. Our goal is to minimize this error to obtain the most accurate value of $m$ and $c$. We will use the Mean Squared Error (MSE) function to calculate the loss. 

$$MSE = \frac{1}{n} \sum_{i=0}^{n} (y_i - (mx_i+c))^2$$

where $y_i$ is the actual value and $(mx_i+c)$ is the predicted value.
