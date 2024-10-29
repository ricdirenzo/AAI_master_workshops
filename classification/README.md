# Classification task
Classification is a supervised machine learning method where the model tries to predict the correct label of a given input data. The model is trained using the training data, and then it is evaluated on test data, before being used to perform prediction on new unseen data.

We'll deal with the classification task, implementing its simplest form known as binary classification which returns two discrete classes:
- 1 ("yes") the positive class
- 0 ("not") the negative class or.

A technique for solving a binary classification task is logistic regression.

## Logistic regression
Logistic regression provides a probabilistic framework for understanding and predicting binary outcomes. It computes a weighted sum of the input features (as the linear regression does) and estimates the probability that outcome belongs to a specific class (instead of returning a continuous value).

$$h(x) = g(\theta_0 + \theta_1x_1 + \theta_2x_2 + \dotsc + \theta_nx_n)$$

where
- $h(x)$ is the hypothesis function
- $x_1, x_2, \dotsc, x_n$ are the independent variables
- weights $\theta_0, \theta_1, \theta_2, \dotsc, \theta_n$ are the parameters of the model
- $n$ is the number of features

its vectorized form is defined as follow:

$$h(\textbf{x}) = f(\theta^T \textbf{x})$$

$g(\cdot)$ is the logistic function that activates the hypothesis function to outcome in a number between 0 and 1 and defined as follow:

$$g(u) := \frac{1}{1+e^{-u}}$$
