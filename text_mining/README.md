# Text mining
**Text mining (or text data mining)**, in Natural Language Processing (NLP), is the process of extracting useful information and patterns from unstructured text data. It involves analyzing text documents to identify meaningful data, such as topics, or more.

## Text data pre-processing
Data pre-processing is a crucial step when working with text data. It involves **preparing and cleaning the data** to ensure that it is in a workable format for model training or analysis. Here’s a list of main steps.
- **Text cleaning**, to removing special characters like symbols, punctuation marks, etc.
- **Tokenization**, to breaking the text into single units of it (token).
- **Stopwords removal**, to eliminating common words that don’t carry significant meaning (g.e. "_the_," "_is_," "_and_," etc.)
- **Lemmatization**, to transform words to their base form (e.g., "makes" becomes "make").
- **Stemming**, to reduce words to their root form, by removing suffixes (e.g., "running" becomes "run").

&nbsp; 

---

## About dataset
The dataset contains a list of over four hundred thousand news headlines published over a period of 7 years, from 2014-01-01 to 2021-12-31. These data sourced from the reputable Australian news source ABC (Australian Broadcasting Corporation).

&nbsp; 

---

## TF-IDF statistic 
**TF-IDF** is a used statistical method in text mining and information retrieval. It helps to evaluate how important a term (word) is to a document in a collection. It is determined based on two factors: **Term Frequency (TF)** and **Inverse Document Frequency (IDF)**.

1. The **Term Frequency (TF)** measures how frequently a term appears in a document. The idea is that the more times a term appears in a document, the more relevant it is to that document. It can be defined as:

    <p align='center'>$$\text{TF}(t, d) = \Large \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}$$</p>
    
    where:
    - $t$ is the term
    - $d$ is the document
    <br /><br />
2. The **Inverse Document Frequency (IDF)** measures how important a term is across the entire collection of documents. It helps to reduce the weight of terms that appear frequently across many documents and gives more importance to terms that are rare. It can be expressed as:

    <p align='center'>$$\text{IDF}(t, D) = \log\left(\Large \frac{\text{Total number of documents } |D|}{\text{Number of documents containing the term } t}\right)$$</p>
    
    where:
    - $|D|$ is the total number of documents in the collection
    - the logarithm is used to scale the IDF value, preventing it from growing too large.
   <br /><br />
   
   The more documents a term appears in, the lower its IDF value becomes. If a term is very rare, it will have a higher IDF, which increases its overall importance.

The **TF-IDF** score of a term $t$ in a document $d$ is the product of the Term Frequency (TF) and the Inverse Document Frequency (IDF):

<p align='center'>$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$</p>

&nbsp; 

---

## Latent Dirichlet Allocation (LDA)

**Latent Dirichlet Allocation (LDA)** is a probabilistic generative model used to discover topics in a collection of text documents. It assumes that each document is a mixture of topics, and each topic is characterized by a distribution over words. It is used in natural language processing and information retrieval for tasks like topic modeling.

### Idea
The model assumes that:

- each document is a mixture of topics, where each topic is a distribution over words;
- the words in documents are drawn from the topics, which are not observed.

The model infers the latent structure of the topics and their word distributions based on the observed words in the documents.

### Key elements
- **$D$** is the total number of documents.
- **$N_d$** is the number of words in document $d$.
- **$K$** is the number of topics (fixed beforehand).
- **$\alpha$** is a hyperparameter for the Dirichlet distribution over topics for each document.
- **$\beta$** is a hyperparameter for the Dirichlet distribution over words for each topic.

### The generative process of the model
The model assumes that each document is generated in the following way:
1. For each document $d$:
   - sample a topic distribution $\theta_d$ from a Dirichlet distribution with parameter $\alpha$
   <p align='center'>$$\theta_d \sim \text{Dirichlet}(\alpha)$$</p>

3. For each word $w_{d,n}$ in document $d$:
   - sample a topic $t_{d,n}$ from the topic distribution $\theta_d$
   - sample a word $w_{d,n}$ from the word distribution $\phi_{t_{d,n}}$
   <p align='center'>$$\phi_k \sim \text{Dirichlet}(\beta)$$</p>

### The variables of the model
- **$w_{d,n}$** is the $n$-th word (_token_) in document $d$
- **$t_{d,n}$** is the topic assignment for the $n$-th word in document $d$
- **$\theta_d$** is the topic distribution for document $d$
- **$\phi_k$** is the word distribution for topic $k$

### The goal of the model
The goal is to infer the latent structure of topics in the documents based on the observed words. Specifically, we want the model to learn:
1. the **topic distribution for each document** $\theta_d$
2. the **word distribution for each topic** $\phi_k$
3. the **topic assignment for each word** $t_{d,n}$

### The likelihood function of the model

#### De Finetti's representation (DFr)
If a finite set of random independent variables $\{X_1, X_2, \dots, X_n\}$ is exchangeable, then the joint probability has a representation as:
<p align='center'>$$P(X_1, X_2, \dots, X_n|\theta) = \int_{\theta}{ \left(\prod_{i=1}^N {P(X_i|\theta)} \right)P(\theta)} d\theta$$</p>

where:
- $X_i$ are conditionally independent
- $P(\theta)$ is the prior over a latent parameter.

In LDA, we assume that observed words $W$ are generated by latent topics (by conditional distributions) and that those topics are exchangeable within a document. By de Finetti’s representation, the probability of a sequence of words and topics has the form:
<p align='center'>$$P(W,T|\theta_d) = \int_{\theta_d} { \left(\prod_{n=1}^{N_d} {P(w_n | t_n) P(t_n | \theta_d)} \right)P(\theta_d)} d\theta_d$$</p>

By marginalizing over the hidden topic variable $t_{d,n}$, however, we can understand LDA as a two-level model.

<p align='center'>$$P(W|\alpha,\beta) = \int_{\theta_d} \left( \prod_{n=1}^{N_d} \sum_{t_{d,n}=1}^K P(w_{d,n} | t_{d,n}, \beta) P(t_{d,n} | \theta_d) \right) P(\theta_d | \alpha) d\theta_d$$</p>

The likelihood function for a set of document $D = \{d_1,d_2,\dots,d_D\}$ given the model parameters $\alpha$ (topic prior), $\beta$ (word prior), $\theta_d$ (topic distribution), $\phi_k$ (word distribution for each topic) is the probability of the observed words $W$ in the documents, marginalized over the latent variables $T$ and $\theta$: 
<p align='center'>$$P(W|\alpha,\beta) = \prod_{d=1}^{D} \int_{\theta_d} \left( \prod_{n=1}^{N_d} \sum_{t_{d,n}=1}^K P(w_{d,n} | t_{d,n}, \phi_{t_{d,n}}) P(t_{d,n} | \theta_d) \right)  P(\theta_d | \alpha) d\theta_d$$</p>

where:
- $P(w_{d,n} | t_{d,n}, \phi_{t_{d,n}})$ is the probability of observing a word given a topic (the word distribution $\phi_{t_{d,n}}$)
- $P(t_{d,n} | \theta_d)$ is the probability of choosing a topic from the topic distribution for document $d$
- $P(\theta_d | \alpha)$ is the prior over the topic distribution $\theta_d$, sampled from a Dirichlet distribution with parameter $\alpha$.

### Inference
Computing the above likelihood function is computationally intractable because of the integral and the sum over all possible topic assignments. To make this workable, Bayesian inference techniques like Gibbs sampling are used to approximate the posterior distributions of the latent variables and compute the likelihood.
