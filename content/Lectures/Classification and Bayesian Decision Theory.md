# 1. Classification
---
It's the task of predicting a **class** for a given input x. 

A machine learning algorithm creates a **classifier** $h(x)$ that approximates the true class, $c(x$) or $y$, for any input $x$. 

$\text{\large An error}$ occurs when our prediction is wrong
$$
\begin{gather*}
h(x)\;\neq\;c(x) \\
\text{(error condition)}
\end{gather*}
$$
We decide how to handle such erros by defining an error function
$$
\begin{gather*}
\mathcal{L}(h(x), c(x)) \\
\text{(error or loss function)}
\end{gather*}
$$
### An example of a classification loss function

$$
\begin{gather*}
\mathcal{L}(h(x), y) \quad=\quad 1_{h(x) \neq y} \\
\text{(Zero-One loss or Uniform Error loss)}
\end{gather*}
$$
In that example, every error had a cost of 1, and a correct prediction had a cost of 0.

But if not all errors are equally bad, we need a more flexible way to define the cost of a mistake. This is done using a **cost matrix**, $\lambda_{\hat{y}y}$.

This matrix defines the penalty for every possible combination of (a predicted class, and a true class). We denote this cost as $\lambda_{\hat{y}y}$ which represents the loss for predicting the class $\hat{y}$​ when the true class was actually $y$.

For example, the cost matrix for a simple **uniform loss** (or 0/1 loss) with classes $\{A, \;B, \;C\}$ would be
> [!example]
> $$
> {\Large\lambda_{\hat{y}y}} \quad= \quad
> \begin{array}{c|ccc}
>     \text{Predicted}(\hat{y})\backslash\text{True}(y)& \mathbf{A} & \mathbf{B} & \mathbf{C} \\
>     \hline
>     \mathbf{A} & 0 & 1 & 1 \\
>     \mathbf{B} & 1 & 0 & 1 \\
>     \mathbf{C} & 1 & 1 & 0
> \end{array}
> $$
### Here are a few examples of regression loss functions
$$
\begin{gathered}
  \mathcal{L}(\hat{y}, y) \quad=\quad |\hat{y}-y|\\
  \text{(absolute value of error loss)}
\end{gathered} \tag{1}
$$
$$
\begin{gathered}
  \mathcal{L}(\hat{y}, y) \quad=\quad ||\hat{y}-y||^2\\
  \text{(square of error loss)}
\end{gathered} \tag{2}
$$
$$
\begin{gathered}
	  \mathcal{L}(\hat{y}, y) =
	\begin{cases}
	0 & |\hat{y} - y| < \epsilon \\
	1 & \text{otherwise}
	\end{cases}\\
  \text{(uniform error loss)}
\end{gathered} \tag{3}
$$
$$
\begin{gathered}
	  \mathcal{L}(\hat{y}, y) =
	\begin{cases}
	\hat{y}-y & \hat{y} - y\geq0  \\
	0 & \text{otherwise}
	\end{cases}\\
  \text{(hinge loss)}
\end{gathered} \tag{4}
$$
---
# 2. Error Risk
---
The **Error Risk** or **Expected Risk** or **Generalization Risk**, $R(h)$, is the mean value of the error for a classifier $h$ over every possible i/o pair $(x, \;y)$.

$$
R(h)=E\big[\mathcal{L}(h(x), y)\big]
$$
To transform the operator $E[\text{expression}]$ into something we can work with (like sums and integrals), we do the following:
1. Identify the random variables in the $\text{expression}$ inside it, they're $x$ and $y$ in $\mathcal{L}(h(x), \;y)$
2. Find their Joint distribution, it's $p(x, \;y)$ for $x$ and $y$, which's the same thing as $p(x \text{ and }y)$.

> [!warning] Notation Alert
> $$
> \begin{align*}
> p(x, y) &=\text{the probability of x and y occurring together} \\
> &=p(x \text{ and } y)
> \end{align*}
> $$

To finally evaluate $E[\text{expression}]$, we sum that $\text{expression}$ over all possible pairs of the random variables, with each term getting weighted by the probability of that pair occurring (from the joint dist.). 
## That is,
$$
\begin{align*}
E\big[\mathcal{L}(h(x), y)\big] &= \text{Sum over every }(x, y) \text{ pair of }\mathcal{L}(h(x), y)\cdot p(x, \;y) \\
&= \int_{x} \sum_{y}\mathcal{L}(h(x), y)\cdot p(x, y) \,dx \\
&=R(h)
\end{align*}
$$
*(The doctor briefly mentioned Cross Validation here)*

---
# 3. Bayesian Decision Theory
---
The goal of Bayesian Decision Theory is to find the best possible **classifier** for a given problem. This ideal classifier is called the **Bayes classifier**.
#### How does it (theoretically) do it?
Given a **Hypothesis Space** H, which is the set of all possible classifiers $\mathcal{H}=\{h_1, h_2, \;...\}$, and a risk function $R$, the Bayes classifier $h_B$ is the specific classifier $h\in\mathcal{H}$ that has the minimum possible **risk**.
$$
h_B=\text{arg min}_{h \in \mathcal{H}}R(h) \tag{1}
$$
[[#An Example on the usage of arg min|(Here's an example showing how arg min works)]]

If you [[#That is,|forgot]] what $R(h)$ is.fff
$$
\begin{align*}
h_B&=\text{arg min}_{h \in \mathcal{H}}\int_{x} \sum_{y}\mathcal{L}(h(x), y)\cdot p(x, y) \,dx \tag{2}
\end{align*}
$$
$\mathcal{L}(h(x),y)$ here is the loss for predicting $h(x)$ when the true class is $y$, and $\mathcal{Y}$ is the set of all possible classes.
#### Since
$$
\begin{align*}
p(x,\;y)&=p(y|x)\cdot p(x) \quad \quad\text{(the product chain law)}\\
&=p(x|y)\cdot p(y)
\end{align*}
$$
$\text{Where:}$

$p(y|x)$ is the probability that $y$ is the class, given that $x$ was the input.
#### Then
$$
\begin{align*}
h_B&=\text{arg min}_{h \in \mathcal{H}}\int_{x} \sum_{y}\bigg(\mathcal{L}(h(x), y)\cdot p(x, y) \bigg)\,dx \tag{2} \\
&=\text{arg min}_{h \in \mathcal{H}}\int_{x} \sum_{y}\bigg(\mathcal{L}(h(x), y)\cdot p(y|x)\cdot p(x)\bigg) \,dx \\
&=\text{arg min}_{h \in \mathcal{H}}\int_{x} p(x)\cdot \sum_{y}\bigg( \mathcal{L}(h(x), y)\cdot p(y|x) \bigg)\,dx \tag{3}\\
&=\text{arg min}_{h \in \mathcal{H}}\int_{x} \text{(the risk for x) }\;\,dx \tag{3}\\
\end{align*}
$$

This expression $(3)$ looks for the one function $h\in\mathcal{H}$ that minimizes the total sum of the risk from every single input $x$. Trying to find this function all at once is incredibly difficult.

However, since this is **equivalent** to minimizing the risk for each $x$ *separately*, summing each risk up, and returning the $h$ producing this sum, we may no longer need to look for $h$.

That is, we're no longer interested in a *global* function $h$ that minimizes this sum and that can be used on any value of $x$.

We will instead, for *any given instance* of $x$, find its its $h_B(x)$ $\text{(which is simply the class having the least risk for x)}$

Of course $h_B(x) \in \mathcal{Y}$   $\textit{(the set of all possible classes)}$ 

$$
\begin{align*}
h_B(x)&=\text{arg min}_{h(x) \in \mathcal{Y}}\;\text{(the risk for x)} \tag{3} \\
&=\text{arg min}_{h(x) \in \mathcal{Y}}\; p(x)\cdot\sum_{y}\bigg(\mathcal{L}(h(x), y)\cdot p(x, y) \bigg)  \\
&=\text{arg min}_{h(x) \in \mathcal{Y}}\; \sum_{y}\bigg(\mathcal{L}(h(x), y)\cdot p(x, y) \bigg) \\
\end{align*}
$$
> [!note] Note
> Since the term $p(x)$ is a positive constant and doesn't change which $h(x)$ wins, we were able to remove it from the formula.
# General notes 
---
> [!example] An Example on the usage of arg min
> $$
> \text{Given an array }A:
> \quad \quad \quad
> \begin{matrix}
> i \\
> 0 \\
> 1 \\
> 2 \\
> 3
> \end{matrix}
> \;
> \begin{matrix}
> A \\
> \begin{bmatrix}
> 6 \\
> 8 \\
> 5 \\
> 4
> \end{bmatrix}
> \end{matrix}
> $$
> $\text{arg min}_{i \in \{0, 1, 2, 3\}}A[i]$ = The index $(i)$ with the minimum value of $A[i]$ = 3
