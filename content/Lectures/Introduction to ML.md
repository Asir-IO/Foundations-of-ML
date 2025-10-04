---
Course: "[[Theoretical Foundations of ML]]"
Previous Lecture:
Next Lecture:
---
# 1. The Why
---
**The goal** of many ML models is to accurately assign an output $(y)$ to some input $(x)$.
- They achieve this via gathering an awful lot of input and their corresponding output, the set that contains every gathered *(input, output)* pair is known as the "**dataset**".
	- That is, they learn the relationship between the i/o and get good at predicting the output for **new, unseen input.**
	- This type of learning is known as "supervised" learning.

**Another goal** is finding out which inputs are more *"similar"* to one another.
- They do it by clustering similar input together and creating a label for each cluster, so that when **a new input** arrives, it knows which cluster it belongs to.
- This type of learning is known as "unsupervised" learning.
---
# 2. Building the Model (supervised learning)
---
We first assume a function, called a hypothesis h, that assigns an output $h(x)$ to each input $(x)$. 
The output is also commonly known as $\hat{y}$.
$$
h(x)=\hat{y}
$$
We then compare this predicted output $\hat{y}$ with the input’s actual output, or label, $y$. 
The difference between them $|\hat{y}-y|$ can be thought of as the **"incorrectness"** or "**loss**" of the model for that specific input.

To improve the overall model, we must combine the loss from every *(input, output)* pair in our dataset and then minimize this *combined* loss.

---
# 3. Error/Risk Calculation
The loss of some $(x, y)$ pair is written as
$$
\mathcal{L}(h(x), y) \quad \text{ or } \quad \lambda_{\hat{y}y}
$$
## Expected Risk
The Expected Risk of a hypothesis $h$, denoted $R(h)$, is the expected value of the loss over all possible $(x, y)$ pairs, not just the pairs that we gathered in our dataset.
$$
R(h) = E[\mathcal{L}(h(x), y)]
\tag{1}
$$
##### \*Here's its formula if the output is discrete and the input is continuous\*
$$
E[\mathcal{L}(h(x), y]=\sum_{y}\int_{x} \mathcal{L}(h(x), y)\cdot p(x, y) \,dx
$$
This formula calculates the **"true risk"** of a model $h$ in a perfect world where we can gather every possible $(x, y)$ pair.

But... this world doesn't exist, that's why we need some approximation, a way to estimate the risk of $h$ with our limited amount of gathered $(x, y)$ pairs.

And that's what the $\text{Empirical risk}$ is:
$$
\text{Given our dataset }S=\{(x_1, y_1), (x_2, y_2),\;...,\; (x_m, y_m)\}
$$
The Empirical risk of $h$ over it is the expected value of the loss across all the pairs in $S$, and thus is calculated as:
$$
\hat{R}_S(h)=\frac{1}{m}\sum_{i=1}^{m}\mathcal{L}(h(x_i), y_i)
$$
### Getting the Empirical risk closer to the Expected risk
To do this, we can either:

$(1).$ Expand our dataset as much as possible $(\text{increase } m)$.

$(2)$. Compute the emp. risk for $h$ over multiple datasets and taking the average of that, but only if each dataset is *identically distributed*.
$$
\text{Given that the sampled datasets live in }\mathcal{S}=\{S_1, S_2,\;...,\; S_n\}
$$
Then,
$$
R(h)=E[\hat{R}_S(h)], \text{ with respect to every S in }\mathcal{S}
$$
$\textit{Proof:}$
$$
\begin{align}
E[\hat{R}_S(h)]&=E[\frac{1}{m}\sum_{i=1}^{m}\mathcal{L}(h(x_i), y_i)]\\
&=\frac{1}{m}\sum_{i=1}^{m}E[\mathcal{L}(h(x_i), y_i)] &\textit{from } (1) \\
&=\frac{1}{m}\sum_{i=1}^{m}R(h) \\
&=\frac{1}{m}\cdot m\cdot R(h)=R(h)
\end{align}
$$
---
# General Notes