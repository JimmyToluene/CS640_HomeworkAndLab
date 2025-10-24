# Oct-21-Week7-CS640

# Introduction-to-NeuralNetworks

### Learning:

- Tuning parameters from a model
- Try to learn a specific pattern from a huge amount of data
- Decreasing costs

### Learning Models ~ Picking Best Function Matching Data

#### Which function is best ?

- Quantify prediction quality w/ loss function
  $$
  minimize_f\; \epsilon L(f(x_i),yi)
  $$
  

  We have a number of datasets $$x_i,y_i$$, and one function for predicting this data. So we want to minimize our function to find a best estimate function

  Example of loss function:
  $$
  L_2 = suqarederror \\
  L_1 = abs.error	\\
  cross\;entropy: -logL
  $$
  
- Which functions do we consider as possible best functions?

​	Linear Regression,

### Learning Models ~ Picking Best Function Parameters

- Given a functional form or function family, what parameters correspond to the function with the best loss?

- For example, if we choose the family of linear functions, so y=mx+b is our functional form, then what are the best values of m and b to match the target function?

### Empirical Risk Minimization

Empirical risk minimization is the principle that models should be optimized to reduce the empirical risk on the training data, as such models are most likely to perform well on similar data.

- Risk=loss in this context. Empirical risk = training loss.

- Explicit assumption that “real” data will have same distribution as training data.

- Overfitting concern?

### Training vs Test Loss

- Observation: training loss is skewed low. Why?
- Best practice: train on some data, test on different data
- What does it mean if training and test loss are significantly different?

### Training vs Validation Loss (?)

Validation sets helps us to check the model good or bad by trying to use different models.

- How do we decide what kind of model to use without looking at training or test losses?

  ```
           /\
          /  \
      Train		Test
  		 /\
  		/  \
  Train  Validation
  ```

  

- Can we overfit on validation data?

## Gradient Descent

- Idea: compute **partial derivatives of loss** with respective to every model parameter.

$$
\nabla_\theta J(\theta) = \begin{bmatrix} \frac{\partial J}{\partial \theta_1} \\ \frac{\partial J}{\partial \theta_2} \\ \vdots \\ \frac{\partial J}{\partial \theta_n} \end{bmatrix}
$$

- Update model parameters using the gradient to identify direction of decrease.

$$
\theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta_t)\\
\alpha = learning\;rate
$$

### Stochastic Gradient Descent

Practice Batch Size = It depends on GPU memory

**Observation: computing the gradient for a 1B row data set is slow.**

- Question 1: Do we need to do that for each step of gradient descent?

​	**Epoch = I round using all data**

​	

- **Question 2: Can we approximate the gradient more cheaply?**

​	If we compute w/ 1 million rows, it will be 1000x faster updates

​	Randomly partition into 1k subsets. Do update w/ each subset.

#### Benefits of Stochastic Gradient Descent

**How many epochs?**

For small medium models, until validation loss steps improving

For large language models, just b/c slow and small improvements

- Speed:

  1 update per batch. Epoch cost with same.

  ```
  # of batches/epoch = #parameter updates/epoch /appro sppedup from SGD
  ```

- Better solution:

​	Batch variance helps traning out of **stucking at local minimal problem**

![myplot](/Users/jimmyjia/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CS640_HomeworkAndLab/Notes/Figures/myplot.png)

### One Example: Evaluation Problem for HMMs

How likely is it that this HMM produced these observations O?

- What is Pr⁡[O│λ]?

**Simpler version w/o argmax to pick best parameters.**

We were able to write recursive formulas for this probability in terms of the model parameters (forward/backward algorithms).

---

<span style="font-variant: small-caps;">BaumWelch</span>
$$
\begin{array}{l}\text{For all } i, j, t, \text{ and with all calculations conditioned on } O... \\ \\ \bullet \text{ Compute probabilities of being at} \\ \quad \text{state } i \text{ at time } t \text{ and state } j \text{ at time } t + 1 \\ \\ \bullet \text{ Compute probability of transitioning from state } i \text{ to state } j. \\ \bullet \text{ Update A to match those probabilities.} \\ \\ \bullet \text{ Compute probability of output } k \text{ when at state } i. \\ \bullet \text{ Update B to match those probabilities.} \end{array}
$$

---

#### Probability Constraints

- Gradient descent does not know that $\pi$ vector should add up to one. Same issues for rows of A and B.

​	**Solution: ** Lagrange

**Logistic Function: **

- Standard logistic function:
  $$
  f(x) = \frac{1}{1+e^{-x}}
  $$

**Softmax Solution**

- Generalization of logistic funtion to more dimensions
  $$
  𝑓_𝑖 (𝐳)= \frac{𝑒^{𝑧_𝑖}}{∑_𝑗𝑒^{𝑧_𝑗 }}
  $$

### Is Gradient Descent a Good Way to Learn HMMs?

- Generally, no, since we have analytical solutions for EM steps.
  - And they are pretty fast.

- Baum-Welch has weakness that it might converge to non-optimal solution.
  - Gradient descent has this problem too. 

- Gradient descent is preferred more often when we do not have an analytical solution, or the analytical solution is slow.

---

## NeuralNetworks

#### 1D Linear Regression



![image-20251021120520925](/Users/jimmyjia/Library/Application Support/typora-user-images/image-20251021120520925.png)

#### Example shallow network

> ![QQ20251021-120842](/Users/jimmyjia/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CS640_HomeworkAndLab/Notes/Figures/QQ20251021-120842.png)

> ![Picture1](/Users/jimmyjia/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CS640_HomeworkAndLab/Notes/Figures/Picture1.png)

$$
h_1 = a[\theta_{10}+\theta_{11}x] \\
h_2	= a[\theta_{20}+\theta_{21}x]	\\
h_3	= a[\theta_{30}+\theta_{31}x]	\\
\\
y = \phi_0 + \phi_{1}h_1 + \phi_{2}h_2 + \phi_{3}h_3
$$

### Networking Pruning, Lottery Ticket



# Oct-23-Week7-CS640

# Introduction-to-NeuralNetworks-Part2

### 1. Unversial Approach of NN

$$
h_d = a[\theta_{d0} + \theta_{d1}x]\\
y = \phi_0 + \sum_{a=1}^D \phi_d h_d
$$

**A formal proof that, with enough hidden units, a shallow neural network can describe any continuous function on a compact subset of $R^D$ to arbitary precision**

1 input : 1 hidden layer, RELU suffices. Makes a piecewise linear approximation

2 input:  2 hidden layers, RELU(anything not polynomial works, originally proved for sigmoid tanh) suffice.

### 2. Why calculate gradients in NN?

#### Problem 1: Computing gradients

**Loss: sum of individual terms:**
$$
L[\phi] = \sum_{i=1}^{I} \ell_i 
= \sum_{i=1}^{I} \, \ell \big(f[x_i, \phi],\, y_i \big)
$$

**SGD Algorithm:**

$$
\phi_{t+1} \leftarrow \phi_t - \alpha \sum_{i \in \mathcal{B}_t} 
\frac{\partial \ell_i[\phi_t]}{\partial \phi}
$$

**Parameters:**

$$
\phi = \{\beta_0, \Omega_0,\; \beta_1, \Omega_1,\; \beta_2, \Omega_2,\; \beta_3, \Omega_3\}
$$

**Need to compute gradients:**

$$
\frac{\partial \ell_i}{\partial \beta_k}
\quad \text{and} \quad
\frac{\partial \ell_i}{\partial \Omega_k}
$$

- But it’s a huge equation, and we need to compute derivative
  - for every parameter
  - for every point in the batch
  - for every iteration of SGD

---

### 3. Backpropagation Intuition

#### An efficient way to compute gradients using the chain rule.

1.Forward pass: compute and save all intermediate values of the computation.

2.Backward pass: compute gradients with respect to all intermediate values using the chain rule.

#### BackProp intuition #1: The forward pass

> ![Picture2](/Users/jimmyjia/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CS640_HomeworkAndLab/Notes/Figures/Picture2.png)

- Orange weight multiplies activation (ReLU output) in previous layer 

- We want to know how change in orange weight affects loss

- If we double activation in previous layer, weight will have twice the effect

- Conclusion: we need to know the activations at each layer.

---

#### BackProp intuition #2: The backward pass

> ![Picture3](/Users/jimmyjia/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CS640_HomeworkAndLab/Notes/Figures/Picture3.png)

To calculate how a small change in a weight or bias feeding into hidden layer **h**3 modifies the loss, we need to know:

- how a change in layer **h**1 affects layer **h**2

- how a change in layer **h**2 affects layer **h**3

- how layer **h**3 changes the model output

- how the model output changes the loss

---

#### Toy Function Example:

$$
f[x, \phi] = \beta_3 + \omega_3 \cdot \cos\left[ \beta_2 + \omega_2 \cdot \exp\left( \beta_1 + \omega_1 \cdot \sin(\beta_0 + \omega_0 \cdot x) \right) \right]
$$

$$
\ell_i = \left( f[x_i, \phi] - y_i \right)^2
$$

- Consists of a series of functions that are composed with each other.
- Unlike in neural networks, it just uses scalars (not vectors).
- “Activation functions” used: $\sin$, $\exp$, $\cos$

$$
\frac{\partial \ell_i}{\partial \beta_0}, \quad
\frac{\partial \ell_i}{\partial \omega_0}, \quad
\frac{\partial \ell_i}{\partial \beta_1}, \quad
\frac{\partial \ell_i}{\partial \omega_1}, \quad
\frac{\partial \ell_i}{\partial \beta_2}, \quad
\frac{\partial \ell_i}{\partial \omega_2}, \quad
\frac{\partial \ell_i}{\partial \beta_3}, \quad \text{and} \quad
\frac{\partial \ell_i}{\partial \omega_3}
$$

*How does a small change in* $\beta_3$ *change the loss* $\ell_i$ *for the $i$-th example?*
$$
\frac{\partial \ell_i}{\partial \omega_0}
= -2 \left( \beta_3 + \omega_3 \cdot \cos\left[ \beta_2 + \omega_2 \cdot \exp\left( \beta_1 + \omega_1 \cdot \sin(\beta_0 + \omega_0 \cdot x_i) \right) \right] - y_i \right)
$$

$$
\cdot \, \omega_1 \omega_2 \omega_3 \cdot x_i 
\cdot \cos(\beta_0 + \omega_0 \cdot x_i) 
\cdot \exp\left( \beta_1 + \omega_1 \cdot \sin(\beta_0 + \omega_0 \cdot x_i) \right)
\cdot \sin\left[ \beta_2 + \omega_2 \cdot \exp\left( \beta_1 + \omega_1 \cdot \sin(\beta_0 + \omega_0 \cdot x_i) \right) \right]
$$

## Foward pass:

$$
f[x,\phi] \;=\; \beta_3 \;+\; \omega_3 \cdot
\cos\!\left[\, \beta_2 \;+\; \omega_2 \cdot
\exp\!\left( \beta_1 \;+\; \omega_1 \cdot \sin(\beta_0 + \omega_0 \cdot x) \right)
\right]
$$

$$
\ell_i \;=\; \big(f[x_i,\phi]-y_i\big)^2
$$

**Introduce intermediate quantities (forward pass):**

$$
\begin{aligned}
f_0 &= \beta_0 + \omega_0 \cdot x_i,            &\qquad f_2 &= \beta_2 + \omega_2 \cdot h_2,\\
h_1 &= \sin(f_0),                                &\qquad h_3 &= \cos(f_2),\\
f_1 &= \beta_1 + \omega_1 \cdot h_1,             &\qquad f_3 &= \beta_3 + \omega_3 \cdot h_3,\\
h_2 &= \exp(f_1),                                &\qquad \ell_i &= (f_3 - y_i)^2.
\end{aligned}
$$

### Backward pass

We want the derivatives of the loss w.r.t. the intermediate quantities, **in reverse order**:

$$
\frac{\partial \ell_i}{\partial f_3},\quad
\frac{\partial \ell_i}{\partial h_3},\quad
\frac{\partial \ell_i}{\partial f_2},\quad
\frac{\partial \ell_i}{\partial h_2},\quad
\frac{\partial \ell_i}{\partial f_1},\quad
\frac{\partial \ell_i}{\partial h_1},\quad
\text{and}\quad
\frac{\partial \ell_i}{\partial f_0}.
$$
**1) First derivative is trivial**

Given $\ell_i = (f_3 - y_i)^2$, the first gradient is
$$
\frac{\partial \ell_i}{\partial f_3} = 2\,(f_3 - y_i).
$$
**2) Remaining derivatives via the chain rule**

Recall the forward definitions (from the forward-pass slide):
$$
\begin{aligned}
 f_0&=\beta_0+\omega_0 x_i,&\quad h_1&=\sin(f_0),&\quad \\
 f_1&=\beta_1+\omega_1 h_1,&\quad h_2&=\exp(f_1),\ \\
 f_2&=\beta_2+\omega_2 h_2,&\quad h_3&=\cos(f_2),&\quad \\
 f_3&=\beta_3+\omega_3 h_3,&\quad \ell_i&=(f_3-y_i)^2 .
 \end{aligned}
$$
**(A) Pure chain-rule form (matches the slide’s nesting)**
$$
\frac{\partial \ell_i}{\partial f_2}

= \frac{\partial h_3}{\partial f_2}

\left(\frac{\partial f_3}{\partial h_3}\,\frac{\partial \ell_i}{\partial f_3}\right),

\qquad

\frac{\partial \ell_i}{\partial h_2}

= \frac{\partial f_2}{\partial h_2}

\left(\frac{\partial h_3}{\partial f_2}\,\frac{\partial f_3}{\partial h_3}\,\frac{\partial \ell_i}{\partial f_3}\right),
$$

$$
\frac{\partial \ell_i}{\partial f_1}

= \frac{\partial h_2}{\partial f_1}

\left(\frac{\partial f_2}{\partial h_2}\,\frac{\partial h_3}{\partial f_2}\,\frac{\partial f_3}{\partial h_3}\,\frac{\partial \ell_i}{\partial f_3}\right),

\qquad

\frac{\partial \ell_i}{\partial h_1}

= \frac{\partial f_1}{\partial h_1}

\left(\frac{\partial h_2}{\partial f_1}\,\frac{\partial f_2}{\partial h_2}\,\frac{\partial h_3}{\partial f_2}\,\frac{\partial f_3}{\partial h_3}\,\frac{\partial \ell_i}{\partial f_3}\right),
$$

$$
\frac{\partial \ell_i}{\partial f_0}

= \frac{\partial h_1}{\partial f_0}

\left(\frac{\partial f_1}{\partial h_1}\,\frac{\partial h_2}{\partial f_1}\,\frac{\partial f_2}{\partial h_2}\,\frac{\partial h_3}{\partial f_2}\,\frac{\partial f_3}{\partial h_3}\,\frac{\partial \ell_i}{\partial f_3}\right).
$$
**(B) With local derivatives substituted (ready for numeric backprop)**
$$
Local\;partials:
 [
 \frac{\partial f_3}{\partial h_3}=\omega_3,\quad 
 \frac{\partial h_3}{\partial f_2}=-\sin(f_2),\quad
 \frac{\partial f_2}{\partial h_2}=\omega_2,\quad\\
 \frac{\partial h_2}{\partial f_1}=e^{f_1}=h_2,\quad
 \frac{\partial f_1}{\partial h_1}=\omega_1,\quad
 \frac{\partial h_1}{\partial f_0}=\cos(f_0).
 ]
$$
Now:
$$
\frac{\partial \ell_i}{\partial f_3} = 2\,(f_3 - y_i),

\qquad

\frac{\partial \ell_i}{\partial h_3} = \omega_3 \,\frac{\partial \ell_i}{\partial f_3},

\frac{\partial \ell_i}{\partial f_2} = \big(-\sin f_2\big)\,\omega_3\,\frac{\partial \ell_i}{\partial f_3},

\qquad

\frac{\partial \ell_i}{\partial h_2} = \omega_2 \big(-\sin f_2\big)\,\omega_3\,\frac{\partial \ell_i}{\partial f_3},
$$

$$
\frac{\partial \ell_i}{\partial f_1} = h_2\,\omega_2 \big(-\sin f_2\big)\,\omega_3\,\frac{\partial \ell_i}{\partial f_3},\qquad

\frac{\partial \ell_i}{\partial h_1} = \omega_1\,h_2\,\omega_2 \big(-\sin f_2\big)\,\omega_3\,\frac{\partial \ell_i}{\partial f_3},
$$

$$
\frac{\partial \ell_i}{\partial f_0} = \cos(f_0)\,\omega_1\,h_2\,\omega_2 \big(-\sin f_2\big)\,\omega_3\,\frac{\partial \ell_i}{\partial f_3}.
$$

![Picture5](/Users/jimmyjia/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CS640_HomeworkAndLab/Notes/Figures/Picture5.png)