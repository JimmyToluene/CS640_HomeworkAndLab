# Measure of Success

### Regression

- Given a data set of $x_i, y_i$ values, construct a function $f(x_i )≈y_i$.
  - $x_i$ often a vector, so **$x_i$** may be more appropriate.
  - $y_i$ usually a scalar value.

### Loss Functions 

Let $y ̂_i=f(x_i )$. How good is $f$ ?

- Mean Squard Error
- Root MSE
- Mean Absolute Error

## Binary Measure of Success

![image-20251014154500387](/Users/jimmytoluene/Library/Application Support/typora-user-images/image-20251014154500387.png)

### Accuracy

Accuracy = # correct predictions / # predictions

- Not all errors are equal

### Confusion Matix

|                        | Prediction: Positive(1) | Prediction: Negative(0) |
| ---------------------- | ----------------------- | ----------------------- |
| **Truth: Positive(1)** | True Positive           | False Negative          |
| **Truth: Negative(0)** | False Positive          | True Negative           |

**1.True Positive Rate (Sensitivity/Recall) =  P(Predict to be True | Actual is true)**
$$
TPR = \frac{TP}{TP+FN}
$$
**2.False Positive Rate = P(Predict to be True|Actual is False)**
$$
FPR = \frac{FP}{TN+FP}
$$


**Precision = P(Actual is True|Predict to be True)**
$$
Precision = \frac{TP}{TP+FP}
$$

### $F_1$ Score

Harmonic Meaning of recall and precision
$$
F_1 = \frac{2}{recall^{-1}+presision^{-1}}= \frac{2}{\frac{TP}{TP+FN}+\frac{TP}{TP+FP}}=\frac{2TP}{2TP+FN+FP}
$$

- If accuracy = 0, means (TP + TN) / (TP + TN + FP + FN) = 0
  - TP and TN all is 0, so $F_1$ = 0
- If accuracy = 1, means FP and FN = 0; 
  - $F_1 = 1$

#### The difference between accuracy and $F_1$ 

**F₁ ignores TN, accuracy includes TN** 

- this makes F₁ better for imbalanced data where TN dominates

---

### Decision Thresholds

```
Threshold = 0.5 
Sample 1: P(positive) = 0.7 → 0.7 ≥ 0.5 → Predict: Positive (1) 
Sample 2: P(positive) = 0.3 → 0.3 < 0.5 → Predict: Negative (0) 
Sample 3: P(positive) = 0.9 → 0.9 ≥ 0.5 → Predict: Positive (1) 
```

The **threshold** is the decision boundary that converts predicted probabilities into class labels. 

### ROC Curve:

![image-20251014162546670](/Users/jimmytoluene/Library/Application Support/typora-user-images/image-20251014162546670.png)

**ROC curve is created by:** 

1. Varying threshold from 0.0 to 1.0 
2. For each threshold, create a confusion matrix 
3. Calculate TPR and FPR from each matrix 
4. Plot (FPR, TPR) points

**The dashed line represents a completely random classifier** 

- like flipping a coin to make predictions.

Key Properties:

- **Goes from (0,0) to (1,1)** 
  - Diagonal line with slope = 1

- **TPR = FPR** at every point on this line
- **AUC = 0.5** (Area Under Curve)
- Represents **random guessing**

## Questions:

#### 1.If ROC curve is below the diagonal (AUC < 0.5)

We can simply FLIP model predictions. 

A model performing **worse than random** (AUC < 0.5) is **actually learning something useful**

- just the **opposite** of what you want!

#### 2.What does it mean when a model has a true positive rate lower than the false positive rate? Give an example improvement to such a model that illustrates the issue.

When TPR<FPR, our model is:

- Making more false alarms than catching real cases.
- Worse at identifying actual positive.

Example: A smoke detector that:

- Only alerts for **20% of actual fires** (low TPR)
- But **false alarms 40% of the time** when cooking (high FPR)

#### Improvement:

**The threshold adjustment** is usually the quickest fix to restore TPR > FPR and make the model practically useful. Usually we let threshold to higher than before.

```
Dataset: 5 cases
- Samples 1, 2 are actual Fires (2 fires)
- Samples 3, 4, 5 are Safe (3 safe)

Predictions (threshold = 0.5):
Sample 1: P(smoke) = 0.3 → Predicted: Safe  (0.3 < 0.5)
Sample 2: P(smoke) = 0.6 → Predicted: Fire  (0.6 ≥ 0.5)
Sample 3: P(smoke) = 0.7 → Predicted: Fire  (0.7 ≥ 0.5)
Sample 4: P(smoke) = 0.4 → Predicted: Safe  (0.4 < 0.5)
Sample 5: P(smoke) = 0.1 → Predicted: Safe  (0.1 < 0.5)

Analysis:
Sample 1: Actual=Fire, Predicted=Safe → FN ✗ (missed!)
Sample 2: Actual=Fire, Predicted=Fire → TP ✓
Sample 3: Actual=Safe, Predicted=Fire → FP ✗
Sample 4: Actual=Safe, Predicted=Safe → TN ✓
Sample 5: Actual=Safe, Predicted=Safe → TN ✓

Confusion Matrix:
              Predicted
           Fire    Safe
Actual Fire  1       1     (TP=1, FN=1)
       Safe  1       2     (FP=1, TN=2)

Calculate:

TPR = TP/(TP+FN) = 1/(1+1) = 0.5 = 50%
FPR = FP/(FP+TN) = 1/(1+2) = 0.33 = 33%

Result: TPR > FPR (50% > 33%)
```

# Responsible AI

## A Framework for Understanding Sources of Harm throughout the Machine Learning Life Cycle 

![image-20251014171504276](/Users/jimmytoluene/Library/Application Support/typora-user-images/image-20251014171504276.png)

### 1.Historical Bias:

Historical bias caused by replicating prevalent bias and stereotypes from the world

**Image Search Results:**

- Search "CEO" → almost all images showed white men
- Search "nurse" → almost all images showed women
- Search "unprofessional hairstyles" → predominantly Black hairstyles

### 2.Representation Bias:

Occurs when collected data **underrepresents** some part of population

**Object Recognition Training Data**

- Dominated by images from **North America and Europe**
- **Underrepresented:** Asia, Africa, South America, Middle East (This ethic are not included)
- Consequence:
  - Model recognizes "wedding" → white wedding dress
  - Fails on traditional weddings from other cultures
  - "Kitchen" → Western-style kitchen, not outdoor cooking areas common elsewhere

### 3.Measurement Bias:

- Occurs while deciding features and labels to use in a prediction problem.
  - **Proxy **(A measurable of stand-in for something you actually care about but can't easily measure) 
  - Proxy is an oversimplification of a complex construct.
  - Measurement of the proxy varies across groups.

**Example:**

[Northpointe’s COMPAS ](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing): Tool for predicting whether the defendant will re-offend.

- Higher FPR for black defendants.

- Used proxy variables such as “arrests” or “rearrests” to measure crime.

- Minority areas are heavily policed

### 4.Learning Bias:

- Occurs when design choices are **overly focused** on optimizing the objective function

Example:

Dataset of 100 emails, 90 not spam and 10 spam.

Focusing more on optimizing the objective will encourage model to predict every mail as not a spam.

### 5.Evaluation Bias:

- Occurs when **benchmark data does not represent the actual population**.

Example:

When evaluating many tech companies Gender Classifier , the accuracy of "Lighter Male" are much higher than "Darker Female"

### 6. Aggregation Bias:

- A single model cannot generalize to all groups(yet).
- Makes system either sub-optimal for all groups or optimal for dominant group.

Example:

- A recent study on NLP tools evaluated on tweets showed that it usually do not understand context-dependent emoji or hash tags.

### 7.Deployment Bias:

- Occurs when there is a mismatch between how the model was intended to be used vs how it is used.

Example: COMPAS

- Treated as an automated system however it does require human decision makers to interpret.

# Rule-Based System

## Prolog

- A "logic programming" language.
  - Turing-complete
  - Only will look at deduction support.

### Prolog - Facts:

- likes(john,mary).
- valuable(gold).
- owns(john,gold).

First part is relationship.

Then objects separated by commas in parenthesis.

- Order matters

Period ends the fact

All relationship and objects names start with lowercase letter.

### Prolog - Rules:

$$
parent(X,Y) :- \;father(X,Y).\\
parent(X,Y) :-\; mother(X,Y).
$$

​			**Left**(X,Y): What is proven 	:- **"if"**->this is a rule 		**Right**:condition

### Prolog - Questions:

$$
?-likes(john,mary).\\
?-likes(john,patrick).
$$

### Prolog - Answers:

Any question will get one of these two answers:

- Yes: relationships matching the question provably do exist.
- No: relationships matching the question cannot be proven (not necessarily to be false)

### Prolog - Conjunctions

- ?-gives(john,X,mary),valuable(X).
- grandfather(X,Z). :- father(X,Y), parent(Y,Z).

### Prolog - Inference:

- ?-gives(john,X,mary),valuable(X).
  - First try to find X such that gives(john,X,mary).
    - Check facts first.
    - Then check for rules (recursion)
  - For each such X find, recursively try to solve the rest of the question.
    - So recurse on valuable(X).
    - Backtracking will be used unless first choice of X succeeds

# Searching and Planning

## Modeling Search Problems

- Typical abstraction models problems as a graph $G=(V,E)$
- Nodes are **states**
- Edges are possible transitions between states
  - Maybe labeled by **action** causing transition

## Search as a Path-Finding Algorithm

- Each node represents a state with all problem-relevant information.
- Each action represented by an edge to the next state

We can search find a path from the start state to a state that represents a solution

## Search and Rule-based Systems

- Each node represents set of known/proved facts.
- Each edge represents adding a new fact.

## Typical Search Assumptions

- Start node is known
- If you see the goal state node, then you reconginze it as a goal state.
- From any node, you can identify all possible actions/edges/transitions to other states
- But do not necessarily have a list of all nodes up front.
  - Explicit list given up front
  - Implicit: maybe a function or experiment.

### 1.BFS(Breadth-First-Search):

<img src="/Users/jimmytoluene/Library/Application Support/typora-user-images/image-20251014210546245.png" alt="image-20251014210546245" style="zoom:50%;" />

- Follow edges from the earliest discovered nodes first.

  **Completeness:** Yes

  **Optimality:** Yes, using unit cost per edge

  **Time Complexity:**

  $O(number\;of\;states)*O(time\;to\;get\;transitions) = O(|V|+|E|)$

  **Space Complexity:**

  Queue size (For tracking visited states) = $O(number\;of\;states)$

```python
from collections import deque

def bfs(graph,s,t):
    """
    Breadth-First Search to find path from start node s to target node t.
    
    Args:
        graph: Dictionary representing adjacency list {node: [neighbors]}
        s: Start node
        t: Target node
    
    Returns:
        List representing path from s to t, or None if no path exists
    """
    if s == t:
      return [s]
    
    queue = deque([s,[s]])
    
    visited = {s}
    
    while(queue):
      # Dequeue the front node(first node) abd its path
      current, path = queue.popleft()
      # explore all neighbors
      for neighbor in graph.get(current,[]):
        #Skip to next iteration if this neighbor is already visited
        if neighbor in visited:
          continue # continue = skip to NEXT iteration
        #Mark this node to be visited
        visited.add(neighbor)
        
        new_path = path + [neighbor]
        # Check if we found our target node
        if neighbor == t:
          return new_path
        #Otherwise, just add to queue for further exploration
        queue.append((neighbor,new_path))
    #No path found
    return None
        
```

### 2. DFS (Depth-First-Search)

<img src="/Users/jimmytoluene/Library/Application Support/typora-user-images/image-20251014213523728.png" alt="image-20251014213523728" style="zoom:50%;" />

Follow edges from the most recently discovered nodes first

**Completeness:** No, DFS not always find one solutions

```python
Problem 1: A graph will go down to infinite brach
Graph:
    A
   / \
  B   C
  |    \
 B1    GOAL
  |
 B2
  |
 ... (infinite)
```

**Optimality:** No, because DFS always find the SHORTEST path (minimal number of edges)

```python
Graph:
 A (start)
    / \
   B   C
  /     \
 D       F (goal)
  
Two path exist:
(DFS ALWAYS SELECT THIS)1.A->B->D->C->F(goal)
2.A->C->F
```

**Time Complexity:**

$O(number\;of\;states)*O(time\;to\;get\;transitions) = O(|V|+|E|)$

**Space Complexity:**

Queue size (For tracking visited states) = $O(number\;of\;states)$

```python
def dfs_recursive(graph, s, t, visited=None, path=None):
  if visited = None:
    visited = set[]
  if path = None:
    path = []
  visited.add[s]
  path = path + [s]
  
  #Base case: found target
  if s == t:
    return path
  #Recursive case: explore each neighbor
  for neighbor in graph.get(s,[])
  	if neighbor not in visited:
      result = dfs_recursive(graph,neighbor,t,visited,path)
      if result is not None:
        return result
  # No path found
  rerutn None
```

### 3.Iterative Deepening DFS

It is an depth-bounded DFS with increasing depth bounds

![à°µà°°à± à°šà± à°µà°²à± à°²à± à°¯à°¾à°¬à± à°¸à±](https://ai2-iiith.vlabs.ac.in/exp/iterative-deepening-dfs/images/iddfs.png)

**Completeness:** Yes

**Optimality:** Yes

**Time Complexity:**

$O(number\;of\;states)*O(time\;to\;get\;transitions) = O(b^d)$

**Space Complexity:**

Queue size (For tracking visited states) = $O(d)$

```python
def dfs_limited(graph, s, t, depth_limit, visited=None, path=None):
  if visited = None:
    visited = set[]
  if path = None:
    path = []
    
  visited.add[s]
  path = path + [s]
  
  # Base case: found target
  if s == t:
    return path
  #Reached depth limit - stop exploring deeper
  if len(path)>depth_limit:
    return None
  #Recursive case: explore each neighbor
  for neighbor in graph.get(s,[])
  	if neighbor not in visited:
      result = dfs_limited(graph,neighbor,t,depth_limit,visited,path)
      if result is not None:
        return result
  # No path found
  rerutn None
  
def iterative_deepening(graph,s,t,max_depth=100):
  for depth in range(max_depth):
    result = dfs_limited(graph,s,t,depth)
    if result is not None:
      return result
  return None
```

### Bidirectional Search

**Core idea: ** Search from BOTH the start and goal simultaneously until they meet (overlap)

**Algorithm Steps:**

1. Initialize two searches:
   - Forward BFS: starts from start node
   - Backward BFS: starts from goal node
2. Alternate expansion:
   - Expand forward frontier by one level
   - Expand backward frontier by one level
3. Check for intersection:
   - After each expansion, check if any node appears in BOTH visited sets
   - If yes → the searches have met!
4. Reconstruct path:
   - Build path from start → meeting point (using forward parent pointers)
   - Build path from meeting point → goal (using backward parent pointers)
   - Combine them

## Search with Costs:

**Core idea:** When do searching for shortest path from start state to goal state, some edges (actions) are more expensive than others.

- We add **costs** on each edge

### Dijkstra's Algorithm

It is common single source shortest path algorithm.

Assumption: no negative costs

![Dijkstra's Algorithm](https://www.researchgate.net/profile/Mohammed-Al-Ibadi/publication/271518595/figure/fig1/AS:360670886416384@1463002048984/a-Network-topology-b-Steps-of-Dijkstra-algorithm.png)

```python
def Dijkstra(graph,s,t):
  distances = {s:0}
  parents = {s:None}
  
  pq = [(0,s)]
  
  visited = set()
  
  while pq:
    current_dist, current = heapq.heappop(pq)
    
    if current in visited:
      continue
    
    visited.add(current)
    if current == t:
      path_reconstruct(parents,s,t)
      return path,current_dist
    
    for neighbor, weight in graph.get(current,[]):
      #if neighbor is visited
      if neighbor in visited:
        continue
      
      new_dist = current_dist + weight
      
      if neighbor not in distances or new_dist < distances[neighbor]:
        distances[neighbor] = new_dist
        parents[neighbor] = current
        heapq.heappush(pq,(new_dist,neighbor))
    
    
```

### A* Algorithm

A* algorithm = Dijkstra + Heuristic

Key idea: Use heuristic to guide search toward goal. 
$$
f(n) = g(n) + h(n)    \\

g(n) = actual\;cost\;from\;start\;to\;n \\       

h(n) = estimated\;cost\;from \;n \;to\; goal\; (heuristic)\\       

f(n) = estimated \;total\; cost\; through\; n
$$
**Admissible Heuristic:**

A heuristic $h(n)$ is **admissible** iff:
$$
h(n)\le h*(n)
$$
where:

$h(n)$ = estimated cost from node $n$ to the goal

$h^*(n)$ =  actual optimal cost from node $n$ to the goal  

```python
def a_star(graph,s,t,heuristic):
  g_costs = {0:s}
  parents = {s:None}
  # Initialized an pq:(f_cost,node)
  # f_cost = g(n) + f(n)
  h_s = heuristic(s)
  pq = [(h_s,s)]
  # Visited nodes
  visited = set()
  while pq:
      # Pop node with lowest f(n) = g(n) + h(n)
      f_cost, current = heapq.heappop(pq)
        
      if current in visited:
          continue
        
      visited.add(current)
      g_current = g_costs[current]
      h_current = heuristic(current)
        
        
      # Found goal!
      if current == goal:
          path = reconstruct_path(parents, start, goal)
          return path, g_current
        
      # Explore neighbors
      for neighbor, edge_cost in graph.get(current, []):
          if neighbor in visited:
              continue
            
          # Calculate g(neighbor) = g(current) + edge_cost
          g_neighbor = g_current + edge_cost
            
          # Update if found better path
          if neighbor not in g_costs or g_neighbor < g_costs[neighbor]:
              g_costs[neighbor] = g_neighbor
              parents[neighbor] = current
                
              # Calculate f(neighbor) = g(neighbor) + h(neighbor)
              h_neighbor = heuristic(neighbor)
              f_neighbor = g_neighbor + h_neighbor
                
              heapq.heappush(pq, (f_neighbor, neighbor))
        
        print()
    
  print(f"✗ No path found")
  return None, float('inf')
```



# Markov Decision Process

## 1.Markov Property and Markov Processes

### Markov Property

1. Only the current state matters to predict next state.

2. "Memoryless"

### Vaiations on Markov Processes

- **Finite** vs infinite 
- **Discrete** vs continuous time
- **Fully observable** vs hidden state

## 2.Finite Markov Processes:

If a **Markov Processes** has **finite number of states**, we can order them as $ S_1,...S_n$
$$
p_{i,j} = Pr[S_{t+1}=S_j|S_t=S_i] \\

​							p_{i,j} = Pr[S_{t+1}=j|S_t=i]
$$

---

### Transition Matrices:

The **all transition** of finite Markov Processes

- Total size is $n*n$ size matrix, which $n$ means total number of states.

$$
\begin{bmatrix}
p_{1,1} &\cdots& p_{1,n} \\
\vdots  &\ddots& \vdots  \\
p_{n,1} &\cdots& p_{n,n} \\
\end{bmatrix}
$$
$i\;row$ : Probability distribution of next state based on current state $i$

### Tricks with Transition Matrices

• There is a lot of analysis that can be done with transition
matrices…
• $𝐏^k$ is a 𝑘 step transition matrix.
• Eigenvectors and eigenvalues tell you about steady state distributions.  

## Markov Reward Processes

A Markov Reward Process = Markov Process + Rewards

•  Like transitions in a Markov process, the next reward only depends on the current state.
•  At time 𝑡, receive reward 𝑅𝑡 based on $S_{t-1}$.

•  Let ℛ𝑠 denote the **average reward** after being in state 𝑠. 

Then:
$$
ℛ𝑠 = E[𝑅_{𝑡+1}|𝑆_{t} = 𝑠]
$$

### Evaluating Markov Reward Processes

$v_*(s)$ (the *state value*) is the **expected total future reward**, starting from s.
$$
v_*(S_{t})=E[(\sum_{t'>t}R_{t'})|S]
$$

- $v_*$ is the value function of this process.

- It accumulates not just the next reward, but **all subsequent ones — possibly discounted**.
  $$
  v_*(S_{t})=E[(\sum_{t'>t}\gamma^{t'-t-1}R_{t'})|S]
  $$
  
- $\gamma$ is a discount factor where $0<\gamma\le1$

  - Only use $\gamma = 1$ when loops are impossible

### Monte Carlo Evaluation of Markov Reward Process

- We evaluate $v_∗ (S_t )$ by using **Monte Carlo simulation**

  - Given the transition matrix $P$ and expected state rewards ℛ, simulate the process many times and calculate the average
    $$
    v_*{(S_t)}=R_{S_t}+\gamma E[v_*{(S_{t+1})}|S_t]
    $$
    

    And we represent $v_*{(s)}\;and\;R_s$ as vectors in the same order as states.

    Rewrite:
    $$
    v_* = R + \gamma Pv_*
    $$
    

    We can solve linear equations to find $v_*$ (Generally works w/ $r<1$ case)
    $$
    						v_* = R + \gamma P v_*\\
    
    ​						Iv_*-\gamma Pv_* = R\\
    
    ​						(I-\gamma P)v_* = R\\
    
    ​						v_* = (I-\gamma P)^{-1}R\\
    $$
    

## Markov Decision Processes (MDP)

Markov decision processes add actions to the process.

- Transition probabilities an rewards are now depend on current state and action
  $$
  p_{i,j}^a = Pr[S_{t+1}=j|S_t=i,A_t=a]\\
  
  ​						R_{s}^a=E[R_{t+1}|S_t = s,A_t = a]
  $$
  

---

### Evaluating MDP:

Bellman equation expressing optimal value:

Rewrite:
$$
v_*(S_{t})=\max_aE[(\sum_{t'>t}R_{t'})|S]
$$
to:
$$
v_*{(S_t)}=\max_a E[R_{t+1}+\gamma v_*{(S_{t+1})}|S_t,A_t = a]
$$
Finally we got:
$$
v_*(S_t)=\max_a[R_{S_t}^a + \gamma E[v_*(S_{t+1})|S_t,A_t=a)]]
$$


At state t with action a, the total expected rewards that from current step receivied reward and the future steps average reward with discount factor

### Policy

A policy is a function mappping states to actions.

- A deterministic policy returns a single action.
- A probabilistic policy returns a probability distribution

Usually denoted as $\pi$ or $\pi(s)$ with subscripts for context...

#### Representing Policies for Finite MDP

- Represent a deterministic policy $\pi$ as a table of $n$ actions
- Represent a probabilistic policy $\pi$ as an $n *k$ matrix for $n$ states and $k$ actions

$$
v_\pi(S_t)=E[R_{t+1}|S_t,\pi]+\gamma E[v_\pi(S_t,\pi)]
$$



Previously:
$$
v_*{(S_t)}=\max_a E[R_{t+1}+\gamma v_*{(S_{t+1})}|S_t,A_t = a]
$$
For a specific policy $\pi$ (optimal or not):
$$
v_\pi(S_t)=E[R_{t+1}+\gamma v_*(S_{t+1})|S_t,\pi]
$$


to:
$$
v_\pi(S_t)=E[R_{t+1}|S_t,\pi]+\gamma E[v_\pi(S_t,\pi)]
$$
**with** 	
$$
R_s^\pi=E[R_{t+1}|S_t,\pi],\;\;\;\;\;\;\;\;\;p_{i,j}^\pi = \sum_a Pr[At=a|S_t=i,\pi]Pr[S_{t+1}|S_t=i,A_t = a]\\
\\
P^\pi =\begin{bmatrix}
p_{1,1}^\pi &\cdots& p_{1,n}^\pi \\
\vdots  &\ddots& \vdots  \\
p_{n,1}^\pi &\cdots& p_{n,n}^\pi \\
\end{bmatrix}
$$
to: 
$$
v_\pi = R^\pi + \gamma P^\pi v_\pi
$$

#### Optimal Policy 

$$
v_*(s) = \max_\pi v_\pi(s)
$$

A policy $\pi$ is optimal if and only if:
$$
\forall{s}\; v_\pi(s)=v_*(s)
$$

---

# Computing Optimal Policy

Previously for Markov reward processes, we derived this solution.
$$
v_* = (I-\gamma P)^{-1}R
$$

- The interpretation of each row $(I-\gamma P)^{-1}$ is the expected number of $\gamma-discounted$ visits to each state
- If $\gamma=1$, at least one state has an infinite number of visits, the matrix inversion will fail.

---

## Value Iteration

1.Set $v_0=[0,…, 0]$.

2.For $i = 0,1,2,\dots$

​	For all states s,

​		Set $v_{i+1}[s] = \max_a[ R_s^a + \gamma E[v_i(S_{t+1})|S_t,A_t=a]]$

​		Set $v_{i+1} = max(R^\pi+\gamma P^\pi*v_i)$

---

What is $v_i$? 

- Best rewards within $i$ steps

## Policy iteration

- Policy iteration methods iterate on policies instead of value function
- Similar intuitions but often converging in fewer iterations.
- Basic idea:
  - Start with any policy $\pi_i$
  - Calculate $v_\pi$ (Using value iteration)
  - Calculate the best $\pi_{i+1}$ assuming $\pi_{i+1}$ will be used for one step and followed by using $\pi_i$

### Implementations

1. Initialize $\pi_0$ to any policy.

2. Set  $v_0 = v_{\pi_{0}}$ using value iteration

3. For i = 0,1,2,...

   Set $\pi_{i+1}(s)=argmax_a [R_s^a+\gamma E[v_{\pi_{i}}(S_{t+1}|S_t = s ,A_t = a)]]$

   Set $v_{i+1} = v_{\pi_{i+1}}$

   $v_{i+1}(s)= \max_a[R_s^a+\gamma E[v_i(S_{t+1}|S_t=s,A_t=a)]]$

#  Model-Free Reinforcement Learning: TD-Learning and Q-Learning

---

What Can We Do Without That Assumption?

- Observe system, drive some actions to see results, develop (rough) value models.

#### Model-Free Reinforcement Learning

- Model-free = no explicit model for rewards and transitions
  - Value iteration and Policy iteration need model information (Transition Matrix, Average Rewards)

- Reinforment learning 

  

## 1.TD-Learning

For a given policy $\pi$, learn $v_\pi$ without a model of the environment.

- Cannot calculate analytically or use value iteration

**Key idea**

- current state value $v_\pi(S_t)$ and next state value $v_\pi(S_{t+1})$ are closely related.
- Try to **optimize away the "temporal difference"**

Works for Markov decision processes with a policy, but same algorithm works for Markov reward processes.

### 1.1 Intuition for TD Learning

- For a particular policy $\pi$, there is a known relationship between the current state value and the expected next state value.

- INPUT: 
  $$
  v_\pi(S_t) = E[R_{t+1}+\gamma v_\pi{(S_{t+1})}|S_t,\pi]
  $$

- Every time we observe a transition from $S_{t}$ to $S_{t+1}$ and receive reward $R_{t+1}$ , 

  - we use current estimate of $v_\pi(S_{t+1})$ to get sample value for $v_\pi(S_{t})$ , If  $v_\pi(S_{t+1})$ is correct,  then $R_{t+1} + \gamma v_\pi(S_{t+1})$ is unbiased estimate of  $v_\pi(S_{t})$

---

### 1.2 The Temporal Difference Learning Algorithm

- **Sample** state transitions following policy $\pi$
  - **Usually this is for testing a policy**
- Pick learning rate $\alpha$
- For each observed state transition from $S_t$ to $S_{t+1}$ receiving reward $R_{t+1}$
  - Update  $v_{WIP}=(1-\alpha)v_{WIP}(S_t)+\alpha(R_{t+1}+\gamma v_{WIP}(S_{t+1}))$

The moving average of Temporal Diffence is exponentially

---

## 2.Q-Learning

**Q-Learning** learns the **action-value function** $Q(s,a)$ instead of the state-value function $v(s)$

Q-Learning learns the **optimal Q-function** $Q_*(s,a)$ directly, without needing to follow the optimal policy ! This is called **off-policy learning**.
$$
q_*(s,a)= E[R_{t+1}+\max_{a'}q_*(S_{t+1},a')|S_t=s,A_t=a]
$$

### 2.1 The Q-Learning Algorithm(Learning):

---

- For i = 0,1,2,...
  - Sample non-final state $s_i$
  - Pick action $a_i$
  - Observe the resulting payoff $r_i$ and next state $t_i$
  - Update $Q[S_i,a_i]=(1-\alpha_i)Q[s_i,a_i]+\alpha_i(r_n+\gamma \max_bQ[t_i,b])$

**QUESTION: In Q-learning, the learning rate alpha is used to bias the weighted average towards more recent data. Why would we want to down-weight the older samples when estimating the value?**

Down-weight older examples allows the agent to **give more importance to recent, higher-quality data collected under improved policies**, which **accelertates convergence** to the optimal Q-values.



### Mininal Example:

3-State MDP

- States: ${S_0,S_1,S_2:terminal}$
- Actions: 
  - At $S_0$: {Right}
  - At $S_1$:{Left,Right}
  - At $S_2$:terminal.
- Rewards: 
  - All transitions: 0
  - Reaching $S_2$ : $r = +1$
- Parameters: 
  - $\gamma = 0.9$ (discount factor)
  - $\alpha = 0.5$ (learning rate)

#### Initial Q-Table

| State | Action | Q-value | Interpretation             |
| ----- | ------ | ------: | -------------------------- |
| S₀    | Right  |       0 | Expected return from start |
| S₁    | Left   |       0 | Go back (suboptimal)       |
| S₁    | Right  |       0 | Go to goal (optimal)       |

**Iteration 0**:

$s_0 = S_0$

Pick action: $a_0$= Right

Observe:

- Payoff: $r_0 = 0$
- Next state: $t_0 = S_1$

**Update**

$Q[S_0,Right]=(1-0.5)*Q[S_0,a_0]+0.5*(r_0+0.9*\max({Q[S_1,Left],Q[S_1,Right])})$

​			 $=0.5*0 + 0.5*(0+0.9*\max({0,0})$ 

​			 $=0$

**After Iteration 0, the Q-Table becomes to**

| State | Action | Q-Value |
| :---- | ------ | ------- |
| $S_0$ | Right  | 0       |
| $S_1$ | Left   | 0       |
| $S_1$ | Right  | 0       |

---

**Iteration 1**:

$s_1 = S_1$

Pick action: $a_1$= Left

Observe:

- Payoff: $r_1 = 0$
- Next state: $t_{1}= S_0$

**Update**

$Q[S_0,Right]=(1-0.5)*Q[S_1,a_1]+0.5*(r_1+0.9*\max({Q[S_0,Right])})$

​			 $=0.5*0 + 0.5*(0+0.9*\max({0}))$ 

​			 $=0$

**After Iteration 1, the Q-Table becomes to**

| State | Action | Q-Value |
| :---- | ------ | ------- |
| $S_0$ | Right  | 0       |
| $S_1$ | Left   | 0       |
| $S_1$ | Right  | 0       |

---

**Iteration 2**:

$s_2 = S_1$

Pick action: $a_2$= Right

Observe:

- Payoff: $r_2 = 1$
- Next state: $t_2 = S_2$

**Update**

$Q[S_1,Right]=(1-0.5)*Q[S_2,a_2]+0.5*(r_2+0.9*\max({Q[S_2,b])})$

​			 $=0.5*0 + 0.5*(1+0.9*0)$ 

​			 $=0.5$

**After Iteration 2, the Q-Table becomes to**

| State | Action | Q-Value |
| :---- | ------ | ------- |
| $S_0$ | Right  | 0       |
| $S_1$ | Left   | 0       |
| $S_1$ | Right  | **0.5** |

---

**Iteration 3**:

$s_3 = S_0$

Pick action: $a_3$= Right

Observe:

- Payoff: $r_3 = 0$
- Next state: $t_3 = S_1$

**Update**

$Q[S_0,Right]=(1-0.5)*Q[S_0,a_0]+0.5*(r_3+0.9*\max({Q[S_1,Left],Q[S_1,Right])})$

​			 $=0.5*0 + 0.5*(0+0.9*\max({0,0.5})$ )

​			 $=0.225$

**After Iteration , the Q-Table becomes to**

| State | Action | Q-Value   |
| :---- | ------ | --------- |
| $S_0$ | Right  | **0.225** |
| $S_1$ | Left   | 0         |
| $S_1$ | Right  | **0.5**   |

---

**Iteration 4**:

$s_4 = S_1$

Pick action: $a_4$= Left

Observe:

- Payoff: $r_4 = 0$
- Next state: $t_4 = S_0$

**Update**

$Q[S_1,Left]=(1-0.5)*Q[S_1,Left]+0.5*(r_4+0.9*\max({Q[S_0,Right])})$

​			 $=0.5*0 + 0.5*(0+0.9*0.225)$ 

​			 $=0$.10125

**After Iteration , the Q-Table becomes to**

| State | Action | Q-Value     |
| :---- | ------ | ----------- |
| $S_0$ | Right  | **0.225**   |
| $S_1$ | Left   | **0.10125** |
| $S_1$ | Right  | **0.5**     |

---

**......... Repeat Until Convergence**

# Hidden Markov Model

## 1.What are hidden Markov models

- Hidden States, probabilistic **observation** ,transitions and rewards.

  - Have models for transition probabilities + observations probabilities
  - But only we see **observations, not states.**

- The true state is unknown

  ---

- **Observable outputs let us make inferences ** about true states

  - We usually cannot **determine true state with 100% certainty**

- Observable output is usually **probabilistic**

  - Even if deterministic, other states might have the same output
  - Otherwise, a unique output will identify true state.

![image-20251014133536789](/Users/jimmytoluene/Library/Application Support/typora-user-images/image-20251014133536789.png)

### Formalizing Hidden Markov Models

- $n$ = number of states
- $m$ = number of distinct observations (aka. alphabet size)

- **$A$** = transition probabilities (Based on current state)

  - $A_{i,j}$ = Probability of state $j$ at the time $t+1$ after state $i$ at time $t$

  - $n * n $ Size, Like $P$ for fully observable.

- **$B$** = **Emission Matrix** = Observation probability Matrix (Based on current state)
  - $b_{i,j}$ is the probability of $j$-th observation from $i$-th hidden state.

  - All rows must sum to 1 (they're probability distributions)

- $\pi$ = initial state probabilities (Assumed)

  - not a policy $\pi$ , the size of it is n * 1

- $\lambda$ = all of these parameters $(A,B,\pi)$ collectively. 
  - (Sometimes $\theta$ also used)

Assume we have $T$ timesteps consideration

- $O = [o_0,o_1,\dots,o_{T-1}] $ observations (known)
- $Q = [q_{0},q_1,\dots,q_{T-1}]$  true states (unknown)

---

## 2.Problem based on hidden Markov models

- ### 2.1 Evaluation: 

  **How likely** is it that this HMM produced these observations O?

  - What is $Pr⁡[O│λ]$?

  - Given state $q_t$ at time $t$, what is the probability of that observations $o_t,\dots,o_{T-1}$ will match desired observations $O$?

  Answer:

  - Let $p_b(t,i)= Pr[o_t,\dots,o_{T-1}|q_t = i,\lambda]$

  - What is $Pr[O|\lambda]$ ?

    ---

    #### Backward algorithm

    **Base Case**

    At the final step $T$ :

    ​								$p_b(t=T,i) = 1\; \forall i$

    ​							

    **Induction Step**

    For $t = T-1,T-2,\dots,1:$

    ​				 	 $p_b(t<T,i)= b_{i,O{t+1}}*\sum_{j=1}^Na_{ij}*p_b(t+1,j)$

    

    $Pr[O|\lambda] = \sum_{i=1}^N\pi(i)*b_{i}(O_1)*\beta_{1}(i)$

    **Setup (size $N$ states, length $T$ sequence):**
  
    - Transition $A \in \mathbb{R}^{N\times N}$, $A_{ij}=P(X_{t+1}=j\mid X_t=i)$
    - Emission $B \in \mathbb{R}^{N\times |\mathcal{O}|}$, $B_{j,o}=P(O_t=o\mid X_t=j)$
    - Initial $\pi \in \mathbb{R}^N$
    - Observation indices $O[0{:}T-1]$
    - Define $b_t \in \mathbb{R}^N$ by $(b_t)_j = B_{j,\,O[t]}$
    
    **Pseudocode**
  
    ```R
    BACKWARD(o, A, B, c):
      N ← number of states, T ← len(o)
      β ← zeros(T, N)
    
      # t = T-1
      for i: β[T-1,i] = c[T-1]     # scale to match forward
    
      # t = T-2..0
      for t in (T-2) down to 0:
        for i in 0..N-1:
          β[t,i] = sum_j A[i,j] * B[j, o[t+1]] * β[t+1,j]
        for i: β[t,i] *= c[t]      # apply same scaling at time t
    
      return β
    ```
    
    **Python Code**
  
    ```python
    T = len(O)
    beta = np.zeros((T, N))
    
    # initialization
    for i in range(N): # complete this loop
        beta[-1] = 1
    
    # inductive steps
    for t in range(T - 2, -1, -1): # complete this loop
        beta[t] = (A * (B[:,O[t+1]]*beta[t+1])[None,:]).sum(axis = 1)
        
    print("beta values:")
    print(beta)
    print()
    
    # final answer
    # complete this part
    answer = (pi * B[:,O[0]] * beta[0]).sum()# this variable should store your final answer
    
    print("P(O | lambda) = " + str(answer))
    ```

---

#### Forward Algorithm

- Let $p_{F}=Pr[o_0,\dots,o_{t-1},q_{t}=i|\lambda]$

  **1.Base Case** , $t=0$

  ​									$p_{F}(t=0,i) = \pi(i)$

  

  **2.Inductive Steps, $t >0$**

  ​							$p_{F}(t>0,i)=\sum_{j}p_{F}(t-1,j)*b_{j,O_{t-1}}*a_{j,i}$


---

### 2.2. Recognition: 

**What sequence of states Q** of this HMM were **most likely to generate these observations O**?

- What is $argmax_Q\;Pr[O|Q,\lambda]$ ?
- What is $argmax_Q\;Pr[q_t = s|O,\lambda]$
- What is $argmax_Q\;Pr[Q,O|\lambda]$

---

#### Viterbi Algorithm

- Given observation $O$, what is the highest probability for a sequence of $t$ states ending in the $i$-th state producting the first $t$ observations
  $$
  p_V(t,i) = \max_{q0,\dots,q_{t-2}} Pr[O_0,\dots,O_{t-2},q_0,\dots,q_{t-2}|q_{t-1}=i,\lambda]
  $$

- $argmax_Q\;Pr[Q,O|\lambda] = max_i (p_V(T,i))$

- $p_V(t=1,i) = \pi(i) * b_{i,0}$

- $p_V(t>1,i) = $ What's the best previous $j$ for prefix of observations and and transitioning to $i$ and getting new observation right
  $$
  =\max_j(p_V(t-1,j)·a_{j,i}·b_{i,o_{t-1}})
  $$

  ---

  

### 2.3. Learning:

 What are the parameters of this HMM given these observations?What is $argmax_Q\;Pr[O|Q,\lambda]$?

### 1. $argmax_λ Pr[O|λ]$

**"Learn the complete HMM model from observations"**

- **λ** represents the entire model: λ = (A, B, π)
- Find parameters that maximize likelihood of seeing observations O
- Most general formulation
- Uses **Baum-Welch algorithm** (EM algorithm for HMMs)
