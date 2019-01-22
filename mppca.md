class: middle, center, title-slide
count: false

# Mixture of PPCA

<br><br>

Minh, 21/01/2019

---
exclude: true

### Summary last meeting

+ Do full Bayesian inference
    * Apply for MPPCA model
    * Redo for PPCA model (ESANN paper)

+ Make slides to explain MPPCA (~10min)
    * idea? model? priors?
    * equations (in Bishop)
    * how to implement them?
    * how to intergrate the constraints


+ Other stuffs:
    * Print paper Mixture PPCA [Tipping1999b]
    * Print lastest version of the survey

---
exclude: true

### Ref

+ AutoGuide AutoDelta: http://docs.pyro.ai/en/dev/contrib.autoguide.html#autodelta

---

# Mixture of PPCA

#### Motivation:

* Mixture model is a tool for soft-clustering, outputs the <span style="color:blue">clustering assignments</span> for each datapoint, <span style="color:red"> but we can not visualize the groups of datapoints in 2D</span>.
* PPCA can reduce the dimensions of data, <span style="color:red"> but we still use the supervised labels to color the points in the visualization</span>.<br/>

--
count: false

* The group of points in 2D  ${\ne}$  cluster of points in high dim. (HD)

--
count: false

* Mixture of PPCA model can infer:
    - the <span style="color:blue">2D positions</span> for each data point
    - and their <span style="color:blue">clustering assignment</span>
* The expected visualization with MPPCA:
    - the points of the same cluster in HD are placed close together and/or in the same group in 2D.
    - multi-views visualization: having multiple visualizations for each component <br/> ${\to}$ discover the local structure of each component.

---

#### Input:
* Observed data: $ \mathbf{X} = \\{ \mathbf{x}_{i} \\} $, $ \quad \mathbf{x}_i  \in \mathbb{R}^{D}, \quad  i=1,..,N$

--
count: false

#### Output:
* Latent position in 2D: $ \mathbf{Z} =  \\{ \mathbf{z}_i \\} $, $\quad \mathbf{z}_i  \in \mathbb{R}^{M}, \quad i=1,..,N$
* Latent clustering assignment: $ \mathbf{G} = \\{ g\_{ik} \\} $, $\quad g\_{ik} \in \\{0, 1\\}, \quad k=1,..,K$ <br/>
($g\_{ik} = 1$ iif $\mathbf{x}\_{i}$ belongs to the $k^{th}$ component)

--
count: false

#### Evaluation:
* Visualization quality (hard to measure)
* Clustering quality (e.g. VMeasure)

---

#### Traditional mixture of Gaussians

+ A discrete indicator variable $g_{ik} \in \\{0,1\\}$ indicates whether the $k^{th}$ component generates the datapoint $\mathbf{x}_i$:

$$
p(\mathbf{x}\_{i} \mid g\_{ik}=1) = \mathcal{N}(\mathbf{x}\_{i} \mid \boldsymbol{\mu}\_k, \sigma\_k \mathbf{I}\_{\_D})
$$

where each component is an isotropic multivariate gaussian with mean $\boldsymbol{\mu}_k \in \mathbb{R}^{D}$ and scalar variance $\sigma_k$.

--
count: false

+ By summing all possible assignment states of each point, its marginal distribution is:
$$ p(\mathbf{x}\_{i})= \sum\_{k=1}^{K} \pi\_{k} \;\mathcal{N}(\mathbf{x}\_{i} \mid \boldsymbol{\mu}\_k, \sigma\_k \mathbf{I}\_{\_D})$$

where the mixing coefficient $\pi_k$ represents the contributon of the $k^{th}$ component.

---

#### Mixture of PPCA

For each component, replace the Gaussian distribution by a PPCA distribution:
$$ p(\mathbf{x}\_{i})= \sum\_{k=1}^{K} \pi\_{k} \; \underbrace{ \text{PPCA}(\mathbf{x}\_{i} \mid \boldsymbol{\mu}\_k, \sigma\_k \mathbf{I}\_{\_D}) }_{\text{in fact, is also a Gaussian}}$$

--
count: false
#### Generative process

Each point $\mathbf{x}_i \in \mathbb{R}^{D}$ is generated from a corresponding latent variable $\mathbf{z}_i \in \mathbb{R}^{M}$ as following:

1. Sample a latent variable $\mathbf{z}_i$ from an unit gaussian.

2. Choose one of $K$ components from which $\mathbf{x}_i$ will be generated.

3. $\mathbf{x}_i$ is mapped from $\mathbf{z}_i$ via a projection matrix $\mathbf{W}_K \in \mathbb{R}^{D \times M}$, then shifted to the center $\boldsymbol{\mu}_k$ and disturbed by a variance $\sigma_k$.

    i.e., $\mathbf{x}\_{i}$ is sampled from
$\text{PPCA}(\mathbf{x}\_{i} \mid k) \equiv
 \mathcal{N}(\boldsymbol{\mu}\_{k} + \mathbf{z}\_{i} \mathbf{W}\_{k}^{T}, \sigma\_{k} \mathbf{I}\_{_{D}})$

---

# Elements of MPPCA model (1)

#### Parameters and their priors

--
count: false
+ Global parameter:
$\boldsymbol{\pi} = [\pi_1, \dots, \pi_K ]$

$$\boldsymbol{\pi} \sim \text{Dirichlet}(\alpha_0), \qquad \alpha_0 = \overbrace{ \big[\textstyle\frac{1}{K}, \dots, \textstyle\frac{1}{K} \big] }^{K \text{elements}}$$

--
count: false
+ Local parameters for each components:

$$\boldsymbol{\mu}\_{k} \sim \mathcal{N}(\mathbf{0}\_{\_{D}}, \mathbf{I}\_{\_{D}})$$

$$\sigma_k \sim \text{LogNormal}(\sigma_k \mid 0, 1)$$

$$\mathbf{W}\_{k}^{(j)} \sim \mathcal{N}(\mathbf{W}\_{k}^{(j)} \mid \mathbf{0}\_{\_{M}}, \mathbf{I}\_{\_{M}})$$

where $\mathbf{W}\_{k}^{(j)}$ is a row of the projection matrix
$\mathbf{W}\_{k} = 
\begin{bmatrix}
\mathbf{W}\_{k}^{(1)} \\\\ 
\vdots \\\\
\mathbf{W}\_{k}^{(D)} 
\end{bmatrix}
$

---

# Elements of MPPCA model (2)

#### Latent variables and their priors

--
count: false
Latent variables for each datapoint:

+ Latent position in low dimensional space:

$$
\mathbf{z}\_{i} \sim \mathcal{N}(\mathbf{z}\_{i} \mid \mathbf{0}\_{\_{D}}, \mathbf{I}\_{\_{D}})
$$


+ Latent clustering assignment:

The cluster id for each $\mathbf{x}\_{i}$ is assigned exactly one of $K$ components (categories): $\mathbf{g}\_{i} = \underbrace{[0, \dots, 1, \dots, 0]}\_{K elements}$.

Thus the clustering assignment is modeled by the discrete Categorical distribution: 
$$
\mathbf{g}_{i} \sim \text{Categorical}(\text{probs}=[\pi_1, \dots, \pi_K ])
$$

---

# Elements of MPPCA model (3)

#### Observed variables

Based on the clustering assigment of each point $\mathbf{g}_i$, we determine the component $k$ to which the datapoint $\mathbf{x}_i$ belongs:

$$
\begin{aligned}
\mathbf{x}\_{i} \mid \mathbf{z}\_{i}, k &\sim \text{PPCA}(\mathbf{x}\_{i} \mid \mathbf{z}\_{i}, k) \\\\
& \equiv \mathcal{N}(\mathbf{x}\_{i} \mid \mathbf{z}\_{i},  k) \\\\
& \equiv \mathcal{N}(\mathbf{x}\_{i} \mid \boldsymbol{\mu}\_{k} + \mathbf{z}\_{i} \mathbf{W}\_{k}^{T}, \sigma\_{k} \mathbf{I}\_{_{D}})
\end{aligned}
$$

---

# The model

$\boldsymbol{\mu} = [ \boldsymbol{\mu}\_{k} ], \quad
\boldsymbol{\sigma} = [ \sigma\_{k} ], \quad
\mathbf{W} = [ \mathbf{W}\_{k} ], \quad
k = 1, \dots, K
$

$\boldsymbol{\theta} = \\{ \boldsymbol{\mu}, \boldsymbol{\sigma}, \mathbf{W}, \mathbf{Z}, \mathbf{G} \\}$ : a set of all params and latent random variables.