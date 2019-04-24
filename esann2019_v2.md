class: middle, center, title-slide
count: true

# User-steering Interpretable Visualization with <br> Probabilistic PCA

<br>

Viet Minh Vu and Benoı̂t Frénay

NADI Institute - PReCISE Research Center

University of Namur, Belgium

25/04/2019

???
Introducing our work on integrating the user's feedbacks on the visualization into a probabilistic DR method.

---

## Problem: Dimensionality Reduction (DR)

.center.width-80[![](figures/esann2019/Fashion_MNIST_samples.png)]
.caption[Samples from the Fashion-MNIST dataset]

.footnote[https://github.com/zalandoresearch/fashion-mnist]

---

## Visualization of high dimensional data

.center.width-60[![](figures/esann2019/FASHION100_init_ppca.png)]
Having an initial visualization with the Probabilistic Principle Component Analysis (PPCA) model ...


---

## Proposed Method: Interactive PPCA (iPPCA)

.center.width-60[![](figures/esann2019/FASHION100_selected_points.png)]
The user can manipulate the visualization by moving some points.


---

## iPPCA Result

.center.width-60[![](figures/esann2019/FASHION100_ippca_result.png)]
The result of the interactive model is explainable to the users.


---

# Motivation
*User interaction in model design and analysis*

.center.width-100[![](figures/esann2019/ml_with_human.png)]
.caption[Visual analytic with Human-in-the-loop]

+ The user can interact directly with the visualization to give their feedbacks.
+ The model can update itself to take into account these feedbacks and produce a new visualization.


.footnote[Sacha, Dominik, et al. "Knowledge generation model for visual analytics." IEEE TVCG 2014]


---
class: smaller
# Existing approaches

*Integrating **user's feedbacks** into existing DR methods*

+ Weighted MDS with the some fixed points to modify the weights $\omega\_{F}$:
.larger[
$$
\mathbf{Y} = \text{argmin}\_{\mathbf{Y}}
\sum\_{i < j \leq n} \rho 	 \Big| d\_{\omega}(i,j) - d\_{Y}(i,j) \Big| + (1-\rho) \Big | d\_{\omega\_{F}}(i,j) - d\_{Y}(i,j) \Big |
$$
]

+ Semi-supervised PCA with ensemble of Must-links (ML) and Cannot-links (CL):
$$
J(\mathbf{W}) = \frac{1}{2 n^2} \sum\_{i,j} { | \mathbf{x}\_{i} - \mathbf{y}\_{j} | }^2 + \frac{\alpha}{2 n\_{CL}} \sum\_{CL} { | \mathbf{x}\_{i} - \mathbf{y}\_{j} | }^2 - \frac{\beta}{2 n\_{ML}} \sum\_{ML}{ | \mathbf{x}\_{i} - \mathbf{y}\_{j} | }^2
$$

+ Constrained Locality Preserving Projections with ML and CL:
$$
\mathbf{W}  = \text{argmin}\_{\mathbf{W}} \frac{1}{2} \Big( \sum\_{i,j}(\mathbf{y}\_{i} - \mathbf{y}\_{j})^2 \widetilde{M}\_{ij} + \sum\_{ML'}(\mathbf{y}\_{i} - \mathbf{y}\_{j})^2  - \sum\_{CL'}(\mathbf{y}\_{i} - \mathbf{y}\_{j})^2 
\Big)
$$

    - $\mathbf{y}\_{j} = \mathbf{W}^T {\mathbf{x}}\_{j}$, $\mathbf{W}$ is projection matrix, $\mathbf{M}$ is weights matrix
    - ML', CL' are the extended set of Must-links and Cannot-links constraints


---
count: false
class: smaller
# Existing approaches

*Integrating **user's feedbacks** into existing DR methods*

+ Weighted MDS with the some fixed points to modify the weights $\omega\_{F}$:
.larger[
$$
\mathbf{Y} = \text{argmin}\_{\mathbf{Y}}
\color{blue}{\sum\_{i < j \leq n} \rho 	 \Big| d\_{\omega}(i,j) - d\_{Y}(i,j) \Big|} + \color{red}{(1-\rho) \Big | d\_{\omega\_{F}}(i,j) - d\_{Y}(i,j) \Big |}
$$
]

+ Semi-supervised PCA with ensemble of Must-links (ML) and Cannot-links (CL):
$$
J(\mathbf{W}) = \color{blue}{\frac{1}{2 n^2} \sum\_{i,j} { | \mathbf{x}\_{i} - \mathbf{y}\_{j} | }^2} + \color{red}{\frac{\alpha}{2 n\_{CL}} \sum\_{CL} { | \mathbf{x}\_{i} - \mathbf{y}\_{j} | }^2 - \frac{\beta}{2 n\_{ML}} \sum\_{ML}{ | \mathbf{x}\_{i} - \mathbf{y}\_{j} | }^2}
$$

+ Constrained Locality Preserving Projections with ML and CL:
$$
\mathbf{W}  = \text{argmin}\_{\mathbf{W}} \frac{1}{2} \Big( \color{blue}{\sum\_{i,j}(\mathbf{y}\_{i} - \mathbf{y}\_{j})^2 \widetilde{M}\_{ij}} + \color{red}{\sum\_{ML'}(\mathbf{y}\_{i} - \mathbf{y}\_{j})^2  - \sum\_{CL'}(\mathbf{y}\_{i} - \mathbf{y}\_{j})^2}
\Big)
$$

    - $\mathbf{y}\_{j} = \mathbf{W}^T {\mathbf{x}}\_{j}$, $\mathbf{W}$ is projection matrix, $\mathbf{M}$ is weights matrix
    - ML', CL' are the extended set of Must-links and Cannot-links constraints

---
count: false
# Existing approaches

*Integrating **user's feedbacks** into existing DR methods*

+ User's feedbacks $\Longrightarrow$ **Explicit regularization term**
+ Jointly optimized with the <span style="color:blue">Objective function</span> of the basic DR method.

*Problems?*
+ Many discrete methods
+ Design the regularization term explicitly

*$\Longrightarrow$ Need another approach*

---
# Probabilistic approach

.center.width-100[![](figures/esann2019/box_model_revise.png)]

.footnote[David Blei, et al. "Variational Inference: Foundations and Modern Methods". NIPS 2016 Tutorial]


---
count: true

# Existing approaches

Integrating **user's feedbacks** into existing DR methods as a **regularization term**

---
count: true

# Existing approaches and ours

Integrating **user's feedbacks** into $\underbrace{\footnotesize{ \text{existing DR methods} }}_{\Downarrow}$ ~~as a **regularization term**~~

.center[$\text{ \small{a probabilistic dimensionality reduction model} }$]

---
count: true

# Existing approaches and ours

Integrating **user's feedbacks** into $\underbrace{\footnotesize{ \text{existing DR methods} }}_{\Downarrow}$ ~~as a **regularization term**~~

.center[$\text{ \small{a probabilistic dimensionality reduction model} }$]

+ Probabilistic PCA (PPCA) as a simple basic model to work with

+ **User's feedbacks** $\Large \approx$ <span style="color:purple">prior knowledge</span> to (re)construct the model.

.center.width-80[![](figures/esann2019/FASHION100_ippca_compare.png)]

---

count: true

.center.width-80[![](figures/esann2019/box_model_apply.png)]

The user-indicated position of selected points is modelled directly

in the <span style="color:purple">prior distribution</span> of the PPCA model.

---
# A closer look at the PPCA model

## A generative view of the probabilistic PCA model.
+ 2-dimensional data $\color{green}{p(\mathbf{x})}$
+ generated from 1-dimensional latent variable $\color{purple}{p(\mathbf{z})}$

.center.width-100[![](figures/esann2019/Bishop_Figure12.9.png)]

.footnote[Bishop's PRML Figure. 12.9]

---
# Proposed interactive PPCA model

+ $\mathbf{X} = \\{ \mathbf{x}_n \\}$: observed dataset of N data points of D-dimensions.
+ The embedded points in the 2D visualization imply the corresponding latent variables $\mathbf{Z} = \\{ \mathbf{z}_n \\}$.
+ The moved points in the visualization are modelled in the prior distribution of $\mathbf{Z}$
.center.width-70[![](figures/esann2019/pz.png)]
+ The iPPCA model:
$
    \mathbf{x}\_n \mid \mathbf{z}\_n \sim \mathcal{N}(\mathbf{x}\_n \mid \mathbf{W}\mathbf{z}\_n, \; \sigma^{2}\mathbf{I}\_{\_{D}}).
$
+ The inference problem:
$
\mathbf{\theta}\_{\_{MAP}} = \text{argmax}\_{\mathbf{\theta}} \log p(\mathbf{\theta} \mid \mathbf{X})
$
where $\mathbf{\theta}$ represents all model's parameters.
+ The MAP estimate of the latent variables $\mathbf{Z}$ is found by following the partial gradient $\nabla_{\mathbf{Z}} \log p(\mathbf{\theta}, \mathbf{X})$ to its local optima.

---
# How the user prior is handled?
+ The user can fix the position of several interested points, with some **level of uncertainty** ($\sigma_{fix}$)
+ A very small **uncertainty** $\Longrightarrow$ the user is very certain.
+ A large **uncertainty** $\Longrightarrow$ the user is not sure.

.grid[
.kol-1-3[
.center.width-100[![](figures/esann2019/prior_sigmafix.png)]
.caption[user's uncertainty $\sigma_{fix}$]
]
.kol-1-3[
.center.width-100[![](figures/esann2019/prior_small_sigmafix.png)]
.caption[Very small $\sigma_{fix} = 1e-4$: very sure]
]
.kol-1-3[
.center.width-90[![](figures/esann2019/prior_large_sigmafix.png)]
.caption[Large $\sigma_{fix} = 0.2$: very uncertain]
]
]

---
# Evaluation of the iPPCA model
*The workflow:*
+ Show the initial visualization of the (original) PPCA model
+ The user selects and moves some anchor points
+ Reconstruct the iPPCA model to create a new visualization.
    - The uncertainty of the feedbacks ($\sigma_{fix}$) is small
    - Hyper parameters of the optimization process are chosen to be the best

*How to evaluate:*
+ Show how to explain the new visualization
    - The level for which we can understand / explain the visualization is considered as a qualitative measure


---

## Quickdraw dataset

.center.width-100[![](figures/esann2019/quickdraw_eg.png)]
.caption[90 sample images from Quickdraw dataset]

+ Move 6 different points of different groups
+ The global structure of the embedding is preserved

---
## Fashion dataset

.center.width-90[![](figures/esann2019/FASHION100_ippca_compare.png)]
.caption[100 sample images from Fashion dataset]

+ Moves 6 points towards the coordinate axes
+ The goal of this interaction is to re-define the axes in the visualization

---
## Fashion dataset

.grid[
.kol-1-2[.width-100[![](figures/esann2019/FASHION100_ippca_result.png)]]
.kol-1-2[.width-100[![](figures/esann2019/annote_iPPCA_FASHION.svg)]
]]

*How to explain the new axes?*
+ Horizontal axis represents **shape**
+ The vertical axis represents **color density**


---
## Automobile dataset
.center.width-80[![](figures/esann2019/automobile_eg.png)]
.caption[203 data points of the Automobile dataset]
.grid[
.kol-1-2[
*How to explain the new axes?*
+ Horizontal axis: cars' **size**
+ Vertical axis: cars' **power**
]
.kol-1-2[
.width-70[![](figures/esann2019/annote_iPPCA_automobile.svg)]
]
]


---
# Advantage of Probabilistic Approach

*Combination of solid theoretical models and modern powerful inference toolboxes*
+ Take any old-class model or modern generative model
+ Plug into a probability framework (Stan, PyMC3, Pyro, Tensorflow Probability) which support modern inference methods like *Stochastic Variational Inference (SVI)*

*Can easily extend the classic models* (Extend the general generative process)
$$
\mathbf{x}_n \mid \mathbf{z}_n \sim \mathcal{N}(f(\mathbf{z}_n), \sigma^{2} \mathbf{I})
$$
+ in PPCA model, $f(\mathbf{z}_n) = \mathbf{W} \mathbf{z}_n$
+ $f(\mathbf{z}_n)$ can be any high-capacity representation function (a neural net)


---

.grid[
.kol-1-2[.width-100[![](figures/esann2019/DIGITS_PCA.png)]]
.kol-1-2[.width-100[![](figures/esann2019/DIGITS_hd50.png)]]
]
.caption[Embedding of 1797 digits with PCA (on the left) and with modified PPCA (on the right)]

+ The decoder $f(\mathbf{z})$ of PPCA is a simple neural network with one hidden layer of 50 units and a sigmoid activation function.
+ The inference is done by the pyro's built-in SVI optimizer.

.footnote[pyro, Deep Universal Probabilistic Programming, http://pyro.ai/]

---
# Recap

Propose the interactive PPCA model allowing the user to control the visualization

+ **[Why]**  
    - *communicate the analytical result*: e.g., create an explainable visualization
    - explore the visualizations (*"what-if" analysis*)

+ **[How]** The user's feedbacks can be efficiently integrated into a probabilistic model via prior distributions of latent variables.

+ **[Potential]** The probabilistic model is flexible to extend and can be easily optimized by the black-box inference methods.

+ **[Future work]** Focus on the *user's feedback modeling* problem without worrying about the complex optimization procedure.

---
class: middle, center, 
count: true

background-image: url(figures/esann2019/bg.png)
background-opacity: 0.1

# User-steering Interpretable Visualization with <br> Probabilistic PCA

<br>

Viet Minh Vu and Benoı̂t Frénay

NADI Institute - PReCISE Research Center

University of Namur, Belgium

-