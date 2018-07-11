class: middle, center, title-slide
count: false

# Some ideas on Probabilistic approaches
for Dimensionality Reduction methods <br>
(potentially with user constraints)

<br><br>

Minh, 11/07/2018
---

class: middle

# Discrete Ideas

- Motivation:
    + Probabilistic ML can be a good direction???
    + More nature way to inject user constraints???

- My thinking:
    + based on t-SNE for InfoViz
    + find the probabilistic approach for DR
    + think about the way to inject (user) constraints into a probabilistic framework.

---

# Combining t-SNE and Gaussian Mixture Models

- Motivation
    + GMM works well in low dimensional space.
    + Can inject user constraints on:
        * the number of clusters
        * the position of some clusters

---

# Using the idea of Variational Inference to optimize t-SNE

---

# Parametric t-SNE

---

# Emsemble t-SNE

---

# AutoEncoder with a Probabilistic Decoder

---


class: middle

# Research Questions

---

# RQ1

How to define a uniform probabilistic model
for both DR and clustering?
- Goal:
    + obtain a visualization with clear structures (as clusters)
    + a 'gateway' to use the constraints that work on clustering methods.

---

# RQ2

How to present (many types of) user constraints
under the probabilistic framework?