# FermiKAN: Physics-Designed Kolmogorov-Arnold Networks for Neural QMC

![Work in Progress](https://img.shields.io/badge/Status-Work_in_Progress-orange)
[![DOI](https://zenodo.org/badge/1327641826.svg)](https://doi.org/10.5281/zenodo.21849868)

## Overview
This repository introduces **PD-KAN (Physics-Designed Kolmogorov-Arnold Networks)**, a universal meta-architecture for physics-informed machine learning, alongside its first concrete implementation: **FermiKAN** for Neural Quantum Monte Carlo (Neural QMC).

### The PD-KAN Universal Framework
Physical systems (from quantum mechanics to fluid dynamics) are often plagued by mathematical singularities and boundary conditions that cause severe optimization instabilities in standard neural networks. **PD-KAN** solves this by establishing a universal, two-step architectural paradigm:
1. **Geometric Manifold Mapping**: Analytically mapping the physical coordinates (and their inherent singularities/symmetries) into an intrinsically smooth geometric manifold.
2. **Adaptive KAN Residuals**: Deploying Kolmogorov-Arnold Networks (KAN) on this smooth manifold to learn the true, underlying physical correlations (residuals) without fighting mathematical divergences.

While this framework is broadly applicable to any field governed by geometric singularities, this repository presents its first proof-of-concept in the realm of quantum chemistry.

---

## The Academic Core: FermiKAN (QMC Implementation)

In Quantum Mechanics, standard models face highly irregular loss landscapes due to electron-electron and electron-nucleus singularities. **FermiKAN** applies the PD-KAN framework to eliminate these bottlenecks not by brute-force computation, but through geometry.

### 1. $\epsilon$-Manifold Embedding (Geometric Smoothing)
By mapping the physical $\mathbb{R}^3$ coordinates into a bounded 4D manifold using an $\epsilon$-regularization mapping:

$\mathbf{q} = \frac{1}{\sqrt{x^2 + y^2 + z^2 + \epsilon^2}}(x, y, z, \epsilon)$

the framework naturally eliminates coordinate singularities (division-by-zero at the origin) and polynomial divergences (at infinity), providing an intrinsically well-conditioned, bounded terrain for evaluating angular representations ($\epsilon$-Regularized Solid Spherical Harmonics).

### 2. Analytic Integration of Kato's Cusp Conditions
We explicitly embed Kato's cusp conditions into the KAN edges. By analytically absorbing the singular behavior at particle coincidence, the KAN is freed to focus solely on learning the smooth, residual many-body correlations—acting as a true, adaptive physical basis set.

---

## The Future Vision: A Neuro-Symbolic Engine for Science

Beyond optimization stability, PD-KAN introduces a crucial advantage for the era of AI Scientists: **Interpretability**. Because the KAN parameters converge into an "interpretable physical basis", this architecture can serve as a grounding module. By directly feeding these symbolic, physical parameters back to Large Language Models (LLMs), we can create a **Neuro-Symbolic feedback loop** that debugs agent hallucinations and enables the autonomous discovery of unknown physical phenomena.

---

## The Engineering Challenge: Call for Collaborators 🚀

While the mathematical foundation of PD-KAN successfully smooths the loss landscape, scaling this architecture to massive systems (e.g., Benzene) exposes a critical engineering bottleneck.

Currently, for small-scale PoCs (like H2 dissociation), the geometrically smoothed landscape allows the model to converge rapidly using only **first-order optimization (Adam)**. 

**However, the official K-FAC optimizer in the upstream FermiNet repository is currently broken under recent JAX/XLA updates.** To unlock the true potential of FermiKAN on massive, highly correlated systems, the revival of second-order optimization (K-FAC) is an absolute necessity. 

I am a theoretical physics student and have pushed the math as far as I can. **I am actively looking for JAX/XLA wizards to help revive K-FAC for this architecture.** PRs are highly welcome!

---

## Repository Structure
- `ferminet/` : Modified JAX codebase integrating the PD-KAN architecture.
- `articles/` : Drafts and explanatory articles regarding the architecture and methodology.
- *Full implementation and PoC execution scripts will be uploaded in upcoming commits.*

---

## Citation
If you use the PD-KAN framework or this codebase in your research, please cite our Zenodo release to acknowledge the academic priority of this architecture.

```bibtex
@software{takamatsu_2026_fermikan,
  author       = {Takamatsu, Tomoaki},
  title        = {{FermiKAN: Physics-Designed Kolmogorov-Arnold Networks for Neural QMC}},
  month        = aug,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v0.1.0-alpha},
  doi          = {10.5281/zenodo.21849868},
  url          = {https://doi.org/10.5281/zenodo.21849868}
}
```
