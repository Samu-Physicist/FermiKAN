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

### Acknowledgements / Prior Work
This project is heavily inspired by and builds upon the foundational work of FermiNet by Google DeepMind (Pfau et al., 2020).
While FermiNet beautifully demonstrated the power of deep learning in ab-initio quantum chemistry, PD-KAN introduces another approach by incorporating LCAO into the base functions of Kolmogorov-Arnold Networks (KANs). We deeply respect the original FermiNet team and the KAN authors (Ziming Liu et al., 2024) for their groundbreaking contributions and for paving the way in AI for Science.

---

## 🚀 Quick Start & Installation

For a detailed walkthrough on setting up the JAX environment and running the experiments, please refer to our comprehensive guide:

👉 **[Read the Full Installation & Usage Manual (manual.md)](manual.md)**

---

## The Academic Core: FermiKAN (QMC Implementation)

In Quantum Mechanics, standard models face highly irregular loss landscapes due to electron-electron and electron-nucleus singularities. **FermiKAN** applies the PD-KAN framework to eliminate these bottlenecks not by brute-force computation, but through geometry.

### 1. $\epsilon$-Manifold Embedding (Geometric Smoothing)
By mapping the physical $\mathbb{R}^3$ coordinates into a bounded 4D manifold using an $\epsilon$-regularization mapping:

$\mathbf{q} = \frac{1}{\sqrt{x^2 + y^2 + z^2 + \epsilon^2}}(x, y, z, \epsilon)$

the framework naturally eliminates coordinate singularities (division-by-zero at the origin) and polynomial divergences (at infinity), providing an intrinsically well-conditioned, bounded terrain for evaluating angular representations ($\epsilon$-Regularized Solid Spherical Harmonics).

### 2. Analytic Integration of Kato's Cusp Conditions
We explicitly embed Kato's cusp conditions into the KAN edges. By analytically absorbing the singular behavior at particle coincidence, the KAN is freed to focus solely on learning the smooth, residual many-body correlations—acting as a true, adaptive physical basis set.

### 3. Parameter Compression & Speedup
By utilizing physics-informed structural biases, FermiKAN achieves a staggering **~330x reduction in parameters** (1,328 vs 437,200) compared to the baseline FermiNet for the H2 molecule, while still converging to decent accuracy (-1.17 Hartree). This massive compression also translates to faster forward-pass execution (wall-time reduced from 687s to 115s).

### 4. Glass-Box Interpretability & Autonomous Hybridization
The goal of PD-KAN is not just efficiency, but **XAI (Explainable AI) in Physics**. By stripping away the black-box MLPs and extracting the trained KAN weights, we successfully translated the neural network's learned state back into mathematical formulas (LCAO molecular orbitals) and extracted the corresponding 1-particle Reduced Density Matrix (1-RDM). 
Through this, we discovered a stunning result: initialized with purely spherical $1s$ orbitals, the network autonomously discovered **chemical hybridization** (mixing in $p_z$ polarization functions) to stretch the electron cloud along the internuclear axis and form a textbook covalent $\sigma$ bond. By comparing the $\alpha$ and $\beta$ spin channels, we observed that FermiKAN achieved this without resorting to **severe** Unrestricted Hartree-Fock (UHF) style symmetry breaking (spin contamination). This indicates that FermiKAN can act as an interpretable neuro-symbolic engine for scientific discovery.

---

## The Future Vision: A Neuro-Symbolic Engine for Science

Beyond optimization stability, PD-KAN introduces a crucial advantage for the era of AI Scientists: **Interpretability**. Because the KAN parameters converge into an "interpretable physical basis", this architecture can serve as a grounding module. By directly feeding these symbolic, physical parameters back to Large Language Models (LLMs), we can create a **Neuro-Symbolic feedback loop** that debugs agent hallucinations and enables the autonomous discovery of unknown physical phenomena.

---

## Known Issues & Next Steps ⚠️

### 1. The Autonomous Hybridization Discovery
While the 330x compression and high accuracy are fantastic, our "Glass-box" interpretability tool (`analyze_weights.py`) revealed something even more profoundly interesting. The network autonomously learned to mix $p_z$ components into the base $s$ orbitals to form a highly directional covalent bond, mimicking human-derived molecular orbital theory (hybridization). Furthermore, it managed to capture significant correlation energy (via the Jastrow factor) while largely preserving the spatial symmetry between the $\alpha$ and $\beta$ spin channels, avoiding severe UHF-style spin contamination.
**Next Step:** We are developing automated post-processing pipelines to extract Natural Bond Orbitals (NBO) directly from the learned density matrices, allowing quantum chemists to directly interact with and interpret the AI's learned wavefunction.

### 2. The Engineering Challenge (K-FAC)
While the mathematical foundation of PD-KAN successfully narrow the loss landscape, scaling this architecture to massive systems (e.g., Benzene) exposes a critical engineering bottleneck.

Currently, for small-scale PoCs (like H2 dissociation), the geometrically smoothed landscape allows the model to converge rapidly using only **first-order optimization (Adam)**. 

**However, the official K-FAC optimizer in the upstream FermiNet repository is currently broken under recent JAX/XLA updates.** To unlock the potential of FermiKAN on massive, highly correlated systems, the revival of second-order optimization (K-FAC) is an absolute necessity. 

I am an R&D researcher at a chemical manufacturer and have pushed the math as far as I can. 
**I am actively looking for:**
1. **JAX/XLA wizards** to help revive K-FAC for this architecture.
2. **Quantum Chemists & Physicists** to brainstorm and implement elegant physical constraints to identify and resolve any other unforeseen physical artifacts!

PRs, forks, and discussions are highly welcome!

---

## Repository Structure
- `ferminet/` : Modified JAX codebase integrating the PD-KAN architecture.
- `articles/` : Drafts and explanatory articles regarding the architecture and methodology.
- *Full implementation and PoC execution scripts will be uploaded in upcoming commits.*

---

## References
1. Pfau, D., Spencer, J. S., Matthews, A. G. D. G., & Foulkes, W. M. C. (2020). Ab initio solution of the many-electron Schrödinger equation with deep neural networks. *Physical Review Research*, 2(3), 033429.
2. Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., ... & Tegmark, M. (2024). KAN: Kolmogorov-Arnold Networks. *arXiv preprint arXiv:2404.19756*.
3. Kato, T. (1957). On the eigenfunctions of many-particle systems in quantum mechanics. *Communications on Pure and Applied Mathematics*, 10(2), 151-177.

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
