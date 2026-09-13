# FermiKAN: The Optimization Journey and Physical Interpretability

This document details the scientific and physical insights discovered during the development and optimization of FermiKAN, a highly compressed Neuro-Symbolic architecture for Quantum Monte Carlo.

## 1. Glass-Box Interpretability & The Journey to RHF Symmetry
The goal of FermiKAN is not just efficiency, but **XAI (Explainable AI) in Physics**. By stripping away the black-box MLPs and extracting the trained KAN weights, we translated the neural network's learned state back into mathematical formulas (LCAO molecular orbitals).

Through this, we discovered an interesting optimization process in the H2 molecule system:
1. **UHF Initialization Trap**: Because our neutral atom initializer (`StochasticHomeAtomInitializer`) starts by placing electrons on isolated atoms, the network begins in a broken-symmetry Unrestricted Hartree-Fock (UHF) state.
2. **The Shortcut**: In the early thousands of iterations, the optimizer uses non-linear Backflow to simulate a bond while keeping the LCAO orbitals trapped in this asymmetric UHF state to easily lower Coulomb repulsion.
3. **Autonomous Convergence to Truth**: However, as we run the optimization to 10,000 iterations, the model seeks the true physical ground state. We observed the LCAO coefficients autonomously merging into a symmetric **Restricted Hartree-Fock (RHF)** covalent bond ($\sigma_g$).

**The Bitter Lesson of Physics-ML:** Highly compressed physical models initially struggle to cross symmetry-breaking energy barriers using first-order optimizers like Adam. However, the architecture possesses the expressivity to capture dynamic correlation within a single spatial orbital. Given enough time, the AI may autonomously rediscover some of the quantum chemistry phenomena.

---

## 2. Expressivity vs. Optimization Dilemma (UHF Symmetry Breaking in LiH)
While the PD-KAN architecture remarkably reaches near-FCI energy for LiH (e.g., -8.055 Hartree vs. -8.070 Hartree, requiring ~10,000 iterations), "Glass-Box" weight extraction reveals that it achieves this by abandoning textbook covalent bonding. 

Given the freedom of an Unrestricted Hartree-Fock (UHF) regime, the network breaks spatial symmetry (e.g., $\alpha$ localizing on H, $\beta$ localizing on Li d-orbitals) to artificially avoid Coulomb repulsion (Left-Right correlation). Since the exact physical ground state for closed-shell LiH possesses Restricted Hartree-Fock (RHF) symmetry, this behavior poses a fundamental physical question: 
Is the optimizer trapped in a high-dimensional UHF local minimum, or does the current highly-compressed, single-determinant ansatz simply lack the expressivity to capture dynamic correlation without breaking symmetry?

**Next Step:** Apply RHF-like constraints (forcing $\alpha$ and $\beta$ to share spatial orbitals) on LiH. If the constrained energy is lower than the UHF energy, it implies the optimizer was previously trapped in a UHF local minimum. If the constrained energy is higher, it proves the architectural expressivity has hit a mathematical ceiling, necessitating the integration of multiple determinants or dynamic LCAO coefficients.

---

## 3. The Future Vision: A Neuro-Symbolic Engine for Science
Beyond optimization stability, PD-KAN introduces a crucial advantage for the era of AI Scientists: **Interpretability**. Because the KAN parameters converge into an "interpretable physical basis", this architecture can serve as a grounding module. By directly feeding these symbolic, physical parameters back to Large Language Models (LLMs), we can create a **Neuro-Symbolic feedback loop** that debugs agent hallucinations and enables the autonomous discovery of unknown physical phenomena.
