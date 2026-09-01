# FermiKAN: Installation & Usage Guide

This guide will walk you through setting up the environment and running the FermiKAN proof-of-concept experiments.

## 1. Local Environment Setup (Micromamba / Conda)

FermiKAN requires Python 3.10+ and JAX with GPU support. We have provided an `environment.yml` file that includes all necessary dependencies (including FermiNet, Haiku, JAX, and PySCF) to ensure reproducibility.

We highly recommend using `micromamba` or `conda` for environment management.

```bash
# Create the environment from the provided YAML file
micromamba env create -f environment.yml

# Activate the environment
micromamba activate fermikan

# Patch kfac_jax for the latest JAX compatibility (replaces deprecated jax.P)
sed -i 's/jax.P/jax.sharding.PartitionSpec/g' $(python -c "import site; print(site.getsitepackages()[0])")/kfac_jax/_src/utils/parallel.py
```

## 2. HPC Environment Setup (Apptainer / Singularity)

For users running on HPC clusters where Docker is restricted, we provide an `Apptainer.def` file to build a portable `.sif` container.

First, build the container image:
```bash
apptainer build fermikan.sif Apptainer.def
```

Once built, you do not need to install anything on the host machine. You can execute the scripts directly through the container. **Make sure to pass the `--nv` flag** to allow the container to access the host's NVIDIA GPUs!

> [!WARNING]
> **WSL2 Users:** Running Apptainer with the `--nv` flag on Windows Subsystem for Linux 2 (WSL2) may result in `GPU access blocked by the operating system` errors due to namespace/cgroup limitations in Microsoft's virtualized GPU passthrough. If you encounter this and JAX falls back to CPU (becoming extremely slow), we recommend running the project locally via `micromamba` on your WSL2 host, and reserving the Apptainer `.sif` for native Linux HPC clusters.

```bash
# Example execution using the container
apptainer exec --nv fermikan.sif python3 run_pdkan_ferminet.py
```


## 3. Running the FermiKAN Experiment (H2 Molecule)

To execute the main proof-of-concept on the H2 molecule, simply run the wrapper script. 
This script monkey-patches the original FermiNet architecture with our PD-KAN implementation and executes a zero-shot, Adam-optimized VMC run for 10,000 steps.

```bash
python3 run_pdkan_ferminet.py
```

*Note: The script is configured to save checkpoints every 2.0 minutes in the `exp_fermiKAN_H2/` directory. The entire run takes approximately 5-6 minutes on a modern GPU.*

## 4. Extracting and Analyzing the Weights (Glass-Box XAI)

The most exciting part of this project is proving that we can extract the physical equations from the trained neural network.

After the training finishes, run the analysis script to extract the LCAO coefficients, static/dynamic breathing parameters ($\xi$), and virtual shell polynomials from the latest checkpoint:

```bash
python3 analyze_weights.py
```

The script will output the mathematical parameters directly to the console and to `weights_analysis_output.txt`. 
