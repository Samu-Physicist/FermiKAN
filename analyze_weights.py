import jax
import jax.numpy as jnp
from ferminet import checkpoint
import os
import sys
import numpy as np

def analyze():
    output_file = "weights_analysis_output.txt"
    sys.stdout = open(output_file, 'w')
    
    ckpt_dir = "exp_fermiKAN_H2"
    if not os.path.exists(ckpt_dir):
        print(f"Error: Checkpoint directory {ckpt_dir} not found.")
        sys.exit(1)
        
    print(f"Looking for checkpoints in {ckpt_dir}...")
    ckpt_file = checkpoint.find_last_checkpoint(ckpt_dir)
    if not ckpt_file:
        print("No checkpoint files found.")
        sys.exit(1)
        
    print(f"Restoring from {ckpt_file}...")
    ckpt_data = checkpoint.restore(ckpt_file, None)
    
    if ckpt_data is None:
        print("Could not restore checkpoint. None returned.")
        sys.exit(1)
        
    if isinstance(ckpt_data, tuple):
        step, data, params, opt_state, mcmc_width, density_state = ckpt_data
        print(f"Restored step: {step}")
    else:
        params = ckpt_data.get("params", ckpt_data)
        
    orbital_key = None
    angular_key = None
    for k in params.keys():
        if k.endswith("fermi_kan__orbitals"):
            orbital_key = k
        elif "vec_angular" in k:
            angular_key = k
            
    if not orbital_key or not angular_key:
        print("Could not find orbital or angular params.")
        sys.exit(1)
        
    orb_params = params[orbital_key]
    ang_params = params[angular_key]
    
    LCAO_coeffs = orb_params.get("LCAO_coeffs", None) # (N_atoms, N_shells, N_MOs) or (1, N_atoms, ...)
    if LCAO_coeffs is None:
        print("No LCAO_coeffs found.")
        sys.exit(1)
        
    if LCAO_coeffs.ndim == 4:
        LCAO_coeffs = LCAO_coeffs[0]
        
    N_atoms, N_shells, N_MOs = LCAO_coeffs.shape
    
    print("\n=== 1. Computing Effective LCAO Tensor (C_tilde) ===")
    
    orbitals_config = [
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 2), (2, 3),
        (3, 0), (3, 1), (3, 2), (3, 3),
        (4, 0), (4, 1), (4, 2), (4, 3),
        (5, 0)
    ]
    
    l_to_nmono = {0: 1, 1: 3, 2: 6, 3: 10}
    
    def get_orbital_name(n, l, m):
        if l == 0:
            return f"({n}s)"
        elif l == 1:
            names = ["px", "py", "pz"]
            return f"({n}{names[m]})"
        elif l == 2:
            names = ["dx2", "dy2", "dz2", "dxy", "dyz", "dzx"]
            return f"({n}{names[m]})"
        elif l == 3:
            names = ["fx3", "fy3", "fz3", "fx2y", "fx2z", "fy2x", "fy2z", "fz2x", "fz2y", "fxyz"]
            return f"({n}{names[m]})"
        return ""
    
    # Flatten all basis functions across all atoms and shells
    # To construct C_eff of shape (N_total_basis, N_MOs)
    C_eff_list = []
    basis_labels = []
    
    for a in range(N_atoms):
        for s, (n, l) in enumerate(orbitals_config):
            C_shell = LCAO_coeffs[a, s, :] # (N_MOs,)
            
            W_key = f"W_shell_{s}_n{n}_l{l}"
            if W_key in ang_params:
                W_param = ang_params[W_key]
                if W_param.ndim == 4:
                    W_param = W_param[0]
                W_shell = W_param[a] # (N_MOs, N_mono)
            else:
                # If W is not found (e.g. constant 1 for l=0), we construct it
                N_mono = l_to_nmono[l]
                W_shell = jnp.ones((N_MOs, N_mono)) if l == 0 else jnp.zeros((N_MOs, N_mono))
            
            # C_tilde for this shell and atom: C_shell * W_shell
            # C_shell is (N_MOs,), W_shell is (N_MOs, N_mono)
            C_tilde = C_shell[:, None] * W_shell # (N_MOs, N_mono)
            
            # Transpose to (N_mono, N_MOs) to append to C_eff_list
            C_eff_list.append(C_tilde.T)
            
            for m in range(l_to_nmono[l]):
                orb_name = get_orbital_name(n, l, m)
                basis_labels.append(f"Atom_{a}_n{n}_l{l}_m{m} {orb_name}")
                
    C_eff = jnp.concatenate(C_eff_list, axis=0) # (N_total_basis, N_MOs)
    print(f"C_eff shape: {C_eff.shape} (N_total_basis, N_MOs)")
    
    print("\n=== 2. Computing Density Matrix (P) ===")
    # Assuming spin-restricted or just calculating total P from all MOs
    # P = C_eff @ C_eff.T
    P = C_eff @ C_eff.T # (N_total_basis, N_total_basis)
    print(f"P shape: {P.shape}")
    
    # Save to disk
    np.save("C_eff.npy", np.array(C_eff))
    np.save("DensityMatrix.npy", np.array(P))
    print("Saved C_eff.npy and DensityMatrix.npy")
    
    print("\n=== 3. Spin Contamination Check (Alpha vs Beta) ===")
    
    # Calculate separate density matrices for Alpha (MO 0) and Beta (MO 1)
    C_alpha = C_eff[:, 0:1]
    C_beta = C_eff[:, 1:2]
    P_alpha = C_alpha @ C_alpha.T
    P_beta = C_beta @ C_beta.T
    
    print(f"{'Basis Label':<25s} | {'Alpha (MO 0)':<12s} | {'Beta (MO 1)':<12s} | {'Diff':<8s}")
    print("-" * 65)
    for i, label in enumerate(basis_labels):
        occ_a = P_alpha[i, i]
        occ_b = P_beta[i, i]
        
        diff = abs(occ_a - occ_b)
        print(f"{label:<25s} | {occ_a:12.4f} | {occ_b:12.4f} | {diff:8.4f}")
            
    # Save raw weights for skeptics
    print("\n=== 4. Dumping Raw Weights ===")
    with open("raw_weights_dump.txt", "w") as f:
        f.write("=== RAW LCAO COEFFICIENTS (C) ===\n")
        f.write("Shape: (N_atoms, N_shells, N_MOs)\n")
        f.write(str(np.array(LCAO_coeffs)) + "\n\n")
        
        f.write("=== RAW ANGULAR WEIGHTS (W) ===\n")
        for k, v in ang_params.items():
            f.write(f"Key: {k}\n")
            f.write(f"Shape: {v.shape}\n")
            f.write(str(np.array(v)) + "\n\n")
    print("Saved raw_weights_dump.txt for full transparency.")

if __name__ == "__main__":
    analyze()

