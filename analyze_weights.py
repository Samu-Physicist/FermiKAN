import jax
import jax.numpy as jnp
from ferminet import checkpoint
import os
import sys

def analyze():
    output_file = "weights_analysis_output.txt"
    sys.stdout = open(output_file, 'w')
    
    ckpt_dir = "exp_fermiKAN_H2"
    if not os.path.exists(ckpt_dir):
        print(f"Error: Checkpoint directory {ckpt_dir} not found.")
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
        if len(ckpt_data) == 6:
            step, data, params, opt_state, mcmc_width, density_state = ckpt_data
        else:
            print(f"Unexpected tuple length: {len(ckpt_data)}")
            sys.exit(1)
        print(f"Restored step: {step}")
    else:
        params = ckpt_data.get("params", ckpt_data)
    
    # print("\n--- Params Keys ---")
    # for k in params.keys():
    #     print(k)
        
    orbital_key = None
    for k in params.keys():
        if "fermi_kan__orbitals" in k:
            orbital_key = k
            break
            
    if not orbital_key:
        print("Could not find fermi_kan__orbitals in params.")
        sys.exit(1)
        
    orb_params = params[orbital_key]
    
    print("\n=== 1. LCAO Coefficients (C_I,n,l) ===")
    LCAO_coeffs = orb_params.get("LCAO_coeffs", None)
    if LCAO_coeffs is not None:
        print("LCAO_coeffs shape:", LCAO_coeffs.shape)
        print("Atom 0 (H at z=-0.7):")
        print(LCAO_coeffs[0])
        print("Atom 1 (H at z=+0.7):")
        print(LCAO_coeffs[1])
        
    print("\n=== 2. Static and Dynamic Breathing (xi) ===")
    for module_name, module_params in params.items():
        if "raw_xi" in module_params:
            raw_xi = module_params["raw_xi"]
            xi_static = jax.nn.softplus(raw_xi)
            print(f"raw_xi (softplus'd) in {module_name}:")
            print(xi_static)
        
    kan_xi_static_key = [k for k in params.keys() if "kan_xi_static" in k]
    kan_xi_dyn_key = [k for k in params.keys() if "kan_xi_dynamic" in k]
    
    if kan_xi_static_key:
        w_stat = params[kan_xi_static_key[0]]['w']
        print(f"kan_xi_static weight norm (L1): {jnp.mean(jnp.abs(w_stat)):.4f}")
    if kan_xi_dyn_key:
        w_dyn = params[kan_xi_dyn_key[0]]['w']
        print(f"kan_xi_dynamic weight norm (L1): {jnp.mean(jnp.abs(w_dyn)):.4f}")
        
    print("\n=== 3. Virtual Shell Polynomials (W_shell) ===")
    for module_name, module_params in params.items():
        for k, v in module_params.items():
            if "W_shell_" in k or "w" in k and "vec_angular" in module_name:
                print(f"{module_name} / {k}: shape={v.shape}, values={v}")
            
if __name__ == "__main__":
    analyze()
