import jax
import jax.numpy as jnp
from ferminet import base_config
from ferminet.utils import system
from ferminet import networks
import sys
import haiku as hk

from ferminet_adapter import make_pdkan_network

def get_h2_config():
    config = base_config.default()
    config.system.electrons = (1, 1)
    config.system.molecule = [
        system.Atom(symbol='H', coords=(0., 0., -0.7)),
        system.Atom(symbol='H', coords=(0., 0., 0.7))
    ]
    return config

def get_lih_config():
    config = base_config.default()
    config.system.electrons = (2, 2)  # 1 spin-up, 1 spin-down
    config.system.molecule = [
        system.Atom(symbol='Li', coords=(0., 0., 0.)),
        system.Atom(symbol='H', coords=(0., 0., 3.01))
    ]
    return config

def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

def print_param_breakdown(params):
    for module_name, module_params in params.items():
        count = count_params(module_params)
        print(f"  {module_name}: {count} params")

def main():
    config = get_h2_config()
    key = jax.random.PRNGKey(42)
    
    # --- 1. Baseline FermiNet (MLP) ---
    print("=== Baseline FermiNet (MLP) ===")
    config.network.determinants = 1
    
    # We must construct atoms array and charges array
    atoms = jnp.stack([jnp.array(atom.coords) for atom in config.system.molecule])
    charges = jnp.array([atom.charge for atom in config.system.molecule])
    spins = config.system.electrons

    feature_layer = networks.make_ferminet_features(
        natoms=charges.shape[0],
        nspins=spins,
        ndim=3,
    )
    network_baseline = networks.make_fermi_net(
        spins,
        charges,
        feature_layer=feature_layer,
        determinants=config.network.determinants
    )
    
    # init expects a subkey
    # The dummy input required by FermiNet is just positions of shape (nelectrons * 3,)
    dummy_pos = jax.random.normal(key, (sum(spins) * 3,))
    
    # Some older versions of FermiNet might require pos in init, but modern Haiku transform wraps it
    try:
        # Check if the network.init wants pos
        params_baseline = network_baseline.init(key, dummy_pos)
    except TypeError:
        # Or maybe it just wants a key?
        params_baseline = network_baseline.init(key)
        
    baseline_count = count_params(params_baseline)
    print(f"Total Parameters: {baseline_count}")
    print_param_breakdown(params_baseline)
    
    # --- 2. PD-KAN FermiNet ---
    print("\n=== PD-KAN FermiNet ===")
    # pdkan network builder expects similar arguments
    network_pdkan, _ = make_pdkan_network(
        atoms,
        spins,
        charges,
        num_determinants=1
    )
    
    params_pdkan = network_pdkan.init(key, dummy_pos, spins, atoms, charges)
    pdkan_count = count_params(params_pdkan)
    print(f"Total Parameters: {pdkan_count}")
    print_param_breakdown(params_pdkan)
    
    print("\n=== Comparison ===")
    print(f"Compression Ratio: {baseline_count / pdkan_count:.2f}x smaller")

if __name__ == "__main__":
    main()
