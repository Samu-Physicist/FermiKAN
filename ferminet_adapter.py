import haiku as hk
import jax.numpy as jnp
from pdkan import FermiKAN_Network

def make_pdkan_network(atoms, spins, charges, **kwargs):
    """
    Adapter function that wraps PD-KAN to match DeepMind's FermiNet expected signature.
    DeepMind's ferminet.networks expects a function that returns a Haiku transformed network
    and any network-specific options.
    """
    
    def network_fn(r_electrons, spins, atoms, charges):
        """
        The internal function wrapped by hk.transform.
        Args:
            r_electrons: shape (batch, N_e * 3) or similar flattened structure.
            spins: tuple of (n_up, n_down)
            atoms: coordinates of atoms
            charges: atomic numbers
        """
        n_up, n_down = spins
        N_e = n_up + n_down
        
        # Initialize PD-KAN
        pdkan = FermiKAN_Network(
            Z_atoms=tuple(charges),
            num_electrons=N_e,
            num_determinants=kwargs.get("num_determinants", 1),
            n_up=n_up,
            n_down=n_down,
        )
        
        # FermiNet might pass electrons flattened: (..., static_N_e * 3)
        # We need to reshape it for PD-KAN: (..., static_N_e, 3)
        # spins could be a BatchTracer here, so we shouldn't use N_e for control flow.
        static_N_e = r_electrons.shape[-1] // 3
        r_electrons = jnp.reshape(r_electrons, (*r_electrons.shape[:-1], static_N_e, 3))
        # Call PD-KAN
        sign, log_abs = pdkan(r_electrons, atoms)
        return sign, log_abs
    
    # FermiNet expects the network to be transformed. 
    # Because we don't have explicit internal state (like batch norm running stats), 
    # we can use hk.without_apply_rng if we don't use rng in __call__. 
    # FermiNet standard requires hk.without_apply_rng.
    transformed_network = hk.without_apply_rng(hk.transform(network_fn))
    
    # Return the network and any auxiliary options dict
    options = {}
    return transformed_network, options
