import sys
try:
    from ferminet import train
    from ferminet import base_config
except ImportError:
    print("Error: Official DeepMind FermiNet is not installed.")
    sys.exit(1)

from absl import app
from absl import logging

def main(argv):
    logging.set_verbosity(logging.INFO)
    print("=== Starting PD-KAN H2 Experiment ===")
    
    # 1. Get default FermiNet config
    config = base_config.default()
    
    # 2. System Definition (H2 Molecule)
    from ferminet.utils import system
    config.system.electrons = (1, 1)  # 1 spin-up, 1 spin-down
    config.system.molecule = [
        system.Atom(symbol='H', coords=(0., 0., -0.7)),
        system.Atom(symbol='H', coords=(0., 0., 0.7))
    ]


    # 3. Network Builder Injection (PD-KAN)
    from ferminet import networks
    import collections
    import jax
    import jax.numpy as jnp
    
    def pdkan_builder(nspins, charges, **kwargs):
        sys.path.append('./archive')
        from ferminet_adapter import make_pdkan_network
        atoms = jnp.stack([jnp.array(atom.coords) for atom in config.system.molecule])
        charges_tuple = tuple([int(atom.charge) for atom in config.system.molecule])
        network_pdkan, options = make_pdkan_network(atoms, nspins, charges_tuple, num_determinants=1)
        
        def init_fn(key, x=None):
            if x is None:
                x = jnp.zeros(sum(nspins)*3)
            return network_pdkan.init(key, x, nspins, atoms, charges_tuple)
            
        def apply_fn(params, x, *args, **kwargs):
            return network_pdkan.apply(params, x, nspins, atoms, charges_tuple)
            
        Network = collections.namedtuple('Network', ['options', 'init', 'apply', 'orbitals'])
        return Network(options=options, init=init_fn, apply=apply_fn, orbitals=None)
    
    networks.make_fermi_net = pdkan_builder
    config.network.determinants = 1
    
    # 4. Tune VMC parameters (Adam, Zero-shot)
    # config.optim.optimizer = 'lamb'
    config.optim.optimizer = 'adam'
    config.optim.lr.rate = 1.0e-3  # LAMB allows slightly larger LR
    # config.optim.lr.decay = 0.5  # Mild decay to help settle at the bottom
    # config.optim.lr.delay = 700
    config.optim.iterations = 10000
    config.mcmc.burn_in = 10
    config.batch_size = 16384
    
    # Zero-shot stabilization! No PySCF pretraining!
    config.pretrain.iterations = 0
    
    # Save Checkpoints (frequency in minutes)
    config.log.save_path = "exp_fermiKAN_H2"
    config.log.save_frequency = 2.0

    
    import time
    print("Launching PD-KAN (Zero-Shot, Adam)...")
    start_time = time.time()
    try:
        train.train(config)
    except Exception as e:
        logging.exception("FermiNet training crashed!")
        raise
    finally:
        end_time = time.time()
        print(f"=== Total Wall-Time: {end_time - start_time:.2f} seconds ===")

if __name__ == "__main__":
    app.run(main)
