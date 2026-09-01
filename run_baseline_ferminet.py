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
    print("=== Starting BASELINE MLP LiH Experiment ===")
    
    # 1. Get default FermiNet config
    config = base_config.default()
        
    # 2. System Definition (H2 Molecule at Dissociation Limit R=5.0 Bohr)
    from ferminet.utils import system
    config.system.electrons = (1, 1)  # 1 spin-up, 1 spin-down
    config.system.molecule = [
        system.Atom(symbol='H', coords=(0., 0., -0.7)),
        system.Atom(symbol='H', coords=(0., 0., 0.7))
    ]
    
    
    # 3. Use default DeepMind MLP
    config.network.determinants = 1  
    
    # 4. Tune VMC parameters (Adam, Zero-shot)
    config.optim.optimizer = 'adam'
    config.optim.lr.rate = 5e-4
    config.optim.iterations = 2000  # Divergence usually happens very fast
    config.mcmc.burn_in = 10
    config.batch_size = 16384
    
    # Zero-shot stabilization! No PySCF pretraining!
    config.pretrain.iterations = 0
    
    import time
    print("Launching Baseline FermiNet (Zero-Shot, Adam)...")
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
