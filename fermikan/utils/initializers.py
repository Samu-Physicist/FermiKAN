import jax
import jax.numpy as jnp
import haiku as hk

def get_slater_alpha(Z: int, n: int) -> float:
    if Z < 1 or Z > 10:
        raise ValueError(f"SlaterInitializer currently supports Z=1 to 10. Got Z={Z}")
    if n == 1:
        S = 0.0 if Z == 1 else 0.30
        return (Z - S) / 1.0
    elif n == 2:
        if Z < 3:
            S = 0.85 * (Z - 1)
            return max(Z - S, 0.1) / 2.0
        S = 1.70 + 0.35 * (Z - 3)
        return (Z - S) / 2.0
    else:
        return 0.1

class StochasticHomeAtomInitializer(hk.initializers.Initializer):
    def __init__(self, Z_atoms: tuple[int, ...], num_determinants: int, n_up: int, n_down: int, stddev: float = 1.0, noise: float = 1e-4):
        self.Z_atoms = Z_atoms
        self.num_determinants = num_determinants
        self.n_up = n_up
        self.n_down = n_down
        self.stddev = stddev
        self.noise = noise

    def __call__(self, shape, dtype=jnp.float32):
        # shape is (N_atoms, N_total_orbitals, N_MOs)
        N_atoms = len(self.Z_atoms)
        N_MOs = shape[2]
        
        # 1. Generate all available orbitals from neutral atoms
        all_orbitals = []
        for a, Z in enumerate(self.Z_atoms):
            for e in range(Z):
                shell = (e // 2) * 4  # 0(1s), 4(2s), 8(3s)...
                all_orbitals.append((a, shell))
                
        # 2. Sort by shell energy so we populate 1s across all atoms first, then 2s, etc.
        all_orbitals.sort(key=lambda x: x[1])
        
        if N_MOs == self.n_up + self.n_down:
            # UHF Mode: Spin-dependent assignment for True Neutral Atoms
            total_e = self.n_up + self.n_down
            if len(all_orbitals) > total_e:
                all_orbitals = all_orbitals[:total_e] # Cation
            elif len(all_orbitals) < total_e:
                all_orbitals += [all_orbitals[-1]] * (total_e - len(all_orbitals)) # Anion
                
            atoms_alpha, shells_alpha = [], []
            atoms_beta, shells_beta = [], []
            
            # Count occurrences to find cores (pairs) and open shells
            counts = {}
            for orb in all_orbitals:
                counts[orb] = counts.get(orb, 0) + 1
                
            open_shells = []
            for orb, count in counts.items():
                if count >= 2:
                    # Distribute a pair
                    atoms_alpha.append(orb[0]); shells_alpha.append(orb[1])
                    atoms_beta.append(orb[0]); shells_beta.append(orb[1])
                    if count > 2:
                        open_shells.extend([orb] * (count - 2))
                elif count == 1:
                    open_shells.append(orb)
                    
            # Distribute open shells to satisfy n_up and n_down
            for orb in open_shells:
                if len(atoms_alpha) < self.n_up:
                    atoms_alpha.append(orb[0]); shells_alpha.append(orb[1])
                elif len(atoms_beta) < self.n_down:
                    atoms_beta.append(orb[0]); shells_beta.append(orb[1])
                    
            home_atoms_flat = jnp.array((atoms_alpha + atoms_beta) * self.num_determinants)
            home_shells_flat = jnp.array((shells_alpha + shells_beta) * self.num_determinants)
            
        else:
            # RHF Mode: We just take the first N_MOs unique spatial orbitals
            unique_orbs = []
            for orb in all_orbitals:
                if orb not in unique_orbs:
                    unique_orbs.append(orb)
            while len(unique_orbs) < N_MOs:
                unique_orbs.append(unique_orbs[-1])
            
            atoms_rhf = [orb[0] for orb in unique_orbs[:N_MOs]]
            shells_rhf = [orb[1] for orb in unique_orbs[:N_MOs]]
            home_atoms_flat = jnp.array(atoms_rhf * self.num_determinants)
            home_shells_flat = jnp.array(shells_rhf * self.num_determinants)
        
        # Truncate to match N_MOs (shape[2]) to prevent Python broadcasting crashes!
        home_atoms_flat = home_atoms_flat[:shape[2]]
        home_shells_flat = home_shells_flat[:shape[2]]
        
        # Atom mask: 1.0 if it's the home atom, else self.noise
        mask_atom = jnp.where(jnp.arange(N_atoms)[:, None] == home_atoms_flat[None, :], 1.0, self.noise)
        
        # Orb mask: 1.0 if it's the target shell, else self.noise
        N_total_orbitals = shape[1]
        mask_orb = jnp.where(jnp.arange(N_total_orbitals)[:, None] == home_shells_flat[None, :], 1.0, self.noise)
        
        # Combine masks: (N_atoms, N_total_orbitals, N_MOs)
        mask = mask_atom[:, None, :] * mask_orb[None, :, :]
        
        key = hk.next_rng_key()
        # Initialize near 1.0 for the targeted 1s home atom, and near 0.0 for others
        base_vals = jnp.where(mask > 0.5, 1.0, 0.0) 
        random_noise = jax.random.normal(key, shape, dtype) * self.noise
        
        return base_vals + random_noise
