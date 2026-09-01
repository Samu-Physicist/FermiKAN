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
    def __init__(self, Z_atoms: tuple[int, ...], num_determinants: int, stddev: float = 1.0, noise: float = 1e-4):
        self.Z_atoms = Z_atoms
        self.num_determinants = num_determinants
        self.stddev = stddev
        self.noise = noise

    def __call__(self, shape, dtype=jnp.float32):
        # shape is expected to be (N_atoms, N_total_orbitals, num_determinants * num_electrons)
        N_atoms = len(self.Z_atoms)
        home_atoms = jnp.repeat(jnp.arange(N_atoms), jnp.array(self.Z_atoms, dtype=jnp.int32))
        home_atoms_flat = jnp.tile(home_atoms, self.num_determinants)
        
        # Atom mask: 1.0 if it's the home atom, else self.noise
        mask_atom = jnp.where(jnp.arange(N_atoms)[:, None] == home_atoms_flat[None, :], 1.0, self.noise)
        
        # We only want to populate the 1s orbital (index 0) initially!
        # If we mix in p and d orbitals, the initial energy explodes.
        N_total_orbitals = shape[1]
        mask_orb = jnp.where(jnp.arange(N_total_orbitals)[:, None] == 0, 1.0, self.noise)
        
        # Combine masks: (N_atoms, N_total_orbitals, num_electrons)
        mask = mask_atom[:, None, :] * mask_orb[None, :, :]
        
        key = hk.next_rng_key()
        # Initialize near 1.0 for the targeted 1s home atom, and near 0.0 for others
        base_vals = jnp.where(mask > 0.5, 1.0, 0.0) 
        random_noise = jax.random.normal(key, shape, dtype) * self.noise
        
        return base_vals + random_noise

def get_monomials(q: jnp.ndarray, max_l: int) -> dict[int, jnp.ndarray]:
    x, y, z = q[..., 0], q[..., 1], q[..., 2]
    res = {0: jnp.ones_like(x)[..., None]}
    if max_l >= 1:
        res[1] = jnp.stack([x, y, z], axis=-1)
    if max_l >= 2:
        res[2] = jnp.stack([x**2, y**2, z**2, x*y, y*z, z*x], axis=-1)
    if max_l >= 3:
        res[3] = jnp.stack([x**3, y**3, z**3, x**2*y, x**2*z, y**2*x, y**2*z, z**2*x, z**2*y, x*y*z], axis=-1)
    return res

class Vectorized_RadialKAN(hk.Module):
    def __init__(self, degree: int, Z_atoms: tuple[int, ...], n_values: list[int], l_values: list[int], name: str | None = None):
        super().__init__(name=name)
        self.degree = degree
        self.Z_atoms = Z_atoms
        self.n_values = n_values
        self.l_values = l_values

    def __call__(self, r_shifted: jnp.ndarray, r_true: jnp.ndarray, h_dynamic: jnp.ndarray = None) -> jnp.ndarray:
        N_atoms = len(self.Z_atoms)
        N_shells = len(self.n_values)
        
        # Init xi exactly from Slater's rules
        init_xi = jnp.zeros((N_atoms, N_shells))
        for a, Z in enumerate(self.Z_atoms):
            for s, n in enumerate(self.n_values):
                init_xi = init_xi.at[a, s].set(get_slater_alpha(Z, n))
                
        raw_xi_init = jnp.log(jnp.exp(init_xi) - 1.0)
        raw_xi = hk.get_parameter("raw_xi", shape=(N_atoms, N_shells), init=hk.initializers.Constant(raw_xi_init))
        
        # Static Orbital Morphing (r_true)
        # Chebyshev up to degree 10 on r_true
        r_true_exp = jnp.expand_dims(r_true, -1) # (..., N_atoms, 1)
        x_cheb_static = jnp.tanh(r_true_exp**2) * 2.0 - 1.0
        T_static = [jnp.ones_like(x_cheb_static), x_cheb_static]
        for k in range(2, 10):
            T_static.append(2.0 * x_cheb_static * T_static[-1] - T_static[-2])
        kan_features_static = jnp.concatenate(T_static, axis=-1) # (..., N_atoms, 10)
        
        delta_xi_static = hk.Linear(N_shells, with_bias=False, w_init=hk.initializers.Constant(0.0), name="kan_xi_static")(kan_features_static)
        
        # Dynamic Orbital Breathing (h_dynamic)
        if h_dynamic is not None:
            # h_dynamic is (..., N_e, 16). We sum over N_e to get global environment (..., 16)
            h_env = jnp.sum(h_dynamic, axis=-2) # (..., 16)
            # We want to map this to (N_atoms, N_shells)
            delta_xi_dynamic_flat = hk.Linear(N_atoms * N_shells, with_bias=False, w_init=hk.initializers.Constant(0.0), name="kan_xi_dynamic")(h_env)
            delta_xi_dynamic = jnp.reshape(delta_xi_dynamic_flat, h_env.shape[:-1] + (N_atoms, N_shells))
        else:
            delta_xi_dynamic = 0.0
            
        raw_xi_total = raw_xi + delta_xi_dynamic + delta_xi_static
        xi = jax.nn.softplus(raw_xi_total)
        

        # Calculate Exact LCAO Baseline (Generalized Laguerre Polynomials)
        R_laguerre_list = []
        for a, Z in enumerate(self.Z_atoms):
            shell_vals = []
            for s, (n, l) in enumerate(zip(self.n_values, self.l_values)):
                x_lag = 2.0 * Z * r_shifted[..., a] / n
                k = n - l - 1
                alpha_lag = 2 * l + 1
                if k == 0:
                    val = jnp.ones_like(x_lag)
                elif k == 1:
                    val = (1.0 + alpha_lag) - x_lag
                elif k == 2:
                    val = 0.5 * x_lag**2 - (alpha_lag + 2.0) * x_lag + 0.5 * (alpha_lag + 1.0) * (alpha_lag + 2.0)
                elif k == 3:
                    val = -x_lag**3 / 6.0 + (alpha_lag + 3.0) / 2.0 * x_lag**2 - (alpha_lag + 2.0) * (alpha_lag + 3.0) / 2.0 * x_lag + (alpha_lag + 1.0) * (alpha_lag + 2.0) * (alpha_lag + 3.0) / 6.0
                elif k == 4:
                    val = x_lag**4 / 24.0 - (alpha_lag + 4.0) / 6.0 * x_lag**3 + (alpha_lag + 3.0) * (alpha_lag + 4.0) / 4.0 * x_lag**2 - (alpha_lag + 2.0) * (alpha_lag + 3.0) * (alpha_lag + 4.0) / 6.0 * x_lag + (alpha_lag + 1.0) * (alpha_lag + 2.0) * (alpha_lag + 3.0) * (alpha_lag + 4.0) / 24.0
                else:
                    val = jnp.ones_like(x_lag)
                shell_vals.append(val)
            R_laguerre_list.append(jnp.stack(shell_vals, axis=-1))
            
        R_laguerre = jnp.stack(R_laguerre_list, axis=-2) # (..., N_atoms, N_shells)
        

        # --- Exact e-n Cusp Protector ---
        # Z_I is the true nuclear charge
        Z_I = jnp.array(self.Z_atoms) # (N_atoms,)
        
        # Transition parameter beta (learnable)
        raw_beta = hk.get_parameter("raw_beta", shape=(N_atoms,), init=hk.initializers.Constant(0.0))
        beta = jax.nn.softplus(raw_beta) + 0.1 # Strictly positive

        
        # We need to correctly broadcast xi, Z_I, beta against r_true
        # r_true shape: (..., N_atoms)
        r_true_exp = jnp.expand_dims(r_true, -1) # (..., N_atoms, 1)
        Z_I_exp = jnp.expand_dims(Z_I, -1) # (N_atoms, 1)
        beta_exp = jnp.expand_dims(beta, -1) # (N_atoms, 1)
        n_expanded = jnp.array(self.n_values).reshape(1, -1) # (1, N_shells)
        
        # [Elegant Hydrogenic Cusp Distribution]
        # The Laguerre polynomial L_{n-1}^{(1)}(2 Z_I r / n) naturally provides a slope of -Z_I * (n-1)/n.
        # To make the TOTAL slope exactly -Z_I for EVERY shell, the Envelope MUST provide exactly -Z_I / n.
        Z_I_target = Z_I_exp / n_expanded
        
        envelope_arg = -xi * r_true_exp - (Z_I_target - xi) / beta_exp * (1 - jnp.exp(-beta_exp * r_true_exp))
        envelope = jnp.exp(envelope_arg) # (..., N_atoms, N_shells)
        
        # Combine Laguerre basis with dynamic envelope
        return R_laguerre * envelope

class Vectorized_ShellAngularKAN(hk.Module):
    def __init__(self, basis_pool: list[tuple[int, int]], N_MOs: int, init_epsilon: float = 1.0, name: str | None = None):
        super().__init__(name=name)
        self.basis_pool = basis_pool
        self.N_MOs = N_MOs
        self.max_l = max([l for n, l in basis_pool])
        self.init_epsilon = init_epsilon

    def __call__(self, r_vec: jnp.ndarray) -> jnp.ndarray:
        N_atoms = r_vec.shape[-2]
        
        # Atom-specific core polarizability radius (epsilon)
        raw_epsilon = hk.get_parameter("raw_epsilon", shape=(N_atoms,), init=hk.initializers.Constant(jnp.log(jnp.exp(self.init_epsilon) - 1.0)))
        epsilon = jax.nn.softplus(raw_epsilon)
        
        eps_expanded = jnp.broadcast_to(epsilon, r_vec.shape[:-1])
        eps_vec = jnp.expand_dims(eps_expanded, -1)
        
        q_unnorm = jnp.concatenate([r_vec, eps_vec], axis=-1)
        norm = jnp.linalg.norm(q_unnorm, axis=-1, keepdims=True)
        q = q_unnorm / norm
        q_xyz = q[..., :3]
        
        monos = get_monomials(q_xyz, self.max_l)
        
        shell_angular_vals = []
        for s, (n, l) in enumerate(self.basis_pool):
            m_q = monos[l] # (..., N_atoms, N_mono)
            N_mono = m_q.shape[-1]
            
            # Learnable polynomial weights W: (N_atoms, N_MOs, N_mono)
            if l == 0:
                init_val = hk.initializers.Constant(1.0)
            else:
                init_val = hk.initializers.Constant(0.0)
                
            W = hk.get_parameter(f"W_shell_{s}_n{n}_l{l}", shape=(N_atoms, self.N_MOs, N_mono), init=init_val)
            
            # m_q is (..., N_atoms, N_mono), W is (N_atoms, N_MOs, N_mono)
            # Contract over N_mono to get (..., N_atoms, N_MOs)
            val = jnp.einsum('...am, aom -> ...ao', m_q, W)
            shell_angular_vals.append(val)
            
        # Stack over shells: (..., N_atoms, N_shells, N_MOs)
        A_vals = jnp.stack(shell_angular_vals, axis=-2)
        return A_vals

class FermiKAN_Orbitals(hk.Module):
    def __init__(self, orbitals_config: list[tuple[int, int]], Z_atoms: tuple[int, ...], num_electrons: int = 2, num_determinants: int = 1, name: str | None = None):
        super().__init__(name=name)
        self.orbitals_config = orbitals_config
        self.Z_atoms = Z_atoms
        self.num_electrons = num_electrons
        self.num_determinants = num_determinants
        
        self.n_values = [n for n, l in orbitals_config]
        self.l_values = [l for n, l in orbitals_config]
        self.N_MOs = self.num_determinants * self.num_electrons

    def __call__(self, r_vecs_true: jnp.ndarray, eta: jnp.ndarray, h_dynamic: jnp.ndarray) -> jnp.ndarray:
        N_atoms = r_vecs_true.shape[-2]
        
        eta_expanded = jnp.expand_dims(eta, axis=-2) # (..., N_e, 1, 3)
        
        r_vecs_shifted_rad = r_vecs_true + eta_expanded
        r_shifted_rad = jnp.linalg.norm(r_vecs_shifted_rad, axis=-1)
        r_true = jnp.linalg.norm(r_vecs_true, axis=-1)
        
        # 1. Vectorized Radial KAN -> Shape: (..., N_atoms, N_shells)
        radial_fn = Vectorized_RadialKAN(degree=10, Z_atoms=self.Z_atoms, n_values=self.n_values, l_values=self.l_values, name="vec_radial")
        R_val = radial_fn(r_shifted_rad, r_true, h_dynamic)
        
        # 2. Vectorized Shell Angular KAN -> Shape: (..., N_atoms, N_shells, N_MOs)
        angular_fn = Vectorized_ShellAngularKAN(basis_pool=self.orbitals_config, N_MOs=self.N_MOs, name="vec_angular")
        A_val = angular_fn(r_vecs_shifted_rad)
        
        # 3. Combine Radial and Angular with r^l scaling
        r_true_L_list = [r_true**l for n, l in self.orbitals_config]
        r_true_L = jnp.stack(r_true_L_list, axis=-1) # (..., N_atoms, N_shells)
        
        R_scaled = R_val * r_true_L # (..., N_atoms, N_shells)
        R_expanded = jnp.expand_dims(R_scaled, -1) # (..., N_atoms, N_shells, 1)
        
        phi_basis = R_expanded * A_val # (..., N_atoms, N_shells, N_MOs)
        
        # 4. Multi-Orbital LCAO Mixing (C_{k, I, s})
        initializer = StochasticHomeAtomInitializer(self.Z_atoms, self.num_determinants, stddev=1.0, noise=1e-4)
        C = hk.get_parameter("LCAO_coeffs", 
                             shape=(N_atoms, len(self.orbitals_config), self.N_MOs), 
                             init=initializer)
        
        phi_weighted = phi_basis * C # (..., N_atoms, N_shells, N_MOs)
        
        # Sum over atoms and shells -> (..., N_e, N_MOs)
        phi_molecular = jnp.sum(phi_weighted, axis=(-3, -2))
        
        # Reshape to (..., N_determinants, N_e, N_e)
        phi_reshaped = jnp.reshape(phi_molecular, phi_molecular.shape[:-1] + (self.num_determinants, self.num_electrons))
        phi = jnp.swapaxes(phi_reshaped, -3, -2)
        
        return phi

class PDKAN_Backflow(hk.Module):
    def __init__(self, degree: int = 10, init_alpha: float = 1.0, name: str | None = None):
        super().__init__(name=name)
        self.degree = degree
        self.init_alpha = init_alpha

    def __call__(self, r_electrons: jnp.ndarray) -> jnp.ndarray:
        N_e = r_electrons.shape[-2]
        
        r_i = jnp.expand_dims(r_electrons, axis=-2)
        r_j = jnp.expand_dims(r_electrons, axis=-3)
        r_ij_vec = r_i - r_j
        
        r_ij_sq = jnp.sum(r_ij_vec**2, axis=-1, keepdims=True)
        
        # Chebyshev Polynomials (bounded [-1, 1])
        x_cheb = jnp.tanh(r_ij_sq) * 2.0 - 1.0
        T_k = []
        T_k.append(jnp.ones_like(x_cheb))
        if self.degree > 1:
            T_k.append(x_cheb)
        for k in range(2, self.degree):
            T_k.append(2.0 * x_cheb * T_k[-1] - T_k[-2])
            
        kan_features = jnp.concatenate(T_k, axis=-1)
        kan_output = hk.Linear(1, with_bias=False, w_init=hk.initializers.Constant(0.0), name="kan_coeffs")(kan_features)
        
        # Alpha-KAN (Single dynamic decay, no spin separation)
        raw_alpha = hk.Linear(1, with_bias=True, 
                              w_init=hk.initializers.Constant(0.0), 
                              b_init=hk.initializers.Constant(jnp.log(jnp.exp(self.init_alpha) - 1.0)), 
                              name="kan_coeffs_alpha")(kan_features)
        
        alpha = jax.nn.softplus(raw_alpha) # (..., N_e, N_e, 1)
        
        envelope = jnp.exp(-alpha * r_ij_sq)
        
        shift_mag = kan_output * envelope # (..., N_e, N_e, 1)
        shift_vec = shift_mag * r_ij_vec  # (..., N_e, N_e, 3)
        
        # We sum over j to get the total shift for electron i
        eta = jnp.sum(shift_vec, axis=-2) # (..., N_e, 3)
        
        # Dynamic Environment Latent Vector
        # Extract a 16-dimensional latent vector representing the local N-body environment
        # for dynamic orbital breathing.
        h_dynamic_ij = hk.Linear(16, with_bias=False, w_init=hk.initializers.Constant(0.0), name="kan_dynamic_xi")(kan_features)
        h_dynamic = jnp.sum(h_dynamic_ij * envelope, axis=-2) # (..., N_e, 16)
        
        return eta, h_dynamic
        
class PDKAN_Jastrow(hk.Module):
    def __init__(self, n_up: int, n_down: int, init_gamma: float = 1.0, name: str | None = None):
        super().__init__(name=name)
        self.n_up = n_up
        self.n_down = n_down
        self.init_gamma = init_gamma
        
    def __call__(self, r_electrons: jnp.ndarray, r_vecs_true: jnp.ndarray) -> jnp.ndarray:
        N_e = r_electrons.shape[-2]
        r_i = jnp.expand_dims(r_electrons, axis=-2)
        r_j = jnp.expand_dims(r_electrons, axis=-3)
        r_ij_sq = jnp.sum((r_i - r_j)**2, axis=-1)
        
        mask = 1.0 - jnp.eye(N_e)
        dist_ij = jnp.where(mask > 0.5, jnp.sqrt(jnp.where(mask > 0.5, r_ij_sq, 1.0)), 0.0)
        
        c_matrix = jnp.zeros((N_e, N_e))
        if self.n_up > 0:
            c_matrix = c_matrix.at[:self.n_up, :self.n_up].set(1.0 / 4.0)
        if self.n_down > 0:
            c_matrix = c_matrix.at[self.n_up:, self.n_up:].set(1.0 / 4.0)
        if self.n_up > 0 and self.n_down > 0:
            c_matrix = c_matrix.at[:self.n_up, self.n_up:].set(1.0 / 2.0)
            c_matrix = c_matrix.at[self.n_up:, :self.n_up].set(1.0 / 2.0)
            
        c_matrix = c_matrix * (1.0 - jnp.eye(N_e))
        
        # e-e Correlation: Uniform learnable gamma for parallel and anti-parallel spins.
        # This guarantees the e-e cusp condition remains perfectly stable.
        
        raw_gamma_p = hk.get_parameter("raw_gamma_p", shape=(), init=hk.initializers.Constant(jnp.log(jnp.exp(self.init_gamma) - 1.0)))
        raw_gamma_a = hk.get_parameter("raw_gamma_a", shape=(), init=hk.initializers.Constant(jnp.log(jnp.exp(self.init_gamma) - 1.0)))
        
        gamma_p = jax.nn.softplus(raw_gamma_p) # Scalar
        gamma_a = jax.nn.softplus(raw_gamma_a) # Scalar
        
        gamma_matrix = jnp.zeros_like(c_matrix)
        if self.n_up > 0:
            gamma_matrix = gamma_matrix.at[:self.n_up, :self.n_up].set(gamma_p)
        if self.n_down > 0:
            gamma_matrix = gamma_matrix.at[self.n_up:, self.n_up:].set(gamma_p)
        if self.n_up > 0 and self.n_down > 0:
            gamma_matrix = gamma_matrix.at[:self.n_up, self.n_up:].set(gamma_a)
            gamma_matrix = gamma_matrix.at[self.n_up:, :self.n_up].set(gamma_a)
        
        J_matrix = c_matrix * dist_ij / (1.0 + gamma_matrix * dist_ij)
        J_val = jnp.sum(J_matrix, axis=(-2, -1)) / 2.0
        
        return J_val

class FermiKAN_Network(hk.Module):
    def __init__(self, Z_atoms: tuple[int, ...], num_electrons: int, num_determinants: int = 1, n_up: int = None, n_down: int = None, name: str | None = None):
        super().__init__(name=name)
        self.num_electrons = num_electrons
        self.num_determinants = num_determinants
        self.n_up = n_up if n_up is not None else num_electrons // 2
        self.n_down = n_down if n_down is not None else num_electrons - self.n_up
        self.backflow = PDKAN_Backflow()
        self.jastrow = PDKAN_Jastrow(self.n_up, self.n_down)
        
        self.orbitals_config = [
            (1, 0), (1, 1), (1, 2), (1, 3), # 1s, 1p, 1d (Virtual tight polarization)
            (2, 0), (2, 1), (2, 2), (2, 3), # 2s, 2p, 2d (Virtual 1-node polarization)
            (3, 0), (3, 1), (3, 2), (3, 3), # 3s, 3p, 3d, 3f
            (4, 0), (4, 1), (4, 2), (4, 3),
            (5, 0)
        ]
        
        self.orbitals = FermiKAN_Orbitals(
            orbitals_config=self.orbitals_config,
            Z_atoms=Z_atoms,
            num_electrons=num_electrons,
            num_determinants=num_determinants
        )

    def __call__(self, r_electrons: jnp.ndarray, R_atoms: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        R_atoms_expanded = jnp.expand_dims(R_atoms, axis=0) if r_electrons.ndim == 2 else R_atoms
        r_electrons_expanded = jnp.expand_dims(r_electrons, axis=-2)
        r_vecs_true = r_electrons_expanded - R_atoms_expanded
        
        eta, h_dynamic = self.backflow(r_electrons)
        phi_matrix = self.orbitals(r_vecs_true, eta, h_dynamic)
        
        phi_up = phi_matrix[..., :self.n_up, :self.n_up]
        sign_up, log_abs_up = jnp.linalg.slogdet(phi_up)
        
        if self.n_down > 0:
            phi_down = phi_matrix[..., self.n_up:, self.n_up:]
            sign_down, log_abs_down = jnp.linalg.slogdet(phi_down)
        else:
            sign_down = jnp.ones_like(sign_up)
            log_abs_down = jnp.zeros_like(log_abs_up)
            
        log_det_k = log_abs_up + log_abs_down
        sign_k = sign_up * sign_down
        
        w_det = hk.get_parameter("w_det", shape=(self.num_determinants,), init=hk.initializers.Constant(1.0))
        max_log = jnp.max(log_det_k, axis=-1, keepdims=True)
        shifted_exp = jnp.exp(log_det_k - max_log)
        linear_sum = jnp.sum(w_det * sign_k * shifted_exp, axis=-1)
        
        total_sign = jnp.sign(linear_sum)
        total_log_abs = jnp.squeeze(max_log, axis=-1) + jnp.log(jnp.abs(linear_sum))
        
        J_val = self.jastrow(r_electrons, r_vecs_true)
        
        return total_sign, total_log_abs + J_val
