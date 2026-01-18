import torch
from torch import Tensor
import math

################################ Multi-Objective Problem #########################
class VLMOP2:
    def __init__(self) -> None:
        self.dim = 6
        # Standard VLMOP2 bounds are usually [-2, 2]^d
        self.bounds = torch.tensor([[-2.] * self.dim, [2.] * self.dim])
        # You return -f(X), so ref_point should be "worse" (more negative) than typical -f values
        self.ref_point = torch.tensor([-1.2, -1.2])

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        m = 2
        X = X.reshape(-1, self.dim)
        fX = torch.empty((m, X.shape[0]), dtype=X.dtype, device=X.device)

        a = torch.tensor(1.0 / math.sqrt(self.dim), dtype=X.dtype, device=X.device)

        s1 = torch.sum(torch.square(X - a), dim=1)
        s2 = torch.sum(torch.square(X + a), dim=1)

        fX[0, :] = 1.0 - torch.exp(-s1)
        fX[1, :] = 1.0 - torch.exp(-s2)

        return -fX.T


class VLMOP3:
   def __init__(self) -> None:
      self.bounds = torch.tensor([[-3.,-3.],[3.,3.]])
      self.ref_point = torch.tensor([-4.,-25.,-0.25])
      self.dim = 2

   def __call__(self, X: torch.Tensor) -> torch.Tensor:
      m = 3  # Assuming m is 3 based on the usage in the function
      X = X.reshape(-1,2)
      fX = torch.empty((m, X.shape[0]), dtype=X.dtype, device=X.device)

      fX[0, :] = 0.5 * (torch.square(X[:, 0]) + torch.square(X[:, 1])) + torch.sin(torch.square(X[:, 0]) + torch.square(X[:, 1]))
      fX[1, :] = torch.square(3*X[:, 0] - 2*X[:, 1] + 4) / 8 + torch.square(X[:, 0] - X[:, 1] + 1) / 27 + 15
      fX[2, :] = 1 / (torch.square(X[:, 0]) + torch.square(X[:, 1]) + 1) - 1.1 * torch.exp(-torch.square(X[:, 0]) - torch.square(X[:, 1]))

      return -fX.T
   




from typing import Optional
import torch
from torch import Tensor
from botorch.test_functions.base import (
    MultiObjectiveTestProblem,
)
import numpy as np
from scipy.integrate import solve_ivp


class SnAr(MultiObjectiveTestProblem):
    r"""A two objective optimization problem for a nucleophilic aromatic substitution
    (SnAr) reaction.

    Design space `x = (tau, equiv_pldn, conc_dfnb, temperature)`:
        - `tau` is the residence time in minutes.
        - `equiv_pldn` is the equivalents of pyrrolidine.
        - `conc_dfnb` is the concentration of 2,4 dinitrofluorobenenze at
            reactor inlet (after mixing) in M.
        - `temperature` is the reactor temperature in degrees celsius.

    Objective `min(-log(sty), log(e_factor))`:
        - `sty` is the space-time yield measured in kg/m^3/h.
        - `e_factor` is the environmental factor.

    This implementation is adapted from
    https://github.com/sustainable-processes/summit/blob/master/summit/benchmarks/snar.py
    """

    dim = 4
    num_objectives = 2
    _bounds = [(0.5, 2.0), (1.0, 5.0), (0.1, 0.5), (30, 120)]
    _ref_point = [0, 120]
    #_ref_point = [-5.5, 5]
    #_ref_point = [-7.5, 3.]

    # Molecular weights (g/mol)
    molecular_weights = [159.09, 71.12, 210.21, 210.21, 261.33]
    # g/mL (should adjust to temp, but just using @ 25C)
    rho_ethanol = 0.789
    # Avogadro kJ/K/mol
    R = 8.314 / 1000
    # Absolute zero
    absolute_zero = 273.15
    # Reservoir concentration of 1 is 1 M = 1 mM
    C1_0 = 2.0
    # Reservoir concentration of  2 is 2 M = 2 mM
    C2_0 = 4.2
    # volume
    V = 5

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        r"""Constructor for SnAr.
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the objectives.
        """
        super().__init__(noise_std=noise_std, negate=negate)

    @classmethod
    def solve_ode(cls, X) -> Tensor:
        """
        Args:
            X: A `batch_shape x d`-dim Tensor containing the designs.

        Returns:
            A `batch_shape x M`-dim Tensor containing the objectives.

        """
        # We consider transforming X into `B x d`-dim Tensor and then performing `B`
        # function evaluations. Certain computations are executed in parallel when
        # simple.
        batch_shape = X.shape[:-1]
        B = torch.prod(torch.tensor(batch_shape))
        new_shape = torch.Size([B]) + torch.Size([cls.dim])
        X_reshaped = X.reshape(new_shape)

        # extract design variables
        tau = X_reshaped[:, 0]
        equiv_pldn = X_reshaped[:, 1]
        conc_dfnb = X_reshaped[:, 2]
        temperature = X_reshaped[:, 3]

        # concentrations
        C_i = torch.zeros(B, 5)
        C_i[:, 0] = conc_dfnb
        C_i[:, 1] = equiv_pldn * conc_dfnb

        # Flow rates
        q_tot = cls.V / tau
        # Flow rate of 1 (dfnb)
        q_1 = C_i[:, 0] / cls.C1_0 * q_tot
        # Flow rate of 2 (pldn)
        q_2 = C_i[:, 1] / cls.C2_0 * q_tot
        # Flow rate of ethanol [This quantity is not used]
        q_ethanol = q_tot - q_1 - q_2

        # Integration step
        # Ideally we should integrate in parallel, but scipy.stats.solve_ivp
        # does not have this feature.
        C_final = np.zeros(shape=(B, 5))
        for b in range(B):
            def _integrand_b(t, concentration, temperature):
                C = concentration
                T = temperature + cls.absolute_zero
                T_ref = 90 + cls.absolute_zero
                # Need to convert from 10^-2 M^-1s^-1 to M^-1min^-1
                k = (
                    lambda k_ref, E_a, temp:
                    0.6 * k_ref * torch.exp(-E_a / cls.R * (1 / temp - 1 / T_ref))
                )

                k_a = k(57.9, 33.3, T)
                k_b = k(2.70, 35.3, T)
                k_c = k(0.865, 38.9, T)
                k_d = k(1.63, 44.8, T)

                # Reaction Rates
                r = torch.zeros(5)
                # Set to reactants when close
                for i in [0, 1]:
                    C[i] = 0 if C[i] < 1e-6 * C_i[b][i] else C[i]

                r[0] = -(k_a + k_b) * C[0] * C[1]
                r[1] = -(k_a + k_b) * C[0] * C[1] \
                       - k_c * C[1] * C[2] \
                       - k_d * C[1] * C[3]
                r[2] = k_a * C[0] * C[1] - k_c * C[1] * C[2]
                r[3] = k_a * C[0] * C[1] - k_d * C[1] * C[3]
                r[4] = k_c * C[1] * C[2] + k_d * C[1] * C[3]

                return r

            res_b = solve_ivp(
                _integrand_b, [0, tau[b]], C_i[b], args=(temperature[b],)
            )
            C_final[b, :] = res_b.y[:, -1]

        # Convert numpy array to tensor
        C_final = torch.tensor(C_final, dtype=torch.double)

        # Calculate STY and E-factor
        # Convert to kg m^-3 h^-1
        sty = 6e4 / 1000 * cls.molecular_weights[2] * C_final[:, 2] * q_tot / cls.V
        sty = sty.clamp_min(1e-6)

        term_2 = 1e-3 * sum(
            [cls.molecular_weights[i] * C_final[:, i] * q_tot
             for i in range(5) if i != 2]
        )
        # Set to a large value if no product formed
        mask = torch.isclose(C_final[:, 2], torch.zeros(B, dtype=torch.double))

        e_factor = torch.where(
            mask,
            1e3 * torch.ones(B, dtype=torch.double),
            (q_tot * cls.rho_ethanol + term_2) /
            (1e-3 * cls.molecular_weights[2] * C_final[:, 2] * q_tot)
        )
        obj_values = torch.column_stack([-torch.log(sty), torch.log(e_factor)])

        return obj_values.reshape(batch_shape + torch.Size([cls.num_objectives]))

    def evaluate_true(self, X: Tensor) -> Tensor:
        return self.solve_ode(X)
    







class Four_bar_truss_design(MultiObjectiveTestProblem):
    """
    RE21 multi-objective optimization problem (Truss Design Problem).
    
    A structural optimization problem for a 10-bar truss design with 2 objectives:
    - f1: Weight (to be minimized)
    - f2: Displacement (to be minimized)
    
    Domain: [0, 1]^4 (normalized), which maps to physical bounds
    """
    
    dim = 4
    num_objectives = 2
    _bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]  # Normalized bounds
    _ref_point = torch.tensor([-2886.3695604236013, -0.039999999999998245])

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False):
        super().__init__(noise_std=noise_std, negate=negate)
        
        # Problem parameters
        self.F = 10.0
        self.sigma = 10.0
        self.E = 2.0e5
        self.L = 200.0
        
        tmp_val = self.F / self.sigma
        
        # Physical bounds (will be used for denormalization)
        self.lbound = torch.tensor([
            tmp_val, 
            np.sqrt(2.0) * tmp_val, 
            np.sqrt(2.0) * tmp_val, 
            tmp_val
        ]).float()
        self.ubound = torch.ones(4).float() * 3 * tmp_val
        
        # Nadir point (reference point for hypervolume)
        # self.ref_point = [2886.3695604236013, 0.039999999999998245]
    
    def evaluate_true(self, X: Tensor) -> Tensor:
        """
        Evaluate the RE21 problem.
        
        Args:
            X: A `batch_shape x d`-dim tensor of normalized inputs in [0,1]^4.
            
        Returns:
            A `batch_shape x num_objectives`-dim tensor of function values.
        """
        # Move bounds to same device as input
        lbound = self.lbound.to(X.device)
        ubound = self.ubound.to(X.device)
        
        # Denormalize from [0,1] to physical bounds
        x = X * (ubound - lbound) + lbound
        
        # Extract design variables
        x1 = x[..., 0]
        x2 = x[..., 1] 
        x3 = x[..., 2]
        x4 = x[..., 3]
        
        # Objective 1: Weight
        f1 = self.L * ((2 * x1) + np.sqrt(2.0) * x2 + torch.sqrt(x3) + x4)
        
        # Objective 2: Displacement
        f2 = ((self.F * self.L) / self.E) * (
            (2.0 / x1) + 
            (2.0 * np.sqrt(2.0) / x2) - 
            (2.0 * np.sqrt(2.0) / x3) + 
            (2.0 / x4)
        )
        
        return torch.stack([f1, f2], dim=-1)
    
    # @property
    # def ref_point(self) -> list:
    #     """Reference point for hypervolume calculation."""
    #     return self._ref_point
    
    def get_physical_bounds(self) -> tuple:
        """
        Get the physical (denormalized) bounds for the problem.
        
        Returns:
            Tuple of (lower_bounds, upper_bounds) in physical units
        """
        return self.lbound.tolist(), self.ubound.tolist()







