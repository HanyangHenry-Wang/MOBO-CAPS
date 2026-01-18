import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import normalize
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.constraints.constraints import Interval
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

import numpy as np
from scipy.optimize import minimize

import sys
import itertools
import gc
# import heapq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double



def get_random_points(bounds, num, seed=0):
    torch.manual_seed(seed)  # Set seed for reproducibility
    lower, upper = bounds[0], bounds[1]
    
    res = lower + (upper - lower) * torch.rand(num, bounds.shape[1])
    return res



def generate_initial_data(problem,n,seed,NOISE_SE):
    # generate training data
    torch.manual_seed(seed)

    train_x = draw_sobol_samples(bounds=problem.bounds, n=n, q=1).squeeze(1)
    train_obj_true = problem(train_x)
    train_obj = train_obj_true + torch.randn_like(train_obj_true) * NOISE_SE
    return train_x, train_obj, train_obj_true


def initialize_model(train_x, train_obj,problem):
    # define models for objective and constraint
    train_x = normalize(train_x, problem.bounds)
    models = []
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i : i + 1]
       
        model_temp =  SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
        # model_temp.likelihood.noise_covar.register_constraint("raw_noise", Interval(1e-8,1e-7)) 


        models.append(model_temp)

    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model




def select_best_m_pareto_solutions(obj_tensor, M, ref_point):

    # Get the Pareto front
    pareto_mask = is_non_dominated(obj_tensor)
    pareto_front = obj_tensor[pareto_mask]

    if pareto_front.shape[0] <= M:
        return pareto_front  # If there are fewer points than M, return all

    # Initialize hypervolume computer
    hv_computer = Hypervolume(ref_point)

    # Generate all subsets of size M
    best_subset = None
    best_hv = -float("inf")


    c = 0
    for subset in itertools.combinations(pareto_front, M):
        subset_tensor = torch.stack(subset)  # Convert to tensor
        hv = hv_computer.compute(subset_tensor)
        c+=1
        # print(c)
        if hv > best_hv:
            best_hv = hv
            best_subset = subset_tensor

    # Find intersection: elements in both best_subset and pareto_front
    common_points = torch.stack([p for p in best_subset if any(torch.all(p.eq(q)) for q in pareto_front)])


    return common_points




def select_best_m_pareto_solutions_fast(obj_tensor: torch.Tensor, M: int, ref_point: torch.Tensor) -> torch.Tensor:


    assert obj_tensor.dim() == 2
    N, d = obj_tensor.shape
    assert d in (2, 3)

    print('fast trick')

    pareto_mask = is_non_dominated(obj_tensor)
    P = obj_tensor[pareto_mask]

    if P.shape[0] <= M:
        return P

    if d == 2:
        # ---- exact 2D DP (same as before; unchanged) ----
        P_min = -P
        r_min = -ref_point

        order = torch.argsort(P_min[:, 0])
        P_min = P_min[order]

        keep, best_y = [], float('inf')
        for i in range(P_min.size(0)):
            yi = float(P_min[i, 1])
            if yi < best_y:
                keep.append(i); best_y = yi
        P_min = P_min[keep]
        x = P_min[:, 0].double(); y = P_min[:, 1].double()
        r1, r2 = float(r_min[0]), float(r_min[1])
        n = P_min.size(0)

        NEG = -1e300
        DP = torch.full((n + 1, M + 1), NEG, dtype=torch.float64, device=P.device)
        PREV = torch.full((n + 1, M + 1), -1, dtype=torch.long, device=P.device)
        DP[0, 0] = 0.0

        y_all = torch.cat([torch.tensor([r2], dtype=torch.float64, device=P.device), y], dim=0)
        x_all = torch.cat([torch.tensor([float('nan')], dtype=torch.float64, device=P.device), x], dim=0)

        for i in range(1, n + 1):
            DP[i, 0] = 0.0
            upto = min(i, M)
            xi = float(x_all[i]); yi = float(y_all[i])
            width = max(0.0, r1 - xi)
            for t in range(1, upto + 1):
                best, arg = NEG, -1
                for j in range(t - 1, i):
                    contrib = width * max(0.0, (float(y_all[j]) - yi))
                    val = DP[j, t - 1] + contrib
                    if val > best:
                        best, arg = val, j
                DP[i, t], PREV[i, t] = best, arg

        i_best = int(torch.argmax(DP[:, M]))
        t, i = M, i_best
        chosen_idx = []
        while t > 0 and i > 0:
            j = int(PREV[i, t])
            chosen_idx.append(i - 1)
            i, t = j, t - 1
        chosen_idx.reverse()

        chosen = (-P_min[chosen_idx]).to(dtype=obj_tensor.dtype, device=obj_tensor.device)
        return chosen

    else:
        hv = Hypervolume(ref_point=ref_point.to(device=P.device, dtype=P.dtype))
        # Filter invalid points
        good = (P > ref_point.to(device=P.device, dtype=P.dtype)).all(dim=1)
        P = P[good]
        if P.size(0) <= M:
            return P

        K = P.size(0)

        def _hv_val(Y):
            v = hv.compute(Y)
            return float(v.detach().cpu().item()) if isinstance(v, torch.Tensor) else float(v)

        # ---- single-box volumes (for UB heuristic) ----
        single_vol = torch.prod(torch.clamp(P - ref_point, min=0), dim=1).cpu().tolist()

        # ---- greedy LB ----
        def greedy_lb():
            chosen, S = [], torch.empty((0, 3), device=P.device, dtype=P.dtype)
            best_hv = 0.0
            remaining = list(range(K))
            for _ in range(M):
                best_gain, best_i = -1.0, None
                for i in remaining:
                    hv_val = _hv_val(torch.cat([S, P[i].unsqueeze(0)], dim=0))
                    gain = hv_val - best_hv
                    if gain > best_gain:
                        best_gain, best_i = gain, i
                if best_i is None or best_gain <= 0:
                    break
                chosen.append(best_i)
                S = torch.cat([S, P[best_i].unsqueeze(0)], dim=0)
                best_hv = _hv_val(S)
                remaining.remove(best_i)
            return chosen, best_hv

        best_set, best_hv = greedy_lb()

        # ---- order candidates by box volume desc ----
        order = sorted(range(K), key=lambda i: single_vol[i], reverse=True)
        ordered_vols = [single_vol[i] for i in order]
        prefix = [0.0]
        for v in ordered_vols:
            prefix.append(prefix[-1] + v)

        def sum_top_from(pos, cnt):
            end = min(len(ordered_vols), pos + cnt)
            return prefix[end] - prefix[pos]

        # ---- cache HVs of partial subsets ----
        from functools import lru_cache
        @lru_cache(maxsize=None)
        def hv_of_indices(fro_idxs):
            if not fro_idxs:
                return 0.0
            idxs = list(fro_idxs)
            Y = P[idxs, :]
            return _hv_val(Y)

        def dfs(start_pos, chosen_idxs, hv_curr):
            nonlocal best_hv, best_set
            chosen_len = len(chosen_idxs)
            capacity = M - chosen_len
            if capacity == 0:
                if hv_curr > best_hv + 1e-15:
                    best_hv = hv_curr
                    best_set = chosen_idxs[:]
                return

            # UB = current HV + sum of top-(capacity) remaining box vols
            UB = hv_curr + sum_top_from(start_pos, capacity)
            if UB <= best_hv + 1e-15:
                return  # prune

            for pos in range(start_pos, K):
                i = order[pos]
                new_idxs = chosen_idxs + [i]
                hv_new = hv_of_indices(frozenset(new_idxs))
                dfs(pos + 1, new_idxs, hv_new)
                # optional quick sibling pruning
                if capacity - 1 > 0:
                    UB_next = hv_new + sum_top_from(pos + 1, capacity - 1)
                    if UB_next <= best_hv + 1e-15:
                        break

        dfs(0, [], 0.0)
        best_set_sorted = sorted(best_set)
    
        return P[best_set_sorted]



def print_variable_sizes(msg="Variable Sizes"):
    print(f"{msg}:")
    for name, var in globals().items():
        try:
            size = sys.getsizeof(var)
            if size > 1e6:  # Show only variables larger than 1MB
                print(f"{name}: {size / 1024 ** 2:.2f} MB")
        except TypeError:
            pass



def M_HV(x,M,input_dim,model,ref_point,beta=1.5):

    x = torch.tensor(x)
    x = x.reshape(M,input_dim)

    with torch.no_grad(): 
        posterior_pred = model.posterior(x) 
        mean_temp = posterior_pred.mean
        std_temp = posterior_pred.variance.sqrt()

        ucb_value = mean_temp + beta*std_temp

        hv_computer = Hypervolume(ref_point)
        hv_val = hv_computer.compute(ucb_value)


    return hv_val  


def M_UCB_optimize(M,input_dim,objective_num,model,ref_point,choice_M,beta=1.5):

    dim = M*input_dim
    opts ={'maxiter':200,'maxfun':200,'disp': False}

    standard_bounds = torch.zeros(2, dim)
    standard_bounds[1] = 1
    standard_bounds = standard_bounds.numpy()

    restart_num = 15
    X_candidate = []
    AF_candidate = []

    for i in range(restart_num):

        print('M_UCB_optimize inner',i)

        if i == -1:
            x0 = choice_M.detach().numpy().reshape(-1,)

        else:
            init_X = np.random.uniform(low=0, high=1, size=(256, dim))

            val_holder = []
            for x_temp in init_X:
                with torch.no_grad():  
                    val_temp = M_HV(x_temp,M,input_dim,model,ref_point,beta)
                    val_holder.append(val_temp)
            
            x0=init_X[np.argmax(val_holder)]


        res = minimize(lambda x: -M_HV(x=x,M=M,input_dim=input_dim,model=model,ref_point=ref_point,beta=beta),x0,
                                    bounds=standard_bounds.T,method="L-BFGS-B",options=opts) #L-BFGS-B  nelder-mead(better for rough function) Powell

        x_best =  res.x  

        with torch.no_grad():
            best_val_temp = M_HV(x_best,M,input_dim,model,ref_point,beta)
        
        
        X_candidate.append(x_best)
        AF_candidate.append(best_val_temp)


    res = X_candidate[np.argmax(AF_candidate)]


    del val_holder, X_candidate, AF_candidate
    gc.collect()
    torch.cuda.empty_cache()

    return res