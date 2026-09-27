if __name__ == '__main__':  # Standard MOBO JES, tutorial logic + multiple test problems + CAPS evaluation
    import os
    import time
    import random
    import warnings
    import argparse

    import numpy as np
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models.gp_regression import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.models.transforms.input import Normalize
    from botorch.optim import optimize_acqf
    from botorch.utils.sampling import draw_sobol_samples
    from botorch.test_functions.multi_objective import (
        ZDT1, BraninCurrin, DTLZ1, DTLZ2, VehicleSafety, Penicillin
    )
    from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

    from botorch.acquisition.multi_objective.utils import (
        compute_sample_box_decomposition,
        random_search_optimizer,
        sample_optimal_points,
    )
    from botorch.acquisition.multi_objective.joint_entropy_search import (
        qLowerBoundMultiObjectiveJointEntropySearch,
    )
    from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
    from botorch.utils.multi_objective.pareto import is_non_dominated
    from botorch.utils.multi_objective.hypervolume import Hypervolume

    from mobo_fixed_size.obj_function import VLMOP3, SnAr, Four_bar_truss_design, VLMOP2
    from mobo_fixed_size.utilis import select_best_m_pareto_solutions_fast

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    torch.set_default_dtype(torch.double)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    tkwargs = {"dtype": dtype, "device": device}

    torch.set_printoptions(precision=10)

    parser = argparse.ArgumentParser(description="Run standard MOBO JES on multiple test problems.")
    parser.add_argument("--random_seed", type=int, required=True, help="Random seed value")
    args = parser.parse_args()

    random_seed = args.random_seed

    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    print(f"Random Seed: {random_seed}")

    # ============================================================
    # Problem list
    # ============================================================
    problem_information = []

    temp = {}
    temp["experiment_name"] = "BraninCurrin2"
    temp["input_dim"] = 2
    temp["obj_num"] = 2
    temp["problem"] = BraninCurrin(negate=True)
    temp["n_init"] = min(3*2,10)
    temp["iter_num"] = 50
    problem_information.append(temp)


    temp = {}
    temp["experiment_name"] = "DTLZ1"
    temp["input_dim"] = 3
    temp["obj_num"] = 2
    temp["problem"] = DTLZ1(dim=3, num_objectives=2, negate=True)
    temp["n_init"] = min(3*3,10)
    temp["iter_num"] = 50
    problem_information.append(temp)


    temp = {}
    temp["experiment_name"] = "DTLZ2"
    temp["input_dim"] = 3
    temp["obj_num"] = 2
    temp["problem"] = DTLZ2(dim=3, num_objectives=2, negate=True)
    temp["n_init"] = min(3*3,10)
    temp["iter_num"] = 50
    problem_information.append(temp)


    temp = {}
    temp["experiment_name"] = "VehicleSafety"
    temp["input_dim"] = 5
    temp["obj_num"] = 3
    temp["problem"] = VehicleSafety(negate=True)
    temp["n_init"] = min(3*5,10)
    temp["iter_num"] = 50
    problem_information.append(temp)

    temp = {}
    temp["experiment_name"] = "VLMOP3"
    temp["input_dim"] = 2
    temp["obj_num"] = 3
    temp["problem"] = VLMOP3()
    temp["n_init"] = min(3*2,10)
    temp["iter_num"] = 50
    problem_information.append(temp)

    temp = {}
    temp["experiment_name"] = "VLMOP2"
    temp["input_dim"] = 6
    temp["obj_num"] = 2
    temp["problem"] = VLMOP2()
    temp["n_init"] = min(3*6,10)
    temp["iter_num"] = 100
    problem_information.append(temp)

    temp = {}
    temp["experiment_name"] = "SnAr"
    temp["input_dim"] = 4
    temp["obj_num"] = 2
    temp["problem"] = SnAr(negate=True)
    temp["n_init"] = min(3*4,10)
    temp["iter_num"] = 50
    problem_information.append(temp)

    temp = {}
    temp["experiment_name"] = "Four_bar_truss_design"
    temp["input_dim"] = 4
    temp["obj_num"] = 2
    temp["problem"] = Four_bar_truss_design(negate=True)
    temp["n_init"] = min(3*4,10)
    temp["iter_num"] = 50
    problem_information.append(temp)

    temp={}
    temp['experiment_name'] = 'Penicillin'  
    temp['input_dim'] = 7
    temp['obj_num'] = 3
    temp['problem'] = Penicillin(negate=True) 
    temp['ref_point'] = Penicillin(negate=True).ref_point 
    temp["n_init"] = min(3*4,10)
    temp['NOISE_SE'] = 0.
    temp['iter_num'] = 100 
    problem_information.append(temp)

    # ============================================================
    # Run all selected problems
    # ============================================================
    for information in problem_information:

        function_name = information["experiment_name"]
        dim = information["input_dim"]
        obj_num = information["obj_num"]
        n = information["n_init"]
        iter_num = information["iter_num"]
        problem = information["problem"]

        bounds = problem.bounds.to(**tkwargs)
        if function_name == 'DTLZ1':
            ref_point = torch.tensor([-50.,-50.]).to(**tkwargs)
        elif function_name == 'Four_bar_truss_design':
            ref_point = Four_bar_truss_design(negate=True)._ref_point.to(**tkwargs)
        else:
            ref_point = problem.ref_point.to(**tkwargs)


        print("\n" + "#" * 100)
        print(f"Start problem: {function_name}")
        print("bounds = ", bounds)
        print("ref_point = ", ref_point)
        print("#" * 100)

        filename = [function_name, "JES", str(random_seed)]
        results_filename = "_".join(filename)

        hvs_all = []
        PF_number_list = []
        time_record = []

        # CAPS evaluation records
        hvs_M_list = []
        for M in [1, 2, 3, 4, 5]:
            hvs_M_list.append([])

        # ---------------------------------------------------------
        # initial data
        # ---------------------------------------------------------
        torch.manual_seed(random_seed)
        train_X = draw_sobol_samples(bounds=bounds, n=n, q=1).squeeze(-2).to(**tkwargs)
        train_Y = problem(train_X).to(**tkwargs)

        bd = DominatedPartitioning(ref_point=ref_point, Y=train_Y)
        hvs_all.append(bd.compute_hypervolume())

        pareto_mask = is_non_dominated(train_Y)
        pareto_front_temp = train_Y[pareto_mask]
        pareto_set_temp = train_X[pareto_mask]

        ref_mask = (pareto_front_temp > ref_point).all(dim=-1)
        pareto_front_temp = pareto_front_temp[ref_mask]
        pareto_set_temp = pareto_set_temp[ref_mask]

        PF_number_list.append(pareto_front_temp.shape[0])

        print("initial HV = ", hvs_all[-1])
        print("initial PF size = ", PF_number_list[-1])

        # initial CAPS evaluation
        for M in [1, 2, 3, 4, 5]:
            if pareto_front_temp.shape[0] == 0:
                hvs_M_list[M - 1].append(0.0)
            else:
                choice_M = select_best_m_pareto_solutions_fast(
                    pareto_front_temp, M, ref_point
                )
                hv_computer = Hypervolume(ref_point)
                hvs_M_list[M - 1].append(hv_computer.compute(choice_M))

        # ---------------------------------------------------------
        # BO loop
        # ---------------------------------------------------------
        for it in range(iter_num):
            print("\n" + "=" * 80)
            print(f"{function_name} | Iteration {it + 1}/{iter_num}")
            print("=" * 80)

            torch.manual_seed(1234 + it)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(1234 + it)

            

            # Step 1. fit model
            print("step 1: fit GP model")
            model = SingleTaskGP(
                train_X,
                train_Y,
                input_transform=Normalize(d=dim, bounds=bounds),
                outcome_transform=Standardize(m=obj_num),
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Step 2. sample Pareto sets / fronts
            t0 = time.monotonic()
            
            print("step 2: sample Pareto sets / fronts")
            optimizer_kwargs = {
                "pop_size": 2000,
                "max_tries": 30,
            }

            num_pareto_samples = 4
            num_pareto_points = 4

            try:
                ps, pf = sample_optimal_points(
                    model=model,
                    bounds=bounds,
                    num_samples=num_pareto_samples,
                    num_points=num_pareto_points,
                    optimizer=random_search_optimizer,
                    optimizer_kwargs=optimizer_kwargs,
                )
            except Exception as e1:
                print("first sample_optimal_points failed:", e1)
                try:
                    ps, pf = sample_optimal_points(
                        model=model,
                        bounds=bounds,
                        num_samples=2,
                        num_points=4,
                        optimizer=random_search_optimizer,
                        optimizer_kwargs={"pop_size": 4000, "max_tries": 50},
                    )
                except Exception as e2:
                    print("second sample_optimal_points failed:", e2)
                    ps, pf = sample_optimal_points(
                        model=model,
                        bounds=bounds,
                        num_samples=2,
                        num_points=2,
                        optimizer=random_search_optimizer,
                        optimizer_kwargs={"pop_size": 6000, "max_tries": 80},
                    )

            print("ps shape = ", ps.shape)
            print("pf shape = ", pf.shape)

            # Step 3. compute box decomposition
            print("step 3: compute hypercell bounds")
            hypercell_bounds = compute_sample_box_decomposition(pf)

            # Step 4. build JES
            print("step 4: build JES acquisition")
            jes_lb = qLowerBoundMultiObjectiveJointEntropySearch(
                model=model,
                pareto_sets=ps,
                pareto_fronts=pf,
                hypercell_bounds=hypercell_bounds,
                estimation_type="LB",
            )

            # Step 5. optimize JES
            print("step 5: optimize JES acquisition")

            raw_samples = 256
            num_restarts = 6

            X_rnd = draw_sobol_samples(
                bounds=bounds,
                n=raw_samples,
                q=1,
            ).squeeze(-2).to(**tkwargs)

            with torch.no_grad():
                acq_vals = jes_lb(X_rnd.unsqueeze(-2)).view(-1)

            finite_mask = torch.isfinite(acq_vals)
            X_good = X_rnd[finite_mask]
            Y_good = acq_vals[finite_mask]

            print("number of finite raw points =", X_good.shape[0])

            if X_good.shape[0] == 0:
                print("no finite raw points, fallback to random point")
                rand_idx = torch.randint(0, X_rnd.shape[0], (1,), device=X_rnd.device)
                candidate = X_rnd[rand_idx]
                acq_value = torch.tensor(float("nan"), device=X_rnd.device, dtype=X_rnd.dtype)
            else:
                if X_good.shape[0] < num_restarts:
                    num_restarts = X_good.shape[0]

                top_idx = torch.topk(Y_good, k=num_restarts).indices
                batch_initial_conditions = X_good[top_idx].unsqueeze(1)

                try:
                    candidate, acq_value = optimize_acqf(
                        acq_function=jes_lb,
                        bounds=bounds,
                        q=1,
                        num_restarts=num_restarts,
                        batch_initial_conditions=batch_initial_conditions,
                        raw_samples=None,
                        sequential=False,
                    )
                except Exception as e:
                    print("optimize_acqf failed:", e)
                    print("fallback to best finite raw point")
                    best_idx = torch.argmax(Y_good)
                    candidate = X_good[best_idx:best_idx + 1]
                    acq_value = Y_good[best_idx:best_idx + 1]

            # Step 6. evaluate candidate
            new_X = candidate.detach()
            new_Y = problem(new_X)

            t1 = time.monotonic()

            print("candidate = ", new_X)
            print("acq_value = ", acq_value)
            print("new_Y = ", new_Y)

            # Step 7. update data
            train_X = torch.cat([train_X, new_X], dim=0)
            train_Y = torch.cat([train_Y, new_Y], dim=0)

            # Step 8. record full HV
            bd = DominatedPartitioning(ref_point=ref_point, Y=train_Y)
            hvs_all.append(bd.compute_hypervolume())

            pareto_mask = is_non_dominated(train_Y)
            pareto_front_temp = train_Y[pareto_mask]
            pareto_set_temp = train_X[pareto_mask]

            ref_mask = (pareto_front_temp > ref_point).all(dim=-1)
            pareto_front_temp = pareto_front_temp[ref_mask]
            pareto_set_temp = pareto_set_temp[ref_mask]

            PF_number_list.append(pareto_front_temp.shape[0])

            # Step 9. CAPS post-selection for M = 1,...,5
            for M in [1, 2, 3, 4, 5]:
                if pareto_front_temp.shape[0] == 0:
                    hvs_M_list[M - 1].append(0.0)
                else:
                    choice_M = select_best_m_pareto_solutions_fast(
                        pareto_front_temp, M, ref_point
                    )
                    hv_computer = Hypervolume(ref_point)
                    hvs_M_list[M - 1].append(hv_computer.compute(choice_M))

            time_record.append(t1 - t0)

            print(
                f"Batch {it:>2}: Hypervolume = {hvs_all[-1]:>4.6f}, "
                f"time = {t1 - t0:>4.2f}"
            )
            print("full pareto set size = ", pareto_front_temp.shape)

            for M in [1, 2, 3, 4, 5]:
                print(f"M={M}: {hvs_M_list[M - 1][-1]}")

            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(script_dir, "JES_results")
            os.makedirs(save_dir, exist_ok=True)

            print("saving to:", save_dir)

            for M in [1, 2, 3, 4, 5]:                             
                try:
                    np.savetxt(f'JES_results/M{M}/{results_filename}', hvs_M_list[M-1])
                except:
                    os.makedirs(f'JES_results/M{M}', exist_ok=True)
                    np.savetxt(f'JES_results/M{M}/{results_filename}', hvs_M_list[M-1])



        print("total time = ", np.sum(np.array(time_record)))

        print("\nfinal full HV = ", hvs_all[-1])
        print("final M-HV values:")