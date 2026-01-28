if __name__ == '__main__':
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    import argparse
    import time

    import numpy as np
    import torch
    from botorch import fit_gpytorch_mll
    from botorch.exceptions import BadInitialCandidatesWarning
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.sampling.stochastic_samplers import StochasticSampler
    from botorch.utils.multi_objective.box_decompositions.dominated import (DominatedPartitioning)
    from botorch.utils.multi_objective.pareto import is_non_dominated
    from botorch.utils.multi_objective.hypervolume import Hypervolume
    from botorch.utils.transforms import unnormalize, normalize
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (FastNondominatedPartitioning,)
    from botorch.test_functions.multi_objective import BraninCurrin,DTLZ1,DTLZ2,VehicleSafety,Penicillin
    from botorch.utils.sampling import sample_simplex
    from botorch.acquisition.objective import GenericMCObjective
    from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
    from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
    from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
    from botorch.acquisition.multi_objective.joint_entropy_search import qLowerBoundMultiObjectiveJointEntropySearch
    from botorch.acquisition.multi_objective.utils import (
    compute_sample_box_decomposition,
    random_search_optimizer,
    sample_optimal_points,)


    import warnings
    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)


    from mobo_fixed_size.variable import NUM_RESTARTS,RAW_SAMPLES,OPTIONS,BATCH_SIZE,SAMPLE_NUM
    from mobo_fixed_size.utilis import generate_initial_data,initialize_model,select_best_m_pareto_solutions_fast
    from mobo_fixed_size.acquisition import (optimize_qehvi_fixed_size_PF,
                                             qExpectedHypervolumeImprovement_FixedSizedParetoFront,get_reference_chebyshev_scalarization,
                                             qExpectedHypervolumeImprovement)
    from mobo_fixed_size.obj_function import VLMOP3,SnAr,Four_bar_truss_design,VLMOP2

    import random

    torch.set_default_dtype(torch.double)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    
    torch.set_printoptions(precision=10)

    parser = argparse.ArgumentParser(description="Run the script with named arguments.")

    parser.add_argument("--random_seed", type=int,required=True, help="Random seed value")
    parser.add_argument("--M_type", type=int, required=True, help="M_type tells how many soltuions we want in a PF; M_type=0 means we do not care about the size")
    parser.add_argument("--M_check", type=int, required=True, help="the bestHV given M_check; This argument is used as M_type can be 0")
    parser.add_argument("--type", type=str,required=True, help="whether we use UCB method to approximate Pareto front")

    args = parser.parse_args()

    random_seed = args.random_seed
    M_type = args.M_type
    M_check = args.M_check
    type = args.type


    print(f"Random Seed: {random_seed}")
    print(f"M_type: {M_type}")
    print(f"M_check: {M_check}")
    print(f"type: {type}")

    no_M_information = {'EHVI','sobol','ParEGO_ref','ParEGO','JES','UCB'}
    if type in no_M_information:
        assert M_type == 0, "M should be 0"
    else:
        assert M_type > 0, "M should be larger than 0"


    ########################### define the problem #############################################
    problem_information = []


    temp={}
    temp['experiment_name'] = 'Four_bar_truss_design'  
    temp['input_dim'] = 4
    temp['obj_num'] = 2
    temp['problem'] = Four_bar_truss_design(negate=True)
    temp['ref_point'] =  Four_bar_truss_design(negate=True)._ref_point 
    temp['NOISE_SE'] = 0.
    temp['iter_num'] = 50
    problem_information.append(temp)


    temp={}
    temp['experiment_name'] = 'SnAr'  
    temp['input_dim'] = 4
    temp['obj_num'] = 2
    temp['problem'] = SnAr(negate=True)
    temp['ref_point'] =  SnAr(negate=True).ref_point 
    temp['NOISE_SE'] = 0.
    temp['iter_num'] = 100
    problem_information.append(temp)
    

    temp={}
    temp['experiment_name'] = 'BraninCurrin'  
    temp['input_dim'] = 2
    temp['obj_num'] = 2
    temp['problem'] = BraninCurrin(negate=True)
    temp['ref_point'] =  BraninCurrin(negate=True).ref_point 
    temp['NOISE_SE'] = 0.
    temp['iter_num'] = 50
    problem_information.append(temp)


    temp={}
    temp['experiment_name'] = 'DTLZ1'   
    temp['input_dim'] = 3
    temp['obj_num'] = 2
    temp['problem'] = DTLZ1(dim=3,negate=True)  
    temp['ref_point'] = torch.tensor([-50.,-50.])
    temp['NOISE_SE'] = 0.
    temp['iter_num'] = 50
    problem_information.append(temp)


    temp={}
    temp['experiment_name'] = 'DTLZ2'  
    temp['input_dim'] = 3
    temp['obj_num'] = 2
    temp['problem'] = DTLZ2(dim=3,negate=True)  
    temp['ref_point'] = DTLZ2(dim=3,negate=True).ref_point 
    temp['NOISE_SE'] = 0.
    temp['iter_num'] = 50
    problem_information.append(temp)


    temp={}
    temp['experiment_name'] = 'DTLZ2_6D'  
    temp['input_dim'] = 6
    temp['obj_num'] = 2
    temp['problem'] = DTLZ2(dim=6,negate=True)  
    temp['ref_point'] = DTLZ2(dim=6,negate=True).ref_point 
    temp['NOISE_SE'] = 0.
    temp['iter_num'] = 100
    problem_information.append(temp)


    temp={}
    temp['experiment_name'] = 'DTLZ2_10D'  
    temp['input_dim'] = 10
    temp['obj_num'] = 2
    temp['problem'] = DTLZ2(dim=10,negate=True)  
    temp['ref_point'] = DTLZ2(dim=10,negate=True).ref_point 
    temp['NOISE_SE'] = 0.
    temp['iter_num'] = 175
    problem_information.append(temp)


    temp={}
    temp['experiment_name'] = 'VehicleSafety'  
    temp['input_dim'] = 5
    temp['obj_num'] = 3
    temp['problem'] = VehicleSafety(negate=True) 
    temp['ref_point'] = VehicleSafety(negate=True) .ref_point 
    temp['NOISE_SE'] = 0.
    temp['iter_num'] = 50
    problem_information.append(temp)


    temp={}
    temp['experiment_name'] = 'VLMOP3'  
    temp['input_dim'] = 2
    temp['obj_num'] = 3
    temp['problem'] = VLMOP3() 
    temp['ref_point'] = VLMOP3().ref_point 
    temp['NOISE_SE'] = 0.
    temp['iter_num'] = 50
    problem_information.append(temp)


    temp={}
    temp['experiment_name'] = 'VLMOP2'  
    temp['input_dim'] = 6
    temp['obj_num'] = 2
    temp['problem'] = VLMOP2() 
    temp['ref_point'] = VLMOP2().ref_point 
    temp['NOISE_SE'] = 0.
    temp['iter_num'] = 100
    problem_information.append(temp)


    temp={}
    temp['experiment_name'] = 'Penicillin'  
    temp['input_dim'] = 7
    temp['obj_num'] = 3
    temp['problem'] = Penicillin(negate=True) 
    temp['ref_point'] = Penicillin(negate=True).ref_point 
    temp['NOISE_SE'] = 0.
    temp['iter_num'] = 100
    problem_information.append(temp)



    for information in problem_information:

        time_record=[]
        AF_record=[]
        AF_gradient_record=[]
        opt_iter_record = []

        function_name = information['experiment_name'] 
        dim = information['input_dim'] 
        obj_num = information['obj_num'] 
        initial_num = min(3*dim,10)
        problem = information['problem'] 
        NOISE_SE = information['NOISE_SE'] 
        iter_num = information['iter_num'] 
        ref_point= information['ref_point'] 

        print('ref point is ', ref_point )


        filename = [function_name, type, f'M={M_type}', str(random_seed)]
        print(filename)
        results_filename = '_'.join(filename)

        # these two are for GP-Hedge algorithm ################
        EHVI_M_IR_record = torch.zeros(iter_num) 
        HD_UCB_IR_record = torch.zeros(iter_num)

        EHVI_M_g_record = torch.zeros(iter_num)
        HD_UCB_g_record = torch.zeros(iter_num)

        start = False
        ###################################################


        hvs_all = []
        hvs_M = []

        af_value = []
        af_value_best = []


        train_x_qehvi, train_obj_qehvi, train_obj_true_qehvi = generate_initial_data(problem,initial_num,random_seed,NOISE_SE)
        train_x_qehvi = train_x_qehvi.to(dtype=torch.double)
        train_obj_qehvi = train_obj_qehvi.to(dtype=torch.double)
        train_obj_true_qehvi = train_obj_true_qehvi.to(dtype=torch.double)
        ref_point = ref_point.to(dtype=torch.double)

        # compute hypervolume (all Pareto Front)
        bd = DominatedPartitioning(ref_point=ref_point, Y=train_obj_true_qehvi)
        volume = bd.compute_hypervolume().item()
        hvs_all.append(volume)


        # compute hypervolume (choose the M solution as the final one)
        M = M_check
        pareto_mask = is_non_dominated(train_obj_qehvi)
        pareto_front_temp = train_obj_qehvi[pareto_mask]

        ref_mask = (pareto_front_temp > ref_point).all(dim=-1) 
        pareto_front_temp = pareto_front_temp[ref_mask]


        choice_M = select_best_m_pareto_solutions_fast(pareto_front_temp, M, ref_point)
            
        hv_computer = Hypervolume(ref_point)
        hvs_M.append(hv_computer.compute(choice_M))



        indices = []
        for row in choice_M:
            for i, y_row in enumerate(train_obj_qehvi):
                if torch.all(row == y_row):  # Check if the entire row matches
                    indices.append(i)
                    break  # Stop searching once a match is found

        try:
            indices = torch.tensor(indices)
            pareto_set_M = normalize(train_x_qehvi[indices].detach(), bounds=problem.bounds)   

        except:
            pareto_set_M = torch.tensor([]).reshape(0,dim)



        for iter in range(iter_num):

            torch.manual_seed(1234+iter)

            # fit the models
            mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi,problem)

            try:
                fit_gpytorch_mll(mll_qehvi)

                
                prev_model_state = {
                    name: param.detach().clone() 
                    for name, param in model_qehvi.state_dict().items()
                }
                
                print("successful model fitting")

            except RuntimeError as e:
                if "Failed to sample a feasible parameter" in str(e):
                    print("!Model fitting failed, using previous parameters")
                    if prev_model_state is not None:
                        model_qehvi.load_state_dict(prev_model_state)
                    else:
                        print("WARNING: No previous model parameters available. Using initialized parameters.")
                else:
                    raise e
    
    
            t0 = time.monotonic()
            # acquisition function
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([SAMPLE_NUM])) #StochasticSampler SobolQMCNormalSampler


            if type == 'sobol':
                candidates,_,_ = generate_initial_data(problem,1,random.randint(0, 10000),NOISE_SE)
                candidates = normalize(candidates.detach(), bounds=problem.bounds)
                val = 0.

            elif type == 'ParEGO':

                with torch.no_grad():
                    pred = model_qehvi.posterior(train_x_qehvi).mean
                    
                
                acq_func_list = []
                for _ in range(1):
                    weights = sample_simplex(obj_num).squeeze()
                    objective = GenericMCObjective(
                        get_chebyshev_scalarization(weights=weights, Y=pred)
                    )
                    
                    acq_func = qNoisyExpectedImprovement(  
                        model=model_qehvi,
                        objective=objective,
                        X_baseline=train_x_qehvi,
                        sampler=sampler,
                        prune_baseline=True,
                    )
                    acq_func_list.append(acq_func)

                # optimize
                standard_bounds = torch.zeros(2, problem.dim)
                standard_bounds[1] = 1

                candidates, val = optimize_acqf_list(
                                                acq_function_list=acq_func_list,
                                                bounds=standard_bounds,
                                                num_restarts=NUM_RESTARTS,
                                                raw_samples=RAW_SAMPLES,  # used for intialization heuristic
                                                options=OPTIONS)
                
            
            elif type == 'ParEGO_ref':

                with torch.no_grad():
                    pred = model_qehvi.posterior(train_x_qehvi).mean
                    
                
                try:
                    acq_func_list = []
                    for _ in range(1):
                        weights = sample_simplex(obj_num).squeeze()
                        objective = GenericMCObjective(
                            get_reference_chebyshev_scalarization(weights=weights, Y=pred, reference_point=ref_point )
                        )


                        acq_func = qNoisyExpectedImprovement(  
                            model=model_qehvi,
                            objective=objective,
                            X_baseline=train_x_qehvi,
                            sampler=sampler,
                            prune_baseline=True,
                        )
                        acq_func_list.append(acq_func)


                    # optimize
                    standard_bounds = torch.zeros(2, problem.dim)
                    standard_bounds[1] = 1

                    candidates, val = optimize_acqf_list(
                                                    acq_function_list=acq_func_list,
                                                    bounds=standard_bounds,
                                                    num_restarts=NUM_RESTARTS,
                                                    raw_samples=RAW_SAMPLES,  # used for intialization heuristic
                                                    options=OPTIONS)

                except:
                    with torch.no_grad():
                        pred = model_qehvi.posterior(train_x_qehvi).mean
                        
                    
                    acq_func_list = []
                    for _ in range(1):
                        weights = sample_simplex(obj_num).squeeze()
                        objective = GenericMCObjective(
                            get_chebyshev_scalarization(weights=weights, Y=pred)
                        )
                        
                        acq_func = qNoisyExpectedImprovement(  
                            model=model_qehvi,
                            objective=objective,
                            X_baseline=train_x_qehvi,
                            sampler=sampler,
                            prune_baseline=True,
                        )
                        acq_func_list.append(acq_func)


                        # optimize
                        standard_bounds = torch.zeros(2, problem.dim)
                        standard_bounds[1] = 1

                        candidates, val = optimize_acqf_list(
                                                        acq_function_list=acq_func_list,
                                                        bounds=standard_bounds,
                                                        num_restarts=NUM_RESTARTS,
                                                        raw_samples=RAW_SAMPLES,  
                                                        options=OPTIONS)
                    

            
            elif  type == 'JES':
                try:
                    standard_bounds = torch.zeros(2, problem.dim)
                    standard_bounds[1] = 1
                
                    train_x_qehvi = train_x_qehvi.to(dtype=torch.double)
                    train_obj_qehvi = train_obj_qehvi.to(dtype=torch.double)
                    train_obj_true_qehvi = train_obj_true_qehvi.to(dtype=torch.double)
                    ref_point = ref_point.to(dtype=torch.double)

                    optimizer_kwargs = {
                            "pop_size": 500,
                            "max_tries": 10,
                        }
                    num_pareto_samples = 4
                    num_pareto_points = 4

                    ps, pf = sample_optimal_points(
                                                model=model_qehvi,
                                                bounds=problem.bounds,
                                                num_samples=num_pareto_samples,
                                                num_points=num_pareto_points,
                                                optimizer=random_search_optimizer,
                                                optimizer_kwargs=optimizer_kwargs,
                                            )
                    

                    hypercell_bounds = compute_sample_box_decomposition(pf)

                    # Here we use the lower bound estimates for the MES and JES
                    jes_lb = qLowerBoundMultiObjectiveJointEntropySearch(
                        model=model_qehvi,
                        pareto_sets=ps,
                        pareto_fronts=pf,
                        hypercell_bounds=hypercell_bounds,
                        estimation_type="LB",
                    )
                
                    candidates, val = optimize_acqf(
                                acq_function=jes_lb,
                                bounds=standard_bounds,
                                q=1,
                                num_restarts=3,
                                raw_samples=256,
                                sequential=True,
                                options={"batch_limit": 5,"maxiter": 200, "sample_around_best": False} 
                            )

                except:
                    with torch.no_grad():
                        pred = model_qehvi.posterior(train_x_qehvi).mean
                    
                
                    acq_func_list = []
                    for _ in range(1):
                        weights = sample_simplex(obj_num).squeeze()
                        objective = GenericMCObjective(
                            get_chebyshev_scalarization(weights=weights, Y=pred)
                        )
                        
                        
                        acq_func = qNoisyExpectedImprovement(  
                            model=model_qehvi,
                            objective=objective,
                            X_baseline=train_x_qehvi,
                            sampler=sampler,
                            prune_baseline=True,
                        )
                        acq_func_list.append(acq_func)


                    # optimize
                    standard_bounds = torch.zeros(2, problem.dim)
                    standard_bounds[1] = 1

                    candidates, val = optimize_acqf_list(
                                                    acq_function_list=acq_func_list,
                                                    bounds=standard_bounds,
                                                    num_restarts=NUM_RESTARTS,
                                                    raw_samples=RAW_SAMPLES,  # used for intialization heuristic
                                                    options=OPTIONS)
                    

            elif type == 'EHVI':
                candidates, val = optimize_qehvi_fixed_size_PF('EHVI',model_qehvi,0,pareto_front_temp,
                                                                pareto_set_M,train_x_qehvi, 
                                                            train_obj_qehvi,sampler,ref_point,problem,batch_size=BATCH_SIZE)  
          


            elif type == 'EHVI_M':

                if choice_M.shape[0]<M_type: 
                    candidates, val = optimize_qehvi_fixed_size_PF('EHVI',model_qehvi,0,pareto_front_temp,
                                                                pareto_set_M,train_x_qehvi, 
                                                            train_obj_qehvi,sampler,ref_point,problem,batch_size=BATCH_SIZE)  
                
                else:

                    candidates, val = optimize_qehvi_fixed_size_PF('EHVI_M',model_qehvi,M_type,choice_M, 
                                                                                    pareto_set_M, train_x_qehvi, 
                                                                                    train_obj_qehvi,sampler,ref_point,problem,batch_size=BATCH_SIZE)  
                    


            elif type == 'HD_EI':
                if choice_M.shape[0]<M_type: 
                    print('not enough M')
                    candidates, val = optimize_qehvi_fixed_size_PF('EHVI',model_qehvi,0,pareto_front_temp,
                                                                pareto_set_M,train_x_qehvi, 
                                                            train_obj_qehvi,sampler,ref_point,problem,batch_size=BATCH_SIZE)  

                
                else:
                    candidates_fixedsize, val_fixedsize = optimize_qehvi_fixed_size_PF('HD_EI',model_qehvi,M_type,choice_M, 
                                                                        pareto_set_M, train_x_qehvi, 
                                                                        train_obj_qehvi,sampler,ref_point,problem,batch_size=BATCH_SIZE)  
                

                    partitioning = FastNondominatedPartitioning(ref_point=ref_point,Y=pareto_front_temp)
                    acq_func = qExpectedHypervolumeImprovement(
                                                                model=model_qehvi,
                                                                ref_point=ref_point,
                                                                partitioning=partitioning,
                                                                sampler=sampler,
                                                                )
                        
                    
                    check_val = acq_func(candidates_fixedsize.reshape(M_type,1,-1))
                    candidates = candidates_fixedsize[torch.argmax(check_val)].reshape(1,-1)
                    val = torch.max(check_val)



            elif type == 'HD_logEI':
                if choice_M.shape[0]<M_type: 
                    print('not enough M')
                    candidates, val = optimize_qehvi_fixed_size_PF('EHVI',model_qehvi,0,pareto_front_temp,
                                                                pareto_set_M,train_x_qehvi, 
                                                            train_obj_qehvi,sampler,ref_point,problem,batch_size=BATCH_SIZE)  
                
                else:
                
                    candidates_fixedsize, val_fixedsize = optimize_qehvi_fixed_size_PF('HD_logEI',model_qehvi,M_type,choice_M, 
                                                                        pareto_set_M, train_x_qehvi, 
                                                                        train_obj_qehvi,sampler,ref_point,problem,batch_size=BATCH_SIZE,logEI=True)  
                

                    partitioning = FastNondominatedPartitioning(ref_point=ref_point,Y=pareto_front_temp)
                    acq_func = qExpectedHypervolumeImprovement(
                                                                model=model_qehvi,
                                                                ref_point=ref_point,
                                                                partitioning=partitioning,
                                                                sampler=sampler,
                                                                )
                        
                    
                    check_val = acq_func(candidates_fixedsize.reshape(M_type,1,-1))
                    candidates = candidates_fixedsize[torch.argmax(check_val)].reshape(1,-1)
                    val = torch.max(check_val)

              

            elif type == 'HD_UCB':
                if choice_M.shape[0]<M_type: 
                    print('not enough M')
                    candidates, val = optimize_qehvi_fixed_size_PF('EHVI',model_qehvi,0,pareto_front_temp,
                                                                pareto_set_M,train_x_qehvi, 
                                                            train_obj_qehvi,sampler,ref_point,problem,batch_size=BATCH_SIZE)  
                    
                else:
                    candidates_fixedsize, val_fixedsize = optimize_qehvi_fixed_size_PF('HD_UCB',model_qehvi,M_type,choice_M, 
                                                                        pareto_set_M, train_x_qehvi, 
                                                                        train_obj_qehvi,sampler,ref_point,problem,batch_size=BATCH_SIZE,beta=2.)  
                    

                    partitioning = FastNondominatedPartitioning(ref_point=ref_point,Y=pareto_front_temp)
                    acq_func = qExpectedHypervolumeImprovement(
                                                                model=model_qehvi,
                                                                ref_point=ref_point,
                                                                partitioning=partitioning,
                                                                sampler=sampler,
                                                                )
                        
                    
                    check_val = acq_func(candidates_fixedsize.reshape(M_type,1,-1))
                    candidates = candidates_fixedsize[torch.argmax(check_val)].reshape(1,-1)
                    val = torch.max(check_val)


        

            new_x_qehvi = unnormalize(candidates.detach(), bounds=problem.bounds)
            new_obj_true_qehvi = problem(new_x_qehvi)
            new_obj_qehvi = new_obj_true_qehvi + torch.randn_like(new_obj_true_qehvi) * NOISE_SE


            print('next solution ',new_x_qehvi)
            print('AF value: ',val)

            # update
            train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
            train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi])
            train_obj_true_qehvi = torch.cat([train_obj_true_qehvi, new_obj_true_qehvi])


            bd = DominatedPartitioning(ref_point=ref_point , Y=train_obj_qehvi)
            volume = bd.compute_hypervolume().item()
            hvs_all.append(volume)

            t1 = time.monotonic()

            # choose the M solution as the final one
            M = M_check
            pareto_mask = is_non_dominated(train_obj_qehvi)
            pareto_front_temp = train_obj_qehvi[pareto_mask]      

            ref_mask = (pareto_front_temp > ref_point).all(dim=-1)
            pareto_front_temp = pareto_front_temp[ref_mask]

            print('full pareto set size ',pareto_front_temp.shape)  
            choice_M = select_best_m_pareto_solutions_fast(pareto_front_temp, M, ref_point)
            
            hv_computer = Hypervolume(ref_point)
            hvs_M.append(hv_computer.compute(choice_M))
            print('M solutions: ',choice_M)


            print(
                f"\nBatch {iter:>2}: Hypervolume= "
                f" {hvs_M[-1]:>4.2f}, "
                f"time = {t1-t0:>4.2f}.",
                end="",)
            
            time_record.append(t1-t0)


            indices = []
            for row in choice_M:
                for i, y_row in enumerate(train_obj_qehvi):
                    if torch.all(row == y_row):  # Check if the entire row matches
                        indices.append(i)
                        break  # Stop searching once a match is found

            try:
                indices = torch.tensor(indices)
                pareto_set_M = normalize(train_x_qehvi[indices].detach(), bounds=problem.bounds)   
            except:
                pareto_set_M = torch.tensor([]).reshape(0,dim)


            # save the result
            try:
                np.savetxt(f'exp/M{M_check}/{results_filename}', hvs_M)
            except:
                os.makedirs(f'exp/M{M_check}', exist_ok=True)
                np.savetxt(f'exp/M{M_check}/{results_filename}', hvs_M)
                

