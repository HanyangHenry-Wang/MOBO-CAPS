import subprocess

no_M_information = {'EHVI','sobol','ParEGO_ref','ParEGO','JES','UCB'}

type_list = ['EHVI']
M_check_record = [1,2,3,4,5]
indices = range(0,20)

for type in type_list:

    if type in no_M_information:
        size_constraint = False
    else:
        size_constraint = True

    for M_check in M_check_record:

        if size_constraint:
            M_type = M_check
        else:
            M_type = 0

        # Run the script with each index
        for index in indices:
            command = f"python run_experiment.py --random_seed {index} --M_type {M_type} --M_check {M_check} --type {type}"
            print(f"Running: {command}")
            subprocess.run(command, shell=True)
