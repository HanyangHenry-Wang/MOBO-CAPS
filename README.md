# MOBO-CAPS: Multi-objective Bayesian Optimization with Cardinality-Aware Pareto Selection

This repository contains the implementation of **MOBO-CAPS (Multi-objective Bayesian Optimization with Cardinality-Aware Pareto Selection)**.

## Introduction

Multi-objective Bayesian optimization (MOBO) traditionally aims to approximate the entire Pareto front. However, in many real-world applications, only a small number of solutions can ultimately be deployed due to practical constraints such as limited manufacturing capacity, experimental resources, or decision-making budget.

MOBO-CAPS considers the setting where the goal is to identify a set of exactly \(M\) solutions with high hypervolume, rather than approximating the entire Pareto front and performing subset selection afterwards.

We provide two main cardinality-aware approaches:

- **REHVI**: Replacement-aware Expected Hypervolume Improvement, which evaluates the improvement obtained when a newly evaluated solution can replace one of the current \(M\) selected solutions.
- **CHO-UCB**: Cardinality-aware Hypervolume Optimization with UCB, which jointly optimizes an \(M\)-point set using optimistic GP predictions and then selects a candidate for expensive evaluation.

The repository also contains implementations of several standard MOBO baselines and additional ablation methods.

---

## Repository Structure

```text
MOBO-CAPS/
│
├── run_experiment.py
├── JES_MOBO.py
├── requirements.txt
├── README.md
│
└── mobo_fixed_size/
    ├── acquisition.py
    ├── obj_function.py
    ├── utilis.py
    └── variable.py
```

The main files are:

- `run_experiment.py`: main experimental script for MOBO-CAPS methods and most baselines.
- `JES_MOBO.py`: separate implementation of the Joint Entropy Search (JES) baseline.
- `mobo_fixed_size/acquisition.py`: acquisition functions and acquisition optimization routines.
- `mobo_fixed_size/obj_function.py`: benchmark objective functions.
- `mobo_fixed_size/utilis.py`: utility functions, including GP model initialization and cardinality-constrained Pareto subset selection.
- `mobo_fixed_size/variable.py`: optimization settings such as the number of restarts, raw samples, and Monte Carlo samples.

---

## Installation

The experiments were developed with **Python 3.10.12**.

Clone the repository:

```bash
git clone https://github.com/HanyangHenry-Wang/MOBO-CAPS.git
cd MOBO-CAPS
```

We recommend creating a separate Conda environment:

```bash
conda create -n mobo-caps python=3.10
conda activate mobo-caps
```

Install the required packages:

```bash
pip install -r requirements.txt
```

The current `requirements.txt` contains:

```text
botorch==0.11.1
numpy==1.26.4
scipy==1.14.1
```

PyTorch and the corresponding dependencies required by BoTorch should also be available in the environment.

You can check the installation with:

```bash
python -c "import torch, botorch; print(torch.__version__); print(botorch.__version__)"
```

---

## Running Experiments

Most experiments are run using:

```bash
python run_experiment.py \
    --random_seed RANDOM_SEED \
    --M_type M_TYPE \
    --M_check M_CHECK \
    --type METHOD
```

The four arguments are:

| Argument | Description |
|---|---|
| `--random_seed` | Random seed for initialization |
| `--M_type` | Cardinality used by the acquisition function; use `0` for methods that do not use cardinality information |
| `--M_check` | Number of solutions retained when computing the cardinality-constrained hypervolume |
| `--type` | Optimization method |

---

## Example: CHO-UCB

To run CHO-UCB with \(M=3\) and random seed 0:

```bash
python run_experiment.py \
    --random_seed 0 \
    --M_type 3 \
    --M_check 3 \
    --type CHO-UCB
```

Here,

```text
M_type = 3
M_check = 3
```

means that CHO-UCB explicitly searches for a set of three solutions and the final performance is also evaluated using the best three solutions.

---

## Example: REHVI

To run REHVI with \(M=3\):

```bash
python run_experiment.py \
    --random_seed 0 \
    --M_type 3 \
    --M_check 3 \
    --type REHVI
```

When fewer than \(M\) suitable Pareto solutions have been observed, the implementation uses standard EHVI. Once an \(M\)-solution set is available, it switches to the replacement-aware acquisition function.

---

## Standard MOBO Baselines

Methods that do **not** explicitly use the cardinality \(M\) during acquisition optimization should use:

```text
M_type = 0
```

while `M_check` specifies how many solutions are retained for evaluation.

For example, standard EHVI evaluated using the best three solutions is run as:

```bash
python run_experiment.py \
    --random_seed 0 \
    --M_type 0 \
    --M_check 3 \
    --type EHVI
```

The distinction is:

```text
Cardinality-aware method:
M_type = M_check = M

Standard MOBO baseline:
M_type = 0
M_check = M
```

---

## Available Methods

| Method | `--type` | `M_type` |
|---|---|---:|
| Random/Sobol search | `sobol` | 0 |
| ParEGO | `ParEGO` | 0 |
| Reference-point ParEGO | `ParEGO_ref` | 0 |
| EHVI | `EHVI` | 0 |
| SMS-EGO | `SMS-EGO` | 0 |
| Hypervolume Knowledge Gradient | `HVKG` | > 0 |
| Replacement-aware EHVI | `REHVI` | > 0 |
| CHO-EI | `CHO-EI` | > 0 |
| CHO-UCB | `CHO-UCB` | > 0 |

For example, ParEGO can be run using:

```bash
python run_experiment.py \
    --random_seed 0 \
    --M_type 0 \
    --M_check 3 \
    --type ParEGO
```

Reference-point ParEGO:

```bash
python run_experiment.py \
    --random_seed 0 \
    --M_type 0 \
    --M_check 3 \
    --type ParEGO_ref
```

---

## Joint Entropy Search (JES)

The JES baseline is implemented separately in:

```text
JES_MOBO.py
```

It can be run using:

```bash
python JES_MOBO.py --random_seed 0
```

---

## Benchmark Problems

`run_experiment.py` currently contains the following benchmark problems:

| Problem | Input dimension | Objectives | BO iterations |
|---|---:|---:|---:|
| Four-Bar Truss Design | 4 | 2 | 50 |
| SnAr | 4 | 2 | 100 |
| BraninCurrin | 2 | 2 | 50 |
| DTLZ1 | 3 | 2 | 50 |
| DTLZ2 | 3 | 2 | 50 |
| DTLZ2-6D | 6 | 2 | 100 |
| DTLZ2-10D | 10 | 2 | 175 |
| VehicleSafety | 5 | 3 | 50 |
| VLMOP3 | 2 | 3 | 50 |
| VLMOP2 | 6 | 2 | 100 |
| Penicillin | 7 | 3 | 100 |

Each run of `run_experiment.py` sequentially executes the selected method on the benchmark problems included in the `problem_information` list.

---

## Initial Design

For each problem, the initial number of evaluations is

```text
N_init = min(3d, 10)
```

where `d` is the input dimensionality.

Initial designs are generated using Sobol samples.

A separate independent Gaussian process is fitted to each objective.

---

## Experimental Settings

The default acquisition optimization settings are defined in:

```text
mobo_fixed_size/variable.py
```

The current settings are:

```python
BATCH_SIZE = 1
NUM_RESTARTS = 12
RAW_SAMPLES = 256
SAMPLE_NUM = 256

OPTIONS = {
    "batch_limit": 5,
    "maxiter": 200,
    "sample_around_best": True,
}
```

The experiments therefore use sequential Bayesian optimization with one expensive function evaluation per BO iteration.

---

## Output

The cardinality-constrained hypervolume history is automatically saved under:

```text
exp/M{M_check}/
```

For example:

```bash
python run_experiment.py \
    --random_seed 0 \
    --M_type 3 \
    --M_check 3 \
    --type CHO-UCB
```

produces files with names such as:

```text
exp/M3/BraninCurrin_CHO-UCB_M=3_0
exp/M3/DTLZ2_CHO-UCB_M=3_0
exp/M3/VehicleSafety_CHO-UCB_M=3_0
```

The filename follows:

```text
{problem}_{method}_M={M_type}_{random_seed}
```

Each file stores the cardinality-constrained hypervolume obtained over the Bayesian optimization iterations.

---

## Experiments for Different Cardinalities

To run CHO-UCB for \(M \in \{1,2,3,4,5\}\) and 20 random seeds:

```bash
for M in {1..5}
do
    for seed in {0..19}
    do
        python run_experiment.py \
            --random_seed $seed \
            --M_type $M \
            --M_check $M \
            --type CHO-UCB
    done
done
```

For REHVI:

```bash
for M in {1..5}
do
    for seed in {0..19}
    do
        python run_experiment.py \
            --random_seed $seed \
            --M_type $M \
            --M_check $M \
            --type REHVI
    done
done
```

For EHVI:

```bash
for M in {1..5}
do
    for seed in {0..19}
    do
        python run_experiment.py \
            --random_seed $seed \
            --M_type 0 \
            --M_check $M \
            --type EHVI
    done
done
```

---
