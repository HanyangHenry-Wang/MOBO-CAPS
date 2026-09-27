# MOBO-CAPS: Multi-objective Bayesian Optimization with Cardinality-Aware Pareto Selection 
[![arXiv](https://img.shields.io/badge/📄_arXiv-2501.18792-b31b1b.svg)](https://arxiv.org/abs/2501.18792)
[![NeurIPS](https://img.shields.io/badge/🎓_NeurIPS-2025-blue.svg)](https://neurips.cc/Conferences/2025)
## Introduction 
Multi-objective Bayesian optimization (MOBO) traditionally aims to approximate the entire Pareto front, but real-world deployment is often limited to a small, representative subset of solutions due to resource constraints. 
Current two-stage methods first approximate the full front before selecting solutions, wasting evaluation budget. We formalize this problem as Multi-objective Bayesian Optimization with Cardinality-aware Pareto Selection (MOBO-CAPS), where the goal is to identify a set of exactly $M$ solutions with high hypervolume. We propose two practical cardinality-aware acquisition strategies: REHVI, a replacement-aware approximation to expected hypervolume improvement, and CHO-UCB, which constructs an optimistic $M$-point set by optimizing a collective UCB-based hypervolume objective. We prove a sublinear simple regret bound for the batched version of CHO-UCB, and empirically show that cardinality-aware acquisition design improves over standard MOBO followed by post-hoc subset selection across nine benchmarks.
## Installation
Our python version is 3.10.12.
```bash
git clone https://github.com/HanyangHenry-Wang/BOPE-MoNNE.git && cd BOPE-MoNNE
pip install -r requirements.txt --upgrade
pip install -e .
```

## Usage
Run the experiment with a random seed:
```bash
python experiment_NN.py 123
```
where `123` is the random seed.
## Configuration

### Test Problem Selection
Select the test problem by commenting/uncommenting the relevant lines in the `.py` file.


