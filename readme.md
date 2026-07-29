Supplementary Files for Neural Networks as Approximate Smoothers
================================================================

This repository contains the code and replicating material for the master’s thesis

**Aaron Sennhenn, “Neural Networks as Approximate Smoothers.”**

## Repository Structure

| Folder | Contents |
|:-------|:---------|
| `data/` | Input data used for empirical chapter. |
| `notebooks/` | Contains notebooks for applicationa nd empirical chapter. |
| `results/` | Stores results obtained from running the empirical evaluation. |
| `src/` | Stores adjusted OutcomeWeights functions for neural networks and code used to obtain empirical results.|


## Computational Requirements

> **Important:** The computationally intensive empirical experiments associated with `src/empirical/` were executed on the **bwUniCluster**.
> In particular, the telescoping smoother requires considerbale compuational resources. Its runtime can therefore become prohibitive on some systems.
> The neural-feature ridge approach is not demanding and can thus be executed on an ordinary personal computer.

## Author

**Aaron Sennhenn**  
Master’s Program in Data Science in Business and Economics  
Eberhard Karls University of Tübingen
