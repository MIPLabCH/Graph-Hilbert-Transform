## Graph Hilbert Transform

This is the repository gathering the experiments for the paper ``Hilbert Transform on Graphs: Let There Be Phase" (10.1109/LSP.2025.3560170). All figures found can be re-generated from this repository.

### Installation

Run the command:
`pip install GyRAPH`

### Run Experiments
```
# List all available experiments
python -m experiments.run --list

# Run a specific experiment
python -m experiments.run --paper paper_graph_hilbert_transform --experiment exp_flower-graph
```

In each experiment folders are located config files you can change to modify the input parameters to experiments.

#### Contents

- Figures from paper ``Hilbert Transform on Graphs: Let There Be Phase"

### Core Module

This repository hosts the core [GyRAPH](https://github.com/miki998/GyRAPH/) implementation powering the Hilbert Transform on Graphs experiments and all reproducible figures from the paper.