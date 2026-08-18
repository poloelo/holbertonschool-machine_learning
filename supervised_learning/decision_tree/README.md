# Decision Tree & Random Forest

## Description

This project implements decision trees from scratch, with numpy only. It
starts with the structure of the tree (depth, number of nodes, printing,
listing the leaves), then builds an efficient vectorized `predict` method
based on the bounds satisfied by each leaf, then makes the tree trainable
with a random splitting criterion and with the Gini impurity criterion.
The last tasks assemble the trees into a random forest, and adapt them
into an isolation forest to detect outliers.

## Requirements

- Ubuntu 20.04 LTS
- Python 3.9
- NumPy 1.25.2
- pycodestyle 2.11.1
- All files must be executable and end with a new line
- The first line of all files must be exactly `#!/usr/bin/env python3`
- All modules, classes and functions must have a documentation string

## Files

| File | Description |
| --- | --- |
| `0-build_decision_tree.py` | `Node.max_depth_below`: depth of the tree |
| `1-build_decision_tree.py` | `Node.count_nodes_below`: number of nodes and leaves |
| `2-build_decision_tree.py` | `Node.__str__`: printing the tree |
| `3-build_decision_tree.py` | `Node.get_leaves_below`: listing the leaves |
| `4-build_decision_tree.py` | `Node.update_bounds_below`: bounds of each node |
| `5-build_decision_tree.py` | `Node.update_indicator`: indicator function of a node |
| `6-build_decision_tree.py` | `Decision_Tree.update_predict`: vectorized prediction |
| `7-build_decision_tree.py` | `Decision_Tree.fit`: training with random splits |
| `8-build_decision_tree.py` | `Gini_split_criterion`: training with Gini impurity |
| `9-random_forest.py` | `Random_Forest`: majority vote over many trees |
| `10-isolation_tree.py` | `Isolation_Random_Tree`: leaves return their depth |
| `11-isolation_forest.py` | `Isolation_Random_Forest`: mean depth and suspects |

## Author

Paul Gioria
