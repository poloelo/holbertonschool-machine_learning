#!/usr/bin/env python3
"""Isolation random forest, used to detect outliers."""
import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest():
    """A forest of isolation random trees.

    The prediction of the forest is the mean depth of the leaves an
    individual falls into, so the smallest predictions point at the
    individuals that are the easiest to isolate.
    """

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initialize an empty forest."""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """Return the mean depth predicted by the trees of the forest.

        Args:
            explanatory: A 2D array of shape (n_individuals, n_features).

        Returns:
            numpy.ndarray: The mean depth of each individual.
        """
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """Grow the trees of the forest.

        Args:
            explanatory: A 2D array of shape (n_individuals, n_features).
            n_trees: The number of trees to grow.
            verbose: If 1, print a summary of the training.
        """
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            T = Isolation_Random_Tree(max_depth=self.max_depth,
                                      seed=self.seed + i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}""")

    def suspects(self, explanatory, n_suspects):
        """Return the individuals that have the smallest mean depth.

        Args:
            explanatory: A 2D array of shape (n_individuals, n_features).
            n_suspects: The number of individuals to return.

        Returns:
            tuple: The rows of ``explanatory`` that are the most likely
            outliers, and their mean depths.
        """
        depths = self.predict(explanatory)
        order = np.argsort(depths)[:n_suspects]
        return explanatory[order], depths[order]
