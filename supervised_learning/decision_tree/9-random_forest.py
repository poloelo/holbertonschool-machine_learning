#!/usr/bin/env python3
"""Random forest built on top of randomly grown decision trees."""
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """A forest of decision trees voting on the class of an individual."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initialize an empty forest."""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """Return the class voted by the majority of the trees.

        Args:
            explanatory: A 2D array of shape (n_individuals, n_features).

        Returns:
            numpy.ndarray: The most frequent prediction for each
            individual.
        """
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return np.array([np.argmax(np.bincount(predictions[:, i]))
                         for i in range(predictions.shape[1])])

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """Grow the trees of the forest on a training set.

        Args:
            explanatory: A 2D array of shape (n_individuals, n_features).
            target: A 1D array of size n_individuals.
            n_trees: The number of trees to grow.
            verbose: If 1, print a summary of the training.
        """
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth,
                              min_pop=self.min_pop, seed=self.seed + i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}
    - Accuracy of the forest on td   : {self.accuracy(explanatory, target)}""")

    def accuracy(self, test_explanatory, test_target):
        """Return the proportion of individuals correctly classified.

        Args:
            test_explanatory: A 2D array of explanatory features.
            test_target: The 1D array of expected values.

        Returns:
            float: The accuracy of the forest on that set.
        """
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size
