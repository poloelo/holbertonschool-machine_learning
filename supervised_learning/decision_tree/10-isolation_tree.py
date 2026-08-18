#!/usr/bin/env python3
"""Isolation random tree, used to detect outliers."""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """A randomly grown tree whose leaves return their own depth.

    An individual that ends up alone in a shallow leaf is easy to isolate
    from the rest of the population, and is therefore a likely outlier.
    """

    def __init__(self, max_depth=10, seed=0, root=None):
        """Initialize the tree and the random generator used to grow it."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """Return a printable representation of the whole tree."""
        return self.root.__str__()

    def depth(self):
        """Return the depth of the tree.

        Returns:
            int: The maximum depth among all the nodes of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count the nodes of the tree.

        Args:
            only_leaves: If True, only the leaves are counted.

        Returns:
            int: The number of nodes, or of leaves, of the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """Compute the lower and upper bounds of every node of the tree."""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Return the list of every leaf of the tree.

        Returns:
            list: The leaves of the tree, left to right.
        """
        return self.root.get_leaves_below()

    def update_predict(self):
        """Build the vectorized prediction function of the tree.

        The prediction of an individual is the depth of the leaf it falls
        into.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            np.array([leaf.indicator(A) * leaf.value for leaf in leaves]),
            axis=0)

    def np_extrema(self, arr):
        """Return the minimum and the maximum of an array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Draw a random feature and a random threshold for a node.

        Args:
            node: The node to split.

        Returns:
            tuple: The chosen feature index and threshold.
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Build the leaf that a node hands a sub population to.

        Args:
            node: The parent node.
            sub_population: The booleans selecting the individuals.

        Returns:
            Leaf: A leaf whose value is its own depth.
        """
        leaf_child = Leaf(node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Build the internal node that a node hands a sub population to.

        Args:
            node: The parent node.
            sub_population: The booleans selecting the individuals.

        Returns:
            Node: A node one level below the parent.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Split a node recursively until its children are leaves.

        Since there is no target, the splitting only stops on the depth
        of the node or on the size of its population.

        Args:
            node: The node to grow.
        """
        node.feature, node.threshold = self.random_split_criterion(node)

        goes_left = self.explanatory[:, node.feature] > node.threshold
        left_population = np.logical_and(node.sub_population, goes_left)
        right_population = np.logical_and(node.sub_population,
                                          np.logical_not(goes_left))

        # Is left node a leaf ?
        is_left_leaf = (node.depth + 1 >= self.max_depth or
                        np.sum(left_population) <= self.min_pop)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = (node.depth + 1 >= self.max_depth or
                         np.sum(right_population) <= self.min_pop)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """Grow the tree on a set of individuals.

        Args:
            explanatory: A 2D array of shape (n_individuals, n_features).
            verbose: If 1, print a summary of the training.
        """
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
