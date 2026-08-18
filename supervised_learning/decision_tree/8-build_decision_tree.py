#!/usr/bin/env python3
"""Building blocks of a decision tree: nodes, leaves and the tree."""
import numpy as np


class Node:
    """An internal node of a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize a node with its split and its two children."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Return the maximum depth among the nodes below this one.

        Returns:
            int: The depth of the deepest node of the subtree rooted here.
        """
        if self.is_leaf:
            return self.depth
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """Count the nodes of the subtree rooted at this node.

        Args:
            only_leaves: If True, only the leaves are counted.

        Returns:
            int: The number of nodes, or of leaves, below and including
            this node.
        """
        below = (self.left_child.count_nodes_below(only_leaves=only_leaves) +
                 self.right_child.count_nodes_below(only_leaves=only_leaves))
        if only_leaves:
            return below
        return below + 1

    def left_child_add_prefix(self, text):
        """Indent the string of the left child under this node."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Indent the string of the right child under this node."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """Return a printable representation of the subtree."""
        if self.is_root:
            head = "root "
        else:
            head = "-> node "
        head += "[feature={}, threshold={}]".format(self.feature,
                                                    self.threshold)
        left = self.left_child_add_prefix(
            self.left_child.__str__().rstrip("\n"))
        right = self.right_child_add_prefix(
            self.right_child.__str__().rstrip("\n"))
        return head + "\n" + left + right

    def get_leaves_below(self):
        """Return the list of every leaf of the subtree.

        Returns:
            list: The leaves found below this node, left to right.
        """
        return (self.left_child.get_leaves_below() +
                self.right_child.get_leaves_below())

    def update_bounds_below(self):
        """Compute recursively the lower and upper bounds of each node.

        The bounds are stored as the dictionaries ``Node.lower`` and
        ``Node.upper``, whose keys are the features constraining the node.
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()

        self.left_child.lower[self.feature] = max(
            self.threshold,
            self.left_child.lower.get(self.feature, -np.inf))
        self.right_child.upper[self.feature] = min(
            self.threshold,
            self.right_child.upper.get(self.feature, np.inf))

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Build the indicator function of the node.

        The indicator takes a 2D array of shape (n_individuals,
        n_features) and returns a 1D array of booleans telling, for each
        individual, whether it satisfies the bounds of the node.
        """
        def is_large_enough(x):
            """Test the individuals against the lower bounds."""
            return np.all(np.array([np.greater(x[:, key], self.lower[key])
                                    for key in self.lower.keys()]), axis=0)

        def is_small_enough(x):
            """Test the individuals against the upper bounds."""
            return np.all(np.array([np.less_equal(x[:, key], self.upper[key])
                                    for key in self.upper.keys()]), axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """Return the prediction of the subtree for one individual.

        Args:
            x: A 1D array holding the features of a single individual.

        Returns:
            The value predicted by the leaf the individual falls into.
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """A leaf of a decision tree, holding a predicted value."""

    def __init__(self, value, depth=None):
        """Initialize a leaf with the value it predicts."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return the depth of this leaf.

        Returns:
            int: The depth of the leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Count this leaf.

        Args:
            only_leaves: Unused, a leaf counts for one either way.

        Returns:
            int: Always 1.
        """
        return 1

    def __str__(self):
        """Return a printable representation of the leaf."""
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """Return this leaf in a list.

        Returns:
            list: A list holding this single leaf.
        """
        return [self]

    def update_bounds_below(self):
        """Stop the recursive computation of the bounds at the leaves."""
        pass

    def pred(self, x):
        """Return the value of the leaf for one individual.

        Args:
            x: A 1D array holding the features of a single individual.

        Returns:
            The value stored in the leaf.
        """
        return self.value


class Decision_Tree():
    """A binary decision tree."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialize the tree and the random generator used to grow it."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

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

    def __str__(self):
        """Return a printable representation of the whole tree."""
        return self.root.__str__()

    def get_leaves(self):
        """Return the list of every leaf of the tree.

        Returns:
            list: The leaves of the tree, left to right.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Compute the lower and upper bounds of every node of the tree."""
        self.root.update_bounds_below()

    def update_predict(self):
        """Build the vectorized prediction function of the tree.

        The individuals are dispatched to the leaves with the indicator
        functions, so that a whole array is predicted at once.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            np.array([leaf.indicator(A) * leaf.value for leaf in leaves]),
            axis=0)

    def pred(self, x):
        """Return the prediction of the tree for one individual.

        Args:
            x: A 1D array holding the features of a single individual.

        Returns:
            The value predicted for that individual.
        """
        return self.root.pred(x)

    def fit(self, explanatory, target, verbose=0):
        """Grow the tree on a training set.

        Args:
            explanatory: A 2D array of shape (n_individuals, n_features).
            target: A 1D array of size n_individuals.
            verbose: If 1, print a summary of the training.
        """
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {self.accuracy(self.explanatory,
                                                 self.target)}""")

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

    def possible_thresholds(self, node, feature):
        """Return the midpoints between the values taken by a feature."""
        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """Find the best threshold on one feature for the Gini criterion.

        Args:
            node: The node to split.
            feature: The index of the feature to test.

        Returns:
            numpy.ndarray: The best threshold and the average Gini
            impurity it achieves.
        """
        thresholds = self.possible_thresholds(node, feature)
        if thresholds.size == 0:
            return np.array([0, np.inf])

        values = (self.explanatory[:, feature])[node.sub_population]
        classes = self.target[node.sub_population]
        distinct = np.unique(classes)

        # (n, t): individual i goes left for threshold j
        left_f = values[:, None] > thresholds[None, :]
        # (n, c): individual i belongs to class k
        class_f = classes[:, None] == distinct[None, :]

        # (t, c): number of individuals of each class in each child
        left_counts = np.sum(left_f[:, :, None] & class_f[:, None, :], axis=0)
        right_counts = np.sum(
            np.logical_not(left_f)[:, :, None] & class_f[:, None, :], axis=0)

        left_sizes = np.sum(left_counts, axis=1)
        right_sizes = np.sum(right_counts, axis=1)

        gini_left = 1 - np.sum(
            (left_counts / left_sizes[:, None]) ** 2, axis=1)
        gini_right = 1 - np.sum(
            (right_counts / right_sizes[:, None]) ** 2, axis=1)

        gini_sum = (left_sizes * gini_left + right_sizes * gini_right)
        gini_average = gini_sum / (left_sizes + right_sizes)

        best = np.argmin(gini_average)
        return np.array([thresholds[best], gini_average[best]])

    def Gini_split_criterion(self, node):
        """Find the split of a node that minimizes the Gini impurity.

        Args:
            node: The node to split.

        Returns:
            tuple: The chosen feature index and threshold.
        """
        X = np.array([self.Gini_split_criterion_one_feature(node, i)
                      for i in range(self.explanatory.shape[1])])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]

    def fit_node(self, node):
        """Split a node recursively until its children are leaves.

        Args:
            node: The node to grow.
        """
        node.feature, node.threshold = self.split_criterion(node)

        goes_left = self.explanatory[:, node.feature] > node.threshold
        left_population = np.logical_and(node.sub_population, goes_left)
        right_population = np.logical_and(node.sub_population,
                                          np.logical_not(goes_left))

        # Is left node a leaf ?
        is_left_leaf = (node.depth + 1 >= self.max_depth or
                        np.sum(left_population) <= self.min_pop or
                        np.unique(self.target[left_population]).size == 1)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = (node.depth + 1 >= self.max_depth or
                         np.sum(right_population) <= self.min_pop or
                         np.unique(self.target[right_population]).size == 1)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Build the leaf that a node hands a sub population to.

        Args:
            node: The parent node.
            sub_population: The booleans selecting the individuals.

        Returns:
            Leaf: A leaf predicting the most represented class.
        """
        value = np.argmax(np.bincount(self.target[sub_population]))
        leaf_child = Leaf(value)
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

    def accuracy(self, test_explanatory, test_target):
        """Return the proportion of individuals correctly classified.

        Args:
            test_explanatory: A 2D array of explanatory features.
            test_target: The 1D array of expected values.

        Returns:
            float: The accuracy of the tree on that set.
        """
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size
