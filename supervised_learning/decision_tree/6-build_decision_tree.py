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
