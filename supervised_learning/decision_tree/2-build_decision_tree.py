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
