class Node(object):
    """
    Node object for the decision tree
    """
    def __init__(self, column=None, label=None, uniques=None, true_node=None, false_node=None, X_train=None, y_train=None):
        """
        initializes all new nodes
        :param column: column of the node
        :param label: label of the edge and opposit of the label for the other edge
        :param uniques: for leaf nodes to set the class
        :param true_node: child for the labeld data
        :param false_node: child for the non labeled data
        :param X_train: training data
        :param y_train: training target
        """
        self.column = column
        self.label = label
        self.uniques = uniques
        self.true_node = true_node
        self.false_node = false_node
        self.X_train = X_train
        self.y_train = y_train
