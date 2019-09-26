import numpy as np
import math
from Node import Node

class DecisionTree(object):
    """
    this class implements a decision tree with the ID3 algorithm
    it can learn, predict and print the tree
    """

    def __init__(self):
        """
        initalize tree object
        """
        self.root_note = None

    def learn(self, X_train, y_train, impurity_measure="entropy", prune=False, prune_size=0.3):
        """
        learn the decision tree with the provided data
        :param X_train: trainings data to learn the tree
        :param y_train: target vector
        :param impurity_measure: can be entropy/information gain or gini
        :param prune: if true the tree will be pruned, therfore the provided data will be split into train and validation data
        :param prune_size: defines the size of the validation data
        """

        # divide into train validation
        validation_data = None
        if (prune):
            X_train, y_train, X_val, y_val = self.__divide_train_validation_data(X_train, y_train, prune_size)

        if(impurity_measure == "entropy"):
            impurity_measure_func = self.__entropy
        else:
            impurity_measure_func = self.__gini

        self.root_node = self.__learn_tree(X_train, y_train, impurity_measure_func)

        if(prune):
            self.__prune_tree(X_val, y_val, self.root_node)


    def predict(self, data):
        """
        is making predictions on the learned tree for new data without target
        :param data: new data
        :return: list with prediction of the target class
        """
        predictions = []
        for row in data:
            predictions.append(self.__predict_help(row, self.root_node))
        return predictions

    def print(self):
        """
        prints the tree
        """
        self.__print_tree(self.root_node, 0)

    def __entropy(self, X_train, y_train):
        """
        calculates the entropy for all classes in a dataset
        :param data: dataset
        :param y_name: target
        :return: entropy of all classes
        """
        entropy = 0.0
        unique_counts = self.__unique_counts(y_train)
        for key in unique_counts:
            tmp = unique_counts[key]/X_train.shape[0]
            entropy -= tmp * math.log(tmp, 2)
        return entropy

    def __IG(self, impurity, P_value, X_label_train, y_label_train, X_non_label_train, y_non_label_train):
        """
        this method calculates the information gain of two classes
        :param impurity: overall entropy (H(DEC))
        :param P_value: proportion value
        :param X_label_train: data of one class
        :param y_label_train: target of one class
        :param X_non_label_train: data of the other class
        :param y_non_label_train: target of the other class
        :param impurity_measure: impurity measure (entropy/IG or gini)
        :return: impurity of the label that divided the data into two sets
        """
        impurity_label = impurity - P_value * self.__entropy(X_label_train, y_label_train) - \
                         (1-P_value)*self.__entropy(X_non_label_train, y_non_label_train)
        return impurity_label

    def __gini(self, X_train, y_train):
        """
        calculates the gini index of a data set
        :param X_train: data
        :param y_train: target
        :return: gini index
        """
        gini = 1.0
        unique_counts = self.__unique_counts(y_train)
        for key in unique_counts:
            tmp = unique_counts[key] / X_train.shape[0]
            gini -= tmp**2
        return gini

    def __divide_data(self, X_train, y_train, column_number, label):
        """
        divides data into two different datasets by the label/variable
        :param X_train: training data
        :param y_train: target data
        :param column_number: column of the label
        :param label: label
        :return: X_train, y_train of the label and X_train, y_train of all the left over data
        """

        current_column = X_train[:,column_number]
        X_label_train = X_train[current_column >= label, :]
        y_label_train = y_train[current_column >= label]

        X_non_label_train = X_train[current_column < label, :]
        y_non_label_train = y_train[current_column < label]

        return X_label_train, y_label_train, X_non_label_train, y_non_label_train


    def __unique_counts(self, y_train):
        """
        gets the count for every class
        :param y_train: target
        :return: dictionary with class and count
        """
        unique, counts = np.unique(y_train, return_counts=True)
        return dict(zip(unique, counts))

    def learn(self, X_train, y_train, impurity_measure="entropy", prune=False, prune_size=0.3):
        """
        learns the decision tree
        :param X_train: training data
        :param y_train: training target
        :param impurity_measure: impurity measure (entropy or gini)
        :param prune: if the tree should be pruned
        :param prune_size: size of the pruning set
        """

        # divide into train validation
        validation_data = None
        if (prune):
            X_train, y_train, X_val, y_val = self.__divide_train_validation_data(X_train, y_train, prune_size)

        if (impurity_measure == "entropy"):
            impurity_measure_func = self.__entropy
        else:
            impurity_measure_func = self.__gini

        self.root_node = self.__best_split(Node(X_train=X_train, y_train=y_train), impurity_measure_func)
        self.__learn_tree(self.root_node, impurity_measure_func)

        if (prune):
            self.__prune_tree(X_val, y_val, self.root_node)

    def __best_split(self, node, impurity_measure):
        """
        updates node after finding the best split
        :param node: current node
        :param impurity_measure: impurity measure (entropy or gini)
        :return: updated current node
        """

        if(impurity_measure == self.__entropy):
            column, label, true_node, false_node = self.__best_split_entropy(node)
        else:
            column, label, true_node, false_node = self.__best_split_gini(node)

        node.column = column
        node.label=label
        node.true_node=true_node
        node.false_node=false_node

        return node

    def __best_split_gini(self, node):
        """
        calculates the gini index and finds the best split
        :param node: current node
        :return: best values for the node
        """

        best_impurity = 999
        best_column = None
        best_label = None
        best_X_train_label = None
        best_y_train_label = None
        best_y_train_non_label = None
        best_X_train_non_label = None

        X_train = node.X_train
        y_train = node.y_train

        column_number = 0
        # calculate smallest gini
        for column in X_train.T:
            # split: divide the data into 3 labels (quantile) to speed up the process
            labels = np.quantile(column, [0.25, 0.5, 0.75])
            #label = np.mean(column)

            for label in labels:
                X_label_train, y_label_train, X_non_label_train, y_non_label_train = self.__divide_data(X_train,y_train,column_number,label)

                gini_label_data = self.__gini(X_label_train, y_label_train) * (X_label_train.shape[0] / X_train.shape[0])
                gini_non_label_data = self.__gini(X_non_label_train, y_non_label_train) * (X_non_label_train.shape[0] / X_train.shape[0])
                gini = gini_label_data + gini_non_label_data

                if (gini < best_impurity):
                    best_impurity = gini
                    best_column = column_number
                    best_label = label
                    best_X_train_label = X_label_train
                    best_y_train_label = y_label_train
                    best_y_train_non_label = y_non_label_train
                    best_X_train_non_label = X_non_label_train

            column_number += 1

        if(best_X_train_non_label.shape[0] == 0):
            best_X_train_non_label = None

        if (best_X_train_label.shape[0] == 0):
            best_X_train_label = None

        return best_column, best_label, Node(X_train=best_X_train_label, y_train=best_y_train_label), Node(X_train=best_X_train_non_label, y_train=best_y_train_non_label)

    def __best_split_entropy(self, node):
        """
        calculates the information gain and finds the best split
        :param node: current node
        :return: best values for the node
        """

        best_impurity = 0
        best_column = None
        best_label = None
        best_X_train_label = None
        best_y_train_label = None
        best_y_train_non_label = None
        best_X_train_non_label = None

        X_train = node.X_train
        y_train = node.y_train
        impurity = self.__entropy(X_train, y_train)

        column_number = 0
        # calculate IG
        for column in X_train.T:
            # split: divide the data into 3 labels (quantile) to speed up the process
            labels = np.quantile(column, [0.25, 0.5, 0.75])

            for label in labels:
                X_label_train, y_label_train, X_non_label_train, y_non_label_train = self.__divide_data(X_train, y_train, column_number, label)

                P_value = X_label_train.shape[0] / X_train.shape[0]
                IG = self.__IG(impurity, P_value, X_label_train, y_label_train, X_non_label_train, y_non_label_train)

                if(IG > best_impurity):
                    best_impurity = IG
                    best_column = column_number
                    best_label = label
                    best_X_train_label = X_label_train
                    best_y_train_label = y_label_train
                    best_y_train_non_label = y_non_label_train
                    best_X_train_non_label = X_non_label_train

            column_number +=1

        return best_column, best_label, Node(X_train=best_X_train_label, y_train=best_y_train_label), Node(X_train=best_X_train_non_label, y_train=best_y_train_non_label)

    def __learn_tree(self, node, impurity_measure):
        """
        learns the decision tree recursive and starts with the root node
        :param node:  current node
        :param impurity_measure: impurity measure
        """

        true_node = node.true_node
        false_node = node.false_node
        X_train_true = true_node.X_train
        X_train_false = false_node.X_train

        #cant split anymore, then make leaf
        if not isinstance(X_train_true, np.ndarray) or not isinstance(X_train_false, np.ndarray):
            uniques = self.__unique_counts(node.y_train)
            for key in uniques:
                node.uniques = key
                true_node = None
                false_node = None
                return
        else:
            self.__best_split(true_node, impurity_measure)
            self.__best_split(false_node, impurity_measure)

            self.__learn_tree(true_node, impurity_measure)
            self.__learn_tree(false_node, impurity_measure)

    def __print_tree(self,root_node, ident):
        """
        helps to print the tree recursive
        :param root_node: root node of the current tree
        :param ident: ident for the tree levels
        """
        # leaf
        if (root_node.uniques != None):
            print(ident * "\t" + str(root_node.uniques))
        else:
            print(ident * "\t" + "Column: " + str(root_node.column) + " " + "Label: " + str(root_node.label) + "\n")
            self.__print_tree(root_node.true_node, ident +1)
            self.__print_tree(root_node.false_node, ident + 1)


    def __predict_help(self, row, node):
        """
        predicts a class for a row
        :param row: row of the (unseen) data
        :param node: current node
        :return: a class for the row
        """
        if(node.uniques != None):
            return node.uniques
        else:
            if (row[node.column] < node.label):
                return self.__predict_help(row, node.false_node)
            else:
                return self.__predict_help(row, node.true_node)

    def __prune_tree(self, X_val, y_val, node):
        """
        prunes the tree - reduced error pruning method (post pruning)
        :param X_val: current data
        :param y_val: current target data
        :param node: current node
        """

        #node before leaf
        if(node.uniques == None):
            child1 = node.true_node
            child2 = node.false_node
            if (child1.uniques != None and child2.uniques != None):
                self.__check_accuracy_prune(X_val, y_val, node)
            else:
                #divide data
                X_label_train, y_label_train, X_non_label_train, y_non_label_train = self.__divide_data(X_val, y_val, node.column, node.label)
                self.__prune_tree(X_label_train, y_label_train, node.true_node)
                self.__prune_tree(X_non_label_train, y_non_label_train, node.false_node)
                self.__check_accuracy_prune(X_val, y_val, node)


    def __check_accuracy_prune(self, X_val, y_val, node):
        """
        checks the accuracy for the pruning
        :param X_val: current data
        :param y_val: current target data
        :param node: current node
        """
        if(X_val.shape[0] > 0):

            predictions = []
            for row in X_val:
                predictions.append(self.__predict_help(row, node))
            differences = np.sum([predictions == y_val])
            accuracy_full_tree = differences / X_val.shape[0]

            unique, counts = np.unique(y_val, return_counts=True)
            highest_idx = np.argmax(counts)
            highest_tuple = (counts[highest_idx], unique[highest_idx])
            accuracy_maternity = highest_tuple[0] / X_val.shape[0]

            if (accuracy_full_tree <= accuracy_maternity):
                node.true_node = None
                node.false_node = None
                node.label = None
                node.column = None
                node.uniques = highest_tuple[1]

    def __divide_train_validation_data(self, X_train, y_train, prune_size):
        """
        divides the data into a train and validation set for the pruning
        :param X_train: training data
        :param y_train: training target
        :param prune_size: pruning size
        :return: training data, training target, validation data, validation target
        """
        len_train = len(X_train)
        idx = np.random.randint(len_train, size=int(len_train*prune_size))
        X_val = X_train[idx,:]
        y_val = y_train[idx]
        X_train = np.delete(X_train, idx, 0)
        y_train = np.delete(y_train, idx, 0)

        return X_train, y_train, X_val, y_val