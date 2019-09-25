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
            print(X_val, y_val)
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

    def __IG(self, impurity, P_value, X_label_train, y_label_train, X_non_label_train, y_non_label_train, impurity_measure):
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
        impurity_label = impurity - P_value * impurity_measure(X_label_train, y_label_train) - \
                         (1-P_value)*impurity_measure(X_non_label_train, y_non_label_train)
        return impurity_label

    def __gini(self, X_train, y_train):
        """
        calculates the gini coefficient of a data set
        :param X_train: data
        :param y_train: target
        :return: gini coefficient
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

    def __learn_tree(self, X_train, y_train, impurity_measure):
        """
        learns the decision tree by a chosen impurity measure
        :param X_train: training data
        :param y_train: training target
        :param impurity_measure: can be entropy/information gain or gini coefficient
        :return: root node of the tree
        """

        best_impurity = 0.0
        best_column = None
        best_label = None
        best_X_train_label = None
        best_y_train_label = None
        best_y_train_non_label = None
        best_X_train_non_label = None

        # calculate entropy H(DEC)
        if (impurity_measure == self.__entropy):
            impurity = impurity_measure(X_train, y_train)

        print("neue runde")

        column_number = 0
        # calculate IG
        for column in X_train.T:
            #split: divide the data into 3 labels (quantile) to speed up the process
            labels = np.quantile(column, [0.25, 0.5, 0.75])

            for label in labels:
                X_label_train, y_label_train, X_non_label_train, y_non_label_train = self.__divide_data(X_train, y_train, column_number, label)
                # if iG and entropy
                if(impurity_measure == self.__entropy):
                    P_value = X_label_train.shape[0] / X_train.shape[0]
                    IG_or_gini= self.__IG(impurity, P_value, X_label_train, y_label_train, X_non_label_train, y_non_label_train, impurity_measure)
                else: # if gini todo mehr als ein unique!!!!
                    gini_label_data = self.__gini(X_label_train, y_label_train) * (X_label_train.shape[0]/X_train.shape[0])
                    gini_non_label_data = self.__gini(X_non_label_train, y_non_label_train) * (X_non_label_train.shape[0]/X_train.shape[0])
                    print("gini label: " + str(gini_label_data) + " gini non label: " +str(gini_non_label_data) + " laenge: " + str(X_label_train.shape[0]) + "," + str(X_non_label_train.shape[0]))
                    IG_or_gini = gini_label_data + gini_non_label_data
                    print("gini total: " +str(IG_or_gini))

                if(IG_or_gini > best_impurity and X_label_train.shape[0] > 0 and X_non_label_train.shape[0] > 0): #todo das problem liegt hier
                   #todo auslagern kleiner vs groeßer
                    best_impurity = IG_or_gini
                    best_column = column_number
                    best_label = label
                    best_X_train_label = X_label_train
                    best_y_train_label = y_label_train
                    best_y_train_non_label = y_non_label_train
                    best_X_train_non_label = X_non_label_train
            column_number += 1

        print("best impu: " + str(best_impurity))
        if(best_impurity > 0):
            true_node = self.__learn_tree(best_X_train_label, best_y_train_label, impurity_measure)
            false_node = self.__learn_tree(best_X_train_non_label, best_y_train_non_label, impurity_measure)
            return Node(column=best_column, label=best_label, true_node=true_node, false_node=false_node)
        else:
            #get last left over value
            uniques = self.__unique_counts(y_train)
            print("uniques in else: " + str(uniques))
            for key in uniques:
                return Node(uniques=key)

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
