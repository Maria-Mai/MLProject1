import numpy as np
import math
import pandas as pd
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

    def learn(self, data, y_name, impurity_measure="entropy", prune=False, prune_size=0.3):
        """
        learn the decision tree with the provided data
        :param data: trainings data to learn the tree containing target
        :param y_name: name of the target
        :param impurity_measure: can be entropy/information gain or gini
        :param prune: if true the tree will be pruned, therfore the provided data will be split into train and validation data
        :param prune_size: defines the size of the validation data
        """

        # divide into train validation
        validation_data = None
        if (prune):
            data, validation_data = self.__divide_train_validation_data(data, prune_size)

        if(impurity_measure == "entropy"):
            impurity_measure_func = self.__entropy
        else:
            impurity_measure_func = self.__gini

        self.root_node = self.__learn_tree(data, y_name, impurity_measure_func)

        if(prune):
            print(data)
            self.print()
            print(validation_data)
            self.__prune_tree(validation_data, y_name, self.root_node)


    def predict(self, X):
        """
        is making predictions on the learned tree for new data without target
        :param X: new data
        :return: list with prediction of the target class
        """
        predictions = []
        for idx, row in X.iterrows():
            predictions.append(self.__predict_help(row, self.root_node))
        return predictions

    def print(self):
        """
        prints the tree
        """
        self.__print_tree(self.root_node, 0)

    def __entropy(self, data, y_name):
        """
        calculates the entropy for all classes in a dataset
        :param data: dataset
        :param y_name: target
        :return: entropy of all classes
        """
        entropy = 0.0
        unique_counts = self.__unique_counts(data, y_name)
        for idx, value in unique_counts.items():
            tmp = value/data.shape[0]
            entropy -= tmp * math.log(tmp, 2)
        return entropy

    def __IG(self, impurity, P_value, label_data, non_label_data, impurity_measure, y_name):
        """
        this method calculates the information gain of two classes
        :param impurity: overall entropy (H(DEC))
        :param P_value: proportion value
        :param label_data: data of one class
        :param non_label_data: data of the other class
        :param impurity_measure: entropy calculation function
        :param y_name: name of the target column
        :return: impurity of the label that divided the data into two sets
        """
        impurity_label = impurity - P_value * impurity_measure(label_data, y_name) - (1-P_value)*impurity_measure(non_label_data, y_name)
        return impurity_label

    def __gini(self, data, y_name):
        """
        calculates the gini coefficient of a data set
        :param data: dataset
        :param y_name: target column name
        :return: gini coefficient
        """
        gini = 0.0
        unique_counts = self.__unique_counts(data, y_name)
        for idx, value in unique_counts.items():
            tmp = value / data.shape[0]
            gini += tmp * (1 - tmp)
        return gini

    def __divide_data(self, data, column, label):
        """
        divides data into two different datasets by the label/variable
        :param data: dataset
        :param column: column of the label
        :param label: label
        :return: dataset of the label and dataset of all the left over data
        """
        if(isinstance(label, int) or isinstance(label, float)):
            label_data = data.loc[data[column] >= label]
            non_label_data = data.loc[data[column] < label]
        else:
            label_data = data.loc[data[column] == label]
            non_label_data = data.loc[data[column] != label]
        return(label_data, non_label_data)

    def __unique_counts(self, data, y_name):
        """
        creates a pivo table with the counts of each class of a dataset
        :param data: dataset
        :param y_name: target variable
        :return: pivot table with class and count
        """
        return data.pivot_table(index=[y_name], aggfunc='size')

    def __learn_tree(self, data, y_name, impurity_measure):
        """
        learns the decision tree by a chosen impurity measure
        :param data: data
        :param y_name: target column
        :param impurity_measure: can be entropy/information gain or gini coefficient
        :return: root node of the tree
        """

        best_impurity = 0.0
        best_column = None
        best_label = None
        best_data = None

        # calculate entropy H(DEC) or gini
        impurity = impurity_measure(data, y_name)

        # calculate IG(label)
        X = data.drop(labels=y_name, axis=1) # dont need last column (classes)
        for column in X:
            labels = np.unique(X[column])

            #for numbers / continous variables the algorithm will only
            #divide the data into 4 labels (quantile) to speed up the process
            if(labels.dtype == np.dtype(float).type or labels.dtype == np.dtype(int).type):
                a_quantil = np.quantile(labels, 0.25)
                labels = np.array([0, a_quantil, a_quantil*2, a_quantil*3])

            for label in labels:
                label_data, non_label_data = self.__divide_data(data, column, label)

                # if iG and entropy
                if(impurity_measure == self.__entropy):
                    P_value = label_data.shape[0] / data.shape[0]
                    IG_or_gini= self.__IG(impurity, P_value, label_data, non_label_data, impurity_measure, y_name)
                else: # if gini
                    gini_label_data = self.__gini(label_data, y_name) * (label_data.shape[0]/data.shape[0])
                    gini_non_label_data = self.__gini(non_label_data, y_name) * (non_label_data.shape[0]/data.shape[0])
                    IG_or_gini = gini_label_data + gini_non_label_data

                if(IG_or_gini > best_impurity and label_data.shape[0] > 0 and non_label_data.shape[0] > 0):
                    best_impurity = IG_or_gini
                    best_column = column
                    best_label = label
                    best_data = (label_data, non_label_data)

        if(best_impurity > 0):
            true_node = self.__learn_tree(best_data[0], y_name, impurity_measure)
            false_node = self.__learn_tree(best_data[1], y_name, impurity_measure)
            return Node(column=best_column, label=best_label, true_node=true_node, false_node=false_node)
        else:
            #get last left over value todo
            uniques = self.__unique_counts(data, y_name)
            for idx, value in uniques.items():
                return Node(uniques=idx)

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
            if (isinstance(row[node.column], int) or isinstance(row[node.column], float)):
                if (row[node.column] < node.label):
                    return self.__predict_help(row, node.false_node)
                else:
                    return self.__predict_help(row, node.true_node)
            else:
                if (row[node.column] == node.label):
                    return self.__predict_help(row, node.true_node)
                else:
                    return self.__predict_help(row, node.false_node)

    def __prune_tree(self, data, y_name, node):
        """
        prunes the tree - reduced error pruning methodd (post pruning)
        :param data: current dataset
        :param y_name: target variable
        :param node: current node
        """
        #node before leaf
        if(node.uniques == None):
            child1 = node.true_node
            child2 = node.false_node
            if (child1.uniques != None and child2.uniques != None):
                self.__check_accuracy_prune(data, y_name, node)
            else:
                #divide data
                label_data, non_label_data = self.__divide_data(data, node.column, node.label)
                self.__prune_tree(label_data, y_name, node.true_node)
                self.__prune_tree(non_label_data, y_name, node.false_node)
                self.__check_accuracy_prune(data, y_name, node)


    def __check_accuracy_prune(self,data, y_name, node):
        """
        checks the accuracy for the pruning
        :param data: current data set
        :param y_name: traget variable
        :param node: current node
        """
        X = data.drop(labels=y_name, axis=1)
        y = data[[y_name]]

        #todo can be empty
        if(data.shape[0] > 0):

            predictions = []
            for idx, row in X.iterrows():
                predictions.append(self.__predict_help(row, node))
            predictions = pd.DataFrame({"pred" : predictions})
            differences = np.sum([predictions.values == y.values])
            accuracy_full_tree = differences / data.shape[0]

            unique_counts = data.pivot_table(index=[y_name], aggfunc='size')
            highest_tuple = (0, None)
            #idx: yes/no value:count
            for idx, value in unique_counts.items():
                if(value > highest_tuple[0]):
                    highest_tuple = (value, idx)
            accuracy_maternity = highest_tuple[0] / data.shape[0]

            if (accuracy_full_tree <= accuracy_maternity):
                node.true_node = None
                node.false_node = None
                node.label = None
                node.column = None
                node.uniques = highest_tuple[1]

    def __divide_train_validation_data(self, data, prune_size):
        """
        divides the data into a train and validation set for the pruning
        :param data: dataset
        :param prune_size: size of the validation set
        :return: training data, validation data
        """
        val_data = data.sample(frac=prune_size)
        train_data = data.drop(val_data.index)
        return train_data, val_data
