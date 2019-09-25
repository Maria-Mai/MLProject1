from DecisionTreeNew import DecisionTree
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn import tree
import numpy as np

#-----------------------proprocess data-----------------------------
file = open("abalone.data", "r")

data_for_sklearn = []
for line in file:
    line_array = line.rstrip().split(",")
    changed_type_line_sklearn= []
    for i in range(len(line_array)):
        if(i == 0):
            if(line_array[0] == "M"):
                changed_type_line_sklearn.append(0)
            elif(line_array[0] == "F"):
                changed_type_line_sklearn.append(1)
            else:
                changed_type_line_sklearn.append(2)
        elif(i >= 1 and i <= 7):
            changed_type_line_sklearn.append(float(line_array[i]))
        else:
            changed_type_line_sklearn.append(float(line_array[i]))
    data_for_sklearn.append(changed_type_line_sklearn)
data_sklearn = np.asarray(data_for_sklearn)
X = data_sklearn[:,:-1]
y = data_sklearn[:,-1]

print(len(X))
print(len(y))

#----------------preprocces data end-------------------------------


# ---------------divide data for 5 fold cross validation -------------------
#train test split, oly use train for cross validation
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.33)
folds = 5
k_fold = KFold(n_splits=folds)
# --------------end


#------------cross validate the data 10 times (gini, gini + prune 0.1, gini + prune 0.2 gini + prune 0.3, gini + prune 0.4,
#                                               entropy, entropy + prune 0.1, 0.2, 0.3 0.4)----------------
method_list = [("entropy", False, 0), ("entropy", True, 0.1), ("entropy", True, 0.2), ("entropy", True, 0.3), ("entropy", True, 0.4),
               ("gini", False, 0), ("gini", True, 0.1), ("gini", True, 0.2), ("gini", True, 0.3), ("gini", True, 0.4)]

accuracy_list = list()

for method in method_list:
    val_accuracy = list()
    for train_index, val_index in k_fold.split(X_train):
        a_tree = DecisionTree()

        X_train[val_index]
        y_train[train_index]
        a_tree.learn(X_train[train_index], y_train[train_index], method=method[0], prune=method[1], prune_size=method[3])

        predictions = a_tree.predict()
        val_accuracy.append(accuracy_score(y_train[val_index], predictions))
    cross_accuracy = sum(val_accuracy)/folds
    accuracy_list.append((method, cross_accuracy))

print(accuracy_list)
best_method = np.argmax(accuracy_list) #todo weiß nicht was
# --------------------------------

#----------------plot the result

#-----------------end--------------

#----------test accuracy with test data
test_tree = DecisionTree()
test_tree.learn(X_train, y_train, best_method[0], best_method[1], best_method[2])
test_predictions = test_tree.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(test_accuracy)

# --------------compare to sklearn and time measure --------------------
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
sklearn_predictions = clf.predict(X_test)
sklearn_test_accuracy = accuracy_score(y_test, sklearn_predictions)
print(sklearn_test_accuracy)



