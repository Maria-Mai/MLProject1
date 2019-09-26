from DecisionTree import DecisionTree
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn import tree
import numpy as np
import timeit
import matplotlib.pyplot as plt

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
#----------------preprocces data end-------------------------------


# ---------------divide data for 5 fold cross validation -------------------
#train test split, oly use train for cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

folds = 5
k_fold = KFold(n_splits=folds)
# --------------end


#------------cross validate the data 10 times (gini, gini + prune 0.1, gini + prune 0.2 gini + prune 0.3, gini + prune 0.4 + 0.5 + 0.6,
#                                               entropy, entropy + prune 0.1, 0.2, 0.3 0.4 0.5 0.6)----------------
method_list = [("entropy", False, 0), ("entropy", True, 0.1), ("entropy", True, 0.2), ("entropy", True, 0.3), ("entropy", True, 0.4), ("entropy", True, 0.5), ("entropy", True, 0.6),
               ("gini", False, 0), ("gini", True, 0.1), ("gini", True, 0.2), ("gini", True, 0.3), ("gini", True, 0.4), ("gini", True, 0.5), ("gini", True, 0.6)]

accuracy_list = list()

for method in method_list:
    val_accuracy = list()
    for train_index, val_index in k_fold.split(X_train):
        a_tree = DecisionTree()
        a_tree.learn(X_train[train_index], y_train[train_index], method[0], prune=method[1], prune_size=method[2])
        predictions = a_tree.predict(X_train[val_index])
        val_accuracy.append(accuracy_score(y_train[val_index], predictions))
    cross_accuracy = sum(val_accuracy)/folds
    accuracy_list.append((method, cross_accuracy))

sorted_list = sorted(accuracy_list, key=lambda tup: tup[1])
best_method = sorted_list[-1][0]
# --------------------------------

#----------------plot the result
entropy_prune_list = list()
gini_prune_list = list()
for element in sorted_list:
    if element[0][0] == "entropy":
        entropy_prune_list.append([element[0][2], element[1]])
    else:
        gini_prune_list.append([element[0][2], element[1]])

entropy_prune_list = sorted(entropy_prune_list, key=lambda tup: tup[0])
gini_prune_list = sorted(gini_prune_list, key=lambda tup: tup[0])

x_val_entropy = [x[0] for x in entropy_prune_list]
y_val_entropy = [x[1] for x in entropy_prune_list]

x_val_gini = [x[0] for x in gini_prune_list]
y_val_gini = [x[1] for x in gini_prune_list]

plt.figure(figsize=(10,8))
plt.plot(x_val_entropy, y_val_entropy, label="entropy")
plt.plot(x_val_gini, y_val_gini, label="gini")
plt.grid()
plt.xlabel('prune size')
plt.ylabel('accuracy')
plt.legend()
plt.show()

#-----------------end--------------

#----------test accuracy with test data
start = timeit.default_timer()
test_tree = DecisionTree()
test_tree.learn(X_train, y_train, best_method[0], best_method[1], best_method[2])
test_predictions = test_tree.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
stop = timeit.default_timer()
run_time = stop -start
print(best_method)
print("custom accuracy: " + str(test_accuracy) + " time: " + str(run_time))

# --------------compare to sklearn and time measure --------------------
start = timeit.default_timer()
clf = tree.DecisionTreeClassifier("entropy")
clf.fit(X_train, y_train)
sklearn_predictions = clf.predict(X_test)
sklearn_test_accuracy = accuracy_score(y_test, sklearn_predictions)
stop = timeit.default_timer()
run_time = stop -start
print("sklearn entropy : " + str(sklearn_test_accuracy) + " time: " + str(run_time))

start = timeit.default_timer()
clf = tree.DecisionTreeClassifier("gini")
clf.fit(X_train, y_train)
sklearn_predictions = clf.predict(X_test)
sklearn_test_accuracy = accuracy_score(y_test, sklearn_predictions)
stop = timeit.default_timer()
run_time = stop -start
print("sklearn gini: " + str(sklearn_test_accuracy) + " time: " + str(run_time))



