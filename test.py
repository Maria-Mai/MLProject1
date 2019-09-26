import numpy as np
from DecisionTree import DecisionTree

##test the algorithm with a smaller data set

test_data = [
    [1,0,0,2],
    [0,0,0,4],
    [0,1,1,2],
    [1,0,1,4],
    [0,0,1,2],
    [1,1,1,4],
    [0,1,1,2],
    [0,1,0,2],
    [0,0,0,4],
    [1,1,1,4]]

data = np.asarray(test_data, dtype=np.int32)
a_tree = DecisionTree()
#a_tree.learn2(data[:,:-1], data[:,-1], "entropy", prune=False)
a_tree.learn(data[:,:-1], data[:,-1], "gini", prune=False)
a_tree.print()
pred = a_tree.predict(data[:,:-1])
print(pred)






""" to test the pruning with this data set

val_data = [
            [0,1,1,4],
            [0,0,0,2],
            [0,1,1,4],
            [0,0,1,4],
            [0,0,0,4]]

        valdata = np.asarray(val_data, dtype=np.int32)
        
    return data[:,:-1], data[:,-1], valdata[:,:-1], valdata[:,-1]
"""