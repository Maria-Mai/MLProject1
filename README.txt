This program creates a decision tree for a specific dataset,
but it can be used for other datasets as well if you just
use the DecisionTree.py file.
The tree can be created, printed and can predict new classes
for unseen data. There are two impurity measures implemented, that
can be used (entropy or gini index).

To run the algorithm/evaluation and execute the file:
(you need the numpy version higher 1.15)

    python3 evaluation.py


Files explanation:

DecisionTree.py: Implementation of the decision tree
Node.py: node class for the decision tree
evaluation.py: run the evaluation
test.py: test on a smaller data set