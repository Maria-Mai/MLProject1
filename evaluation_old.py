import pandas as pd
from DecisionTreeOld import DecisionTree

#-----------------------proprocess data-----------------------------
file = open("abalone.data", "r")

data = []
for line in file:
    line_array = line.rstrip().split(",")
    changed_type_line= []
    for i in range(len(line_array)):
        if(i == 0):
            changed_type_line.append(line_array[i])
        elif(i >= 1 and i <= 7):
            changed_type_line.append(float(line_array[i]))
        else:
            changed_type_line.append(int(line_array[i]))
    data.append(changed_type_line)


# 1-8 features, class = rings
column_names = ["Sex", "Length", "Diameter", "Height",
                "Whole weight", "Sucked weight",
                "Viscera weight", "Shell weight", "Rings"]

y_name = "Rings"
df_data = pd.DataFrame(data, columns=column_names)
a_tree = DecisionTree()
a_tree.learn(df_data, y_name, "entropy") # was to slow, so I reprogrammed it without dataframes, but therefore it had no problems with strings
a_tree.print()
predictions = a_tree.predict(df_data.iloc[:,0:8])
print(len(predictions))
print(df_data.shape)











