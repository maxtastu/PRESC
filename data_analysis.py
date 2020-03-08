#a first look at the wine quality dataset

import pandas as pd

#adapts pandas DF display to dataset
pd.set_option('display.max_columns', 14)
pd.set_option('display.width', 200)
pd.set_option("precision", 3)

#loads dataset and checks how it looks
wine = pd.read_csv("datasets\winequality.csv")
#wine["SO2ratio"] = (wine["total sulfur dioxide"] - wine["free sulfur dioxide"]) / wine["free sulfur dioxide"]
print(wine.head(10))

#according to the dataset source, there should be no missing value.
#but it doesn't hurt to check quickly.
#after a quick look at the csv file, I didn't detect any missing on wrong data.
#the dataset appears to already be cleaned.
#let's check if there are any NaNs.
print(wine.isna().sum()) #there is none!

#spreading of quality grades
print(wine["quality"].value_counts().sort_index())
print(wine["recommend"].value_counts().sort_index())


#let's look at basic descriptive stats
wineStats = pd.concat(
        {"mean":wine.mean(),
         "median":wine.median(),
#         "variance":wine.var(),
         "std":wine.std(),
         "min":wine.min(),
         "max":wine.max(),
                },
        axis=1
        )

print(wineStats)

#pd.plotting.scatter_matrix(wine.drop(["quality","recommend"], axis=1), figsize=(25, 25),c=wine["recommend"])
wine_cleaned = wine.drop([2781,4745]) #drop 2 valeurs extrêmes qui pourrissent tout

#drop 1 valeur de densité >= 1.01 qui pourrit encore pas mal
#2 points correspondent : 1653 et 1663. les lignes sont exactement les 2 mêmes.
#ça ressemble à une erreur.
wine_cleaned.drop(wine_cleaned.loc[wine_cleaned["density"]>=1.01].index, inplace=True)

#et puis je drop les hautes valeurs de sel
#wine_cleaned.drop(wine_cleaned.loc[wine_cleaned["chlorides"]>=0.10].index, inplace=True)


#pd.plotting.scatter_matrix(wine_cleaned.drop(["quality","recommend"], axis=1), figsize=(25, 25),c=wine_cleaned["recommend"])
#
#pd.plotting.scatter_matrix(wine_cleaned.drop(["quality","recommend"], axis=1), figsize=(25, 25),c=wine_cleaned["quality"], cmap="RdBu")


### first model training test ###

#splits dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        wine_cleaned.drop(["quality","recommend"], axis=1),
        wine_cleaned["recommend"],
        random_state=0
        )

#k-nearest neighbors

from sklearn.neighbors import KNeighborsClassifier

#def knn(k):
#    """trains a knn algorithm with k neighbors
#    and show scores on test data
#    
#    """
#    knn = KNeighborsClassifier(n_neighbors=k)
#    print("neighbors :{}".format(k))
#
#    knn.fit(X_train, y_train)
#    
#    training_acc = knn.score(X_train, y_train)
#    test_acc  = knn.score(X_test,y_test)
#    
#    print("Test set score: {:.2f}".format(test_acc))
#
#    y_eval = y_test.to_frame()
#    y_eval["prediction"] = knn.predict(X_test)
#    print(y_eval.sum(axis=1).value_counts())
#    
#    return training_acc, test_acc
#
#training_accuracy = []
#test_accuracy = []
#
#for i in range(1,11):
#        training_acc, test_acc = knn(i)
#        training_accuracy.append(training_acc)
#        test_accuracy.append(test_acc)
#
#import matplotlib.pyplot as plt
#plt.plot(range(1,11), training_accuracy, label="training accuracy")
#plt.plot(range(1,11), test_accuracy, label="test accuracy")
#plt.legend()

# decision tree

#from sklearn.tree import DecisionTreeClassifier
#
#tree = DecisionTreeClassifier(
#        max_depth=4, 
#        random_state=0,
##        min_samples_leaf=50,
#        )
#tree.fit(X_train, y_train)
#print("accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
#print("accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
#
#
#from sklearn.tree import export_graphviz
#export_graphviz(
#        tree, 
#        out_file="tree.dot",
#        feature_names=X_train.columns,
#        class_names=["bad","good"],
#        impurity=False,
#        filled=True,
#        )
#
#import graphviz
#with open("tree.dot") as f:
#    dot_graph = f.read()
#graph = graphviz.Source(dot_graph)
#graph.view()

# random forest

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=5)
forest.fit(X_train, y_train)