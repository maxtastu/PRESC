#a first look at the wine quality dataset

import pandas as pd

#adapts pandas DF display to dataset
pd.set_option('display.max_columns', 13)
pd.set_option('display.width', 200)
pd.set_option("precision", 3)

#loads dataset and checks how it looks
wine = pd.read_csv("datasets\winequality.csv")
print(wine.head(10))

#according to the dataset source, there should be no missing value.
#but it doesn't hurt to check quickly.
#after a quick look at the csv file, I didn't detect any missing on wrong data.
#the dataset appears to already be cleaned.
#let's check if there are any NaNs.
print(wine.isna().sum()) #there is none!

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

#residual sugar: there has to be a mistake? 65 g/l?
#aside from this one value, residual sugar still varies a lot. it means
#these wines are not the same at all.