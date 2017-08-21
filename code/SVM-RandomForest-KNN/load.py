import pandas as pd

df = pd.read_csv("UCI HAR Dataset/train/X_train.txt", sep=r"\s+", header=None)
print(df.head())
