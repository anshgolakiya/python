import pandas as pd

df = pd.read_csv("employee.csv")

data = df["Age"].dropna()

#mean
print("Mean: ", df["Age"].mean())

#describe()
print(df["Age"].describe())

#median
print("Median: ", df["Age"].median())

#mode
print("Mode: ", df["Age"].mode())

#variance
print("Variance: ", df["Age"].var())

#range
max = df["Age"].max()
min = df["Age"].min()
print("age range of employee in company: ", min , "to", max)

#iqr of age column
q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)
print("IQR of Age: ", q3 - q1)
