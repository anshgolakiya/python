import pandas as pd

#create a series
s = pd.Series([10,20,30])
print(s)

#create a dataframe
df = pd.DataFrame({
    "Name" : ["Ansh","Amit","Raj","Priya"],
    "dept" : ["IT","HR","IT","HR"],
    "Age" : [25,30,35,40],
    "salary" : [100000,60000,70000,80000]
})
print(df)

#data selection
print(df["Name"])

#data filtering
print(df[df["salary"]>70000])

#grouping
print(df.groupby("dept").groups)

#aggregation
print(df.groupby("dept")["salary"].mean())
