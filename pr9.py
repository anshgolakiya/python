import pandas as pd
df = pd.read_csv("employee.csv")

#Univariate Analysis
df["Age"].dropna()
print(df["Age"].describe())

#bivariate Analysis
df[["Age", "MonthlyIncome"]].dropna()
print(df[["Age", "MonthlyIncome"]].describe())

#Multivariate Analysis
df[["Age", "MonthlyIncome", "TotalWorkingYears"]].dropna()
print(df[["Age", "MonthlyIncome", "TotalWorkingYears"]].describe())