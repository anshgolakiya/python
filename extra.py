import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("employee.csv")

age = df["Age"]
salary = df["MonthlyIncome"]

corelation = age.corr(salary)
print("correlation betwen age and salary:")
print(corelation)

plt.scatter(age,salary)
plt.xlabel("Employee Age")
plt.ylabel("salary")
plt.title("relation between age and salary")
plt.show()
