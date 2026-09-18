import numpy as np

print("1. array creation")
arr = np.array([[1,2,3],[4,5,6]])

print("original array:")
print(arr)

#indexing
print("element at index (0,1):",arr[0,1])

#broadcasting
print("Broadcasting:")
print(arr + 10)

#math operations
print("Math Operations:")
print(arr * 2)
print(arr + 2)
print(arr ** 2)
print("Sum of array:", np.sum(arr))
print("Mean of array:", np.mean(arr))
print("Max of array:", np.max(arr))

#reshaping
print("Dimension of array:", arr.ndim)
print("Shape of array:", arr.shape)
new_arr = arr.reshape(3,2)
print("Reshaped array:")
print(new_arr)