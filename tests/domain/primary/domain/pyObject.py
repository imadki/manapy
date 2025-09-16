import manapy_domain32
import numpy as np

print("Hi")

a = np.array([
  [[1, 2, 3, 4],
  [5, 6, 7, 8],
  [4, 5, 6, 9],
  [4, 5, 6, 9]],

  [[1, 2, 3, 9],
  [4, 5, 6, 9],
  [4, 5, 6, 9],
  [4, 5, 6, 9]],
], dtype=np.float64)

b = np.arange(2*2).reshape(2, 2).astype(np.float64)
print(b)

res = manapy_domain32.test_fun(b)
print("res=>", res)