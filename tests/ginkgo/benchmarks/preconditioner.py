import pandas as pd
import plotly.express as px
import pandas as pd
import numpy as np

def filter(arr):
  filter = []
  for item in arr:
      residual = item[-1]
      time = item[-2]
      iter = item[-3]
      if residual != float('nan') and time != float('nan')  and iter != float('nan'):
          if iter > 0 and time > 0 and residual > 0 and residual < 10e-1:
              filter.append(item)
  return filter
        
def parse_ginkgo_benchmarks(file_name):
  ginkgo = open(file_name, "r")

  ginkgo_lines = ginkgo.readlines()

  arr = []
  for line in ginkgo_lines:
    line = line.strip()
    info = line.split()
    if len(info) > 4:
      residual = float(info[-1])
      time = int(info[-2].split(":")[1][:-2])
      iterations = int(info[-3].split(":")[1])
      preconditioner = info[-4]
      solver = info[-5]
      exec = info[-6]

      # if solver == "cg" or exec == "cuda":
      #     continue
      arr.append([
          "ginkgo",
          exec,
          solver,
          preconditioner,
          iterations,
          time,
          residual
      ])
  
  ginkgo.close()
  return arr

def parse_petsc_benchmarks(file_name):
  petsc = open(file_name, "r")

  petsc_lines = petsc.readlines()

  arr = []
  for line in petsc_lines:
      line = line.strip()
      info = line.split()
      if len(info) > 4:
          residual = float(info[-1])
          time = int(float(info[-2].split(":")[1][:-1]) * 1000)
          iterations = int(info[-3].split(":")[1])
          preconditioner = info[-4]
          solver = info[-5]
          exec = "cpu-1" if len(info) == 6 and info[-6] == '*' else "cpu-4"
          arr.append([
              "petsc",
              exec,
              solver,
              preconditioner,
              iterations,
              time,
              residual
          ])
  petsc.close()
  return arr

def print_data(arr):
  print("Petsc Or Ginkgo, Exec, Solver, Preconditioner, Number of Iteration, Time in ms, Residual")
  def tt(a):
      return str(a).ljust(12)
  for item in arr:
      print(" ".join(map(tt, item)))

arr = parse_ginkgo_benchmarks("./ginkgo_benchmarks_zero.txt")
arr = filter(arr)
arr.sort(key=lambda item: item[-5]) # Choose Index to sort by

arr = np.array(arr)
print(arr[0])
# print_data(arr[:][0])

# Example dataset


data = {
  "Solver": arr[:, 0],
  "exec": arr[:, 1],
  "solver_name": arr[:, 2],
  "preconditioner": arr[:, 3],
  "nb_iteration": (arr[:, 4]).astype(int),
  "time": (arr[:, 5]).astype(int),
  "residual": (arr[:, 6]).astype(float)
}

df = pd.DataFrame(data)


active_solver_name = ["bicg", "bicgstab", "cg", "gcr", "gmres", "cgs"]
active_preconditioners = ["none", "ILU"]
active_exec = ["omp", "cuda"]
# df_filtered = df[df["solver_name"].isin(active_preconditioners)]
df_filtered = df
df_filtered = df_filtered[df_filtered["exec"].isin(active_exec)]
df_filtered = df_filtered[df_filtered["solver_name"].isin(active_solver_name)]

# print(df_filtered)

# Plot nb_iteration vs solver
fig = px.line(
    df_filtered,
    x="preconditioner",
    y="residual",
    color="solver_name",
    line_dash="exec",
    markers=True,
    title="Residual comparison"
)

fig.show()