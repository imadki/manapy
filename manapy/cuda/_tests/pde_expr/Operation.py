class Operation():
  Laplace = 0
  Grad = 1
  Div = 2
  Dx = 3
  Dy = 4
  Dz = 5
  Dt = 6
  Mul = 7
  Add = 8
  
  def __init__(self, opType : int):
    m = ["Laplace", "Grad", "Divergence", "Dx", "Dy", "Dz", "Dt", "Mul", "Add"]
    self.name = m[opType]
    self.type = opType
  
  def __str__(self):
    return f"OP[{self.name}]"
