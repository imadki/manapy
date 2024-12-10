from .utils import get_math_symbol

class Constant():
  def __init__(self, name : str):
    self.name = get_math_symbol(name)
  
  def __str__(self):
    return f"C[{self.name}]"
  
  def __repr__(self) -> str:
    return str(self)