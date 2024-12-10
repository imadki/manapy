from .Node import Node
import re

class Symbol():
  Variable = 0
  Constant = 1

  arrDic = [{}, {}] #Variable Constant

  @staticmethod
  def symbols(s : str, sb_type : int):
    res = []
    pattern = r'^[a-zA-Z ]*$'
    valid = bool(re.match(pattern, s))
    if not valid:
      raise RuntimeError("invalid input")
    tokens = s.split(' ')
    for t in tokens:
      res.append(Symbol(t, sb_type))
    return tuple(res)

  def __init__(self, value, sb_type : int = None):
    if isinstance(value, (float, int)):
      value = str(round(value, 3))
      sb_type = Symbol.Constant
    if not isinstance(value, (Node, str)):
      raise RuntimeError(f"Type Error ${type(value)}")
    
    if isinstance(value, Node):
      self.node = value
    if isinstance(value, str):
      if sb_type not in [Symbol.Constant, Symbol.Variable]:
        raise RuntimeError("Invalid sb_type value") 
      symbolsDic = Symbol.arrDic[sb_type]
      if value in symbolsDic:
        self.node = symbolsDic[value]
      else:
        is_constant = (sb_type == Symbol.Constant)
        self.node = Node(value, is_constant)
        symbolsDic[value] = self.node
    
  
  def laplace(self):
    return Symbol(self.node.laplace())

  def grad(self):
    return Symbol(self.node.grad())
  
  def divergence(self):
    return Symbol(self.node.divergence())

  def dx(self):
    return Symbol(self.node.dx())

  def dy(self):
    return Symbol(self.node.dy())
  
  def dz(self):
    return Symbol(self.node.dz())

  def dt(self):
    return Symbol(self.node.dt())
  
  def print(self):
    self.node.print()
  
  # def exec(self):
  #   return self.node.exec()
  
  def as_string(self):
    return self.node.as_string()
    
  # def getInfo(self, dim):
  #   return self.node.getInfo(dim)
  
  def decompose(self):
    return Symbol(self.node.decompose())

  def __mul__(self, x):
    if not isinstance(x, Symbol):
      x = Symbol(x)
    return Symbol(self.node * x.node)

  def __add__(self, x):
    if not isinstance(x, Symbol):
      x = Symbol(x)
    return Symbol(self.node + x.node)

  # 3 + Symbol => Symbol(3) + Symbol
  def __rmul__(self, x):
    x = Symbol(x)
    return x * self

  def __radd__(self, x):
    x = Symbol(x)
    return x + self
  
  def __str__(self):
    return str(self.node)
  
  def __repr__(self):
    return str(self.node.token.name)

  def __eq__(self, other):
    if not isinstance(other, Symbol):
      return False
    return self.node == other.node
  
  def __hash__(self) -> int:
    return hash(self.node)
  
############################
### Functionalities
############################

def grad(s : Symbol):
  if not isinstance(s, Symbol):
    s = Symbol(s)
  return s.grad()

def laplace(s : Symbol):
  if not isinstance(s, Symbol):
    s = Symbol(s)
  return s.laplace()

def divergence(s : Symbol):
  if not isinstance(s, Symbol):
    s = Symbol(s)
  return s.divergence()

def dx(s : Symbol):
  if not isinstance(s, Symbol):
    s = Symbol(s)
  return s.dx()

def dy(s : Symbol):
  if not isinstance(s, Symbol):
    s = Symbol(s)
  return s.dy()

def dz(s : Symbol):
  if not isinstance(s, Symbol):
    s = Symbol(s)
  return s.dz()

def dt(s : Symbol):
  if not isinstance(s, Symbol):
    s = Symbol(s)
  return s.dt()

def dx2(s : Symbol):
  if not isinstance(s, Symbol):
    s = Symbol(s)
  return s.dx().dx()

def dy2(s : Symbol):
  if not isinstance(s, Symbol):
    s = Symbol(s)
  return s.dy().dy()

def dz2(s : Symbol):
  if not isinstance(s, Symbol):
    s = Symbol(s)
  return s.dz().dz()

def dt2(s : Symbol):
  if not isinstance(s, Symbol):
    s = Symbol(s)
  return s.dt().dt()