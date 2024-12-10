from .Constant import Constant
from .Variable import Variable
from .Operation import Operation
from .EquationInfo import EquationInfo
from collections import deque

class Node():
  def __init__(self, value, is_constant = False):
    if not isinstance(is_constant, bool):
      raise TypeError("Type error is_constant must be of type bool")
    if isinstance(value, str):
      value = Variable(value) if is_constant == False else Constant(value)
    if not isinstance(value, (Operation, Variable, Constant)):
      raise TypeError(f"can't construct Node from {type(value)}")
    self.token = value
    self.right = None
    self.left = None


  def __checkOperation(self, operation : Operation):
    return True

  def __addOperation(self, operation : Operation):
    self.__checkOperation(operation)
    res = Node(operation)
    res.left = self
    res.right = None
    return res

  def laplace(self):
    return self.__addOperation(Operation(Operation.Laplace))

  def grad(self):
    return self.__addOperation(Operation(Operation.Grad))
  
  def divergence(self):
    return self.__addOperation(Operation(Operation.Div))

  def dx(self):
    return self.__addOperation(Operation(Operation.Dx))

  def dy(self):
    return self.__addOperation(Operation(Operation.Dy))
  
  def dz(self):
    return self.__addOperation(Operation(Operation.Dz))

  def dt(self):
    return self.__addOperation(Operation(Operation.Dt))

  def exec(self):
    if isinstance(self.token, Operation):
      if self.token.type == Operation.Add:
        return f"{self.left.exec()} + {self.right.exec()}"
      elif self.token.type == Operation.Mul:
        return f"({self.left.exec()})({self.right.exec()})"
      elif self.token.type == Operation.Dt:
        return f"Dt{self.token.order}({self.left.exec()})"
      elif self.token.type == Operation.Dx:
        return f"Dx{self.token.order}({self.left.exec()})"
      elif self.token.type == Operation.Dy:
        return f"Dy{self.token.order}({self.left.exec()})"
      elif self.token.type == Operation.Dz:
        return f"Dz{self.token.order}({self.left.exec()})"
      elif self.token.type == Operation.Laplace:
        return f"Laplace({self.left.exec()})"
      elif self.token.type == Operation.Grad:
        return f"Grad({self.left.exec()})"
    elif isinstance(self.token, (Variable, Constant)):
      return self.token.name
    return ""

  def as_string(self, color = 0):
    arrColors = ['\033[31m', '\033[32m', '\033[33m', '\033[34m']
    ENDC = '\033[0m'
    arr = self.expand()
    res = ""
    for i, add in enumerate(arr):
      for j, mul in enumerate(add):
        if isinstance(mul.token, Operation):
          clr = arrColors[color % len(arrColors)]
          res += f"{mul.token.name}{clr}({ENDC}{mul.left.as_string(color + 1)}{clr}){ENDC}"
        else :
          res += mul.token.name
        if j < len(add) - 1:
          res += "."
      if i < len(arr) - 1:
        res += " + "
    return res

  """
    return an array of array
      the parent array represent adding
      the nested array represent multiplication
      ex : 2 * (a + b) -> 2 * a + 2 * b -> [[2, a], [2, b]]
  """
  def expand(self):
    if isinstance(self.token, Operation):
      if self.token.type == Operation.Add:
        return self.left.expand() + self.right.expand()
      elif self.token.type == Operation.Mul:
        a = self.left.expand()
        b = self.right.expand()
        res = []
        for i in a:
          for j in b:
            z = i + j
            res.append(z)
        return res
    return [[self]]

  """
    get the expanded tree from the result of expand method
  """
  @staticmethod
  def expand_tree(arr):
    addArr = []
    for add in arr:
      addArr.append(Node.mulArrFun(add))
    return Node.addArrFun(addArr)
  


  """
    self : is the content inside the operation
      ex : grad(a + b) -> self will be -> a + b
    
    operation: is lambda function that define the operation
      it can be grad, laplace, curl, div or rot
      ex : lambda x : x.grad()
    
    Given constants K and L, and functions a and b
      apply OP(Ka + Lb)= K.OP(a) + L.OP(b) 
    
    return : the expression that will replace the operation Aka the parent of self
  """
  def linear_operator_decomposition(self, operation):
    arr = self.expand()

    mulFun = self.mulArrFun
    addFun = self.addArrFun

    addArr = []
    for add in arr:
      mulArrVar = [item for item in add if not isinstance(item.token, Constant)]
      mulArrConst = [item for item in add if isinstance(item.token, Constant)]
      if len(mulArrVar) != 0 and len(mulArrConst) != 0:
        addArr.append(mulFun(mulArrConst) * operation(mulFun(mulArrVar)))
      elif len(mulArrVar) != 0:
        addArr.append(operation(mulFun(mulArrVar)))
      elif len(mulArrConst) != 0:
        if len(mulArrConst) > 1:
          addArr.append(mulFun(mulArrConst[1:]) * operation(mulArrConst[0]))
        else:
          addArr.append(operation(mulArrConst[0]))
    return addFun(addArr)

  """
    self : is the content inside the operation
      ex : grad(a . b) -> self will be -> a . b
    
    self expected to be only in a product form

    Given functions a and b
      apply grad(a.b) = a * grad(b) + b * grad(a) recursively
    

    return : the expression that will replace the operation Aka the parent of self
  """
  def gradient_product_rule(self):
    arr = self.expand()[0]

    def rule2_req(arr):
      if len(arr) == 0:
        raise RuntimeError("??")
      if len(arr) == 1:
        return arr[0].grad()
      return arr[0] * rule2_req(arr[1:]) + self.mulArrFun(arr[1:]) * arr[0].grad()

    return rule2_req(arr)

  """
    self is an expression (Node)

    apply gradient_product_rule to th 
  """
  def apply_gradient_product_rule(self):
    arr = self.expand()
    for add in arr:
      for i, mul in enumerate(add):
        if isinstance(mul.token, Operation) and mul.token.type == Operation.Grad:
          add[i] = mul.left.gradient_product_rule()
    return self.expand_tree(arr)

  """
  
  """
  def __decompose(self):
    arr = self.expand()
    operations = {
      Operation.Grad : lambda x : x.grad(),
      Operation.Laplace : lambda x : x.laplace(),
      Operation.Div : lambda x : x.divergence(),
    }
    for add in arr:
      for i, mul in enumerate(add):
        allowed = [Operation.Grad, Operation.Laplace, Operation.Div]
        if isinstance(mul.token, Operation) and mul.token.type in allowed:
          target = mul.left
          target = target.__decompose()
          expanded = target.linear_operator_decomposition(operations[mul.token.type])
          if mul.token.type == Operation.Grad:
            expanded = expanded.apply_gradient_product_rule()
          add[i] = expanded
    return self.expand_tree(arr)
  
  def decompose(self):
    res = self.__decompose()
    res = res.expand()
    return self.expand_tree(res)

  def getInfo(self):
    #! TODO calculate order
    res = EquationInfo()
    arr = self.expand()
    for add in arr:
      for mul in add:
        if isinstance(mul.token, Variable):
          res.variables.append(mul)
        elif isinstance(mul.token, Constant):
          res.constants.append(mul)
        elif isinstance(mul.token, Operation):
          op = mul.token
          if op.type == Operation.Grad:
            res.hasGrad = True
            res.gradVar.append(mul.left)
          if op.type == Operation.Laplace:
            res.hasLaplace = True
            res.laplaceVar.append(mul.left)
          res.variables.append(mul.left)
    return res


  """
   - if x = 1 or self.token = 1 constant, just return the node Ex: 1 * a = a, a * 1 = a, 1 * 1 = 1
   - 1/Avoid multiplication between operations such as (dx * dx), (dy * dx), or (dy * grad)
    - grad * grad | grad * laplace | laplace * laplace => possible for the same variable
   - 2/Prevent multiplication between variables and operations involving variables, such as dx(u) * v
   - 3/Avoid multiplication between different variables, such as v * u
  """
  def __mul__(self, x):
    if not isinstance(x, Node):
      x = Node(x)
    
    res = Node(Operation(Operation.Mul))
    res.left = self
    res.right = x
    return res

  def __rmul__(self, x):
    x = Node(x)
    return x * self

  def __add__(self, x):
    if not isinstance(x, Node):
      x = Node(x)
    res = Node(Operation(Operation.Add))
    res.left = self
    res.right = x
    return res

  def __radd__(self, x):
    x = Node(x)
    return x + self

  def __eq__(self, other):
    if not isinstance(other, Node):
      return False
    if type(self.token) == type(other.token):
      if isinstance(self.token, Operation):
        if (self.token.type in [Operation.Add, Operation.Mul]) and self.token.type == other.token.type:
          return (self.left == other.left and self.right == other.right) or \
            (self.left == other.right and self.right == other.left)
        elif self.token.type == other.token.type:
          return self.left == other.left
      else:
        return self.token == other.token
    return False
  
  def __hash__(self) -> int:
    return hash(self.token)
   

  # region Printing
    ##########################################
    ## Printing
    ##########################################


  def __print_item_in_center(self, item : str, size : int):
    if len(item) > size:
      print(item[:size], end='.')
    else:
      nb_space = (size - len(item)) // 2
      print(' ' * nb_space, end='')
      print(item, end='')
      print(' ' * (size - nb_space - len(item)), end='')
  
  def __print_branch(self, size : int):
    nb = size // 4
    print(' ' * nb, end='')
    print('╭', end='')
    i = nb + 1
    while i + nb + 2 < size:
      if (i == (size - 1) // 2):
        print("┴", end='')
      else:
        print("─", end='')
      i += 1
    print('╮', end='')
    print(' ' * (size - (i + 1)), end='')
  
  def __print_tree(self, tree_height : int):
    level_item = tree_height
    item_len = 2
    buffer_size = pow(2, level_item) * item_len

    level = 1
    depth = 0
    level_type = 1
    level_item = 1

    queue = deque()
    is_empty_node = deque()

    queue.append(self)
    while depth < tree_height:
      if (level_type == 1):
        node = queue.popleft()
        if node :
          self.__print_item_in_center(str(node.token), buffer_size)
          queue.append(node.left)
          queue.append(node.right)
          is_empty_node.append(True)
        else :
          queue.append(None)
          queue.append(None)
          self.__print_item_in_center(" ", buffer_size)
          is_empty_node.append(False)
        level_item -= 1
        if (level_item == 0) :
          level_type = 2
          level_item = level
          level *= 2
          depth += 1
          print()
      else :
        while level_item != 0:
          is_empty = is_empty_node.popleft()
          if is_empty:
            self.__print_branch(buffer_size)
          else:
            self.__print_item_in_center(" ", buffer_size)
          level_item -= 1
        level_item = level
        level_type = 1
        buffer_size //= 2
        print()

  def tree_height(self, node):
      if node is None:
          return -1
      left_height = self.tree_height(node.left)
      right_height = self.tree_height(node.right)
      return 1 + max(left_height, right_height)

  # endregion

  def print(self):
    h = self.tree_height(self) + 1
    self.__print_tree(h)
  
  @staticmethod
  def mulArrFun(arr):
    if len(arr) == 0:
      return None
    else:
      res = arr[0]
      for item in arr[1:]:
        res = res * item
      return res

  @staticmethod
  def addArrFun(arr):
    if len(arr) == 0:
      return None
    else:
      res = arr[0]
      for item in arr[1:]:
        res = res + item
      return res

  def __str__(self):
    return str(self.token)
  
  def __repr__(self):
    return str(self.token.name)