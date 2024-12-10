### 1. Pde Expr

A symbolic module is designed to create a PDE expression in the form of an abstract tree.
* It support
  - Application of `dx, dy, dz, dt, dx2, dy2, dz2, laplace, grad, and divergence` operations to an expression.
  - Expansion of `OP(αF+βG)=αOP(F)+βOP(G)` for `OP=(grad, laplace, divergence)`
  - Expansion of `∇(f⋅g)=f⋅∇g+g⋅∇f`
  - Display expression as a string using `.as_string()`
  - Decomposition of operations using `.decompose()`
  - Displaying the syntax tree using `.print()`
  - Checking equality of two expressions
* Cautions
  - `.as_string()` expands the expression and displays it, even if the expression is factorized. For example ` 2 * (a + b) will displayed as 2 * a + 2 * b`
  - `.decompose()` expands the expression and applies math operations of laplace, grad, and divergence recursively.
  - Equality is compared with tree leaves, so `2*a` and `a + a` are not equal using expression equality However `a + b` and `b + a` are equal because of the associativity of the +, * operators.

\
Details : [pde_expr/main.ipynb](pde_expr/main.ipynb)

### 2. Create System

An expression parser aims to generate a system solver for PDE expressions using manapy atomic functionalities \
\
Details : [create_system/main.ipynb](create_system/main.ipynb)

### Gpu accelerator




