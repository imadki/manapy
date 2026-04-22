import jax.numpy as jnp
from jax import grad, vmap
import numpy as np

def f(x, y):
    return x**3 + y**2

grad_x = grad(f, argnums=0)
grad_y = grad(f, argnums=1)


x = jnp.array(np.arange(-3.0, 3.0, 0.1))
y = jnp.array(np.arange(-3.0, 3.0, 0.1))
print(vmap(grad_x)(x, y))