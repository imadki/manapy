#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:06:29 2024

@author: kissami
"""

import sympy as sp

# Define the symbolic variables
rho, u, E, gamma, p = sp.symbols('rho u E gamma p')
#p = (gamma - 1) * (E - 0.5 * rho * u**2)  # Ideal gas law
#E = p/(gamma -1) + (1/2)*u**2
#z = sp.Symbol('z')


U1, U2, U3 = sp.symbols('U1 U2 U3')
# Define the conservative variables vector and the flux vector
U = sp.Matrix([U1, U2, U3])
F = sp.Matrix([U2, 1/2*(3-gamma)*(U2**2/U1) + (gamma - 1)*U3, gamma*(U2*U3)/U1 - 1/2*(gamma - 1)* U2**3/U1**2])

#p = (gamma - 1) * (U3 - 0.5 * rho * (U2**2/U1))  # Ideal gas law

##E = p/(gamma -1) + (1/2)*(U2**2/U1)
#
#expression = F[0]
#expression_substituted = expression.subs(U[0], z)
#derivative_wrt_z = sp.diff(expression_substituted, z)
#A00 = derivative_wrt_z.subs(z, U[0])
#
#print(expression, "derivative", A00.simplify())
#
#expression = F[1]
#expression_substituted = expression.subs(U[0], z)
#derivative_wrt_z = sp.diff(expression_substituted, z)
#A10 = derivative_wrt_z.subs(z, U[0])
#
#print(expression, "derivative", A10.simplify())
#
#expression = F[2]
#expression_substituted = expression.subs(U[0], z)
#derivative_wrt_z = sp.diff(expression_substituted, z)
#A20 = derivative_wrt_z.subs(z, U[0])
#
#print(expression, "derivative", A20.simplify())
#
#
################################################################################
#expression = F[0]
#expression_substituted = expression.subs(U[1], z)
#derivative_wrt_z = sp.diff(expression_substituted, z)
#A01 = derivative_wrt_z.subs(z, U[1])
#
#print(expression, "derivative", A01.simplify())
#
#expression = F[1]
#expression_substituted = expression.subs(U[1], z)
#derivative_wrt_z = sp.diff(expression_substituted, z)
#A11 = derivative_wrt_z.subs(z, U[1])
#
#print(expression, "derivative", A11.simplify())
#
#expression = F[2]
#expression_substituted = expression.subs(U[1], z)
#derivative_wrt_z = sp.diff(expression_substituted, z)
#A21 = derivative_wrt_z.subs(z, U[1])
#
#print(expression, "derivative", A21.simplify())
#
#
################################################################################
#expression = F[0]
#expression_substituted = expression.subs(U[2], z)
#derivative_wrt_z = sp.diff(expression_substituted, z)
#A02 = derivative_wrt_z.subs(z, U[2])
#
#print(expression, "derivative", A02.simplify())
#
#expression = F[1]
#expression_substituted = expression.subs(U[2], z)
#derivative_wrt_z = sp.diff(expression_substituted, z)
#A12 = derivative_wrt_z.subs(z, U[2])
#
#print(expression, "derivative", A12.simplify())
#
#expression = F[2]
#expression_substituted = expression.subs(U[2], z)
#derivative_wrt_z = sp.diff(expression_substituted, z)
#A22 = derivative_wrt_z.subs(z, U[2])
#
#print(expression, "derivative", A22.simplify())
#
##import sys; sys.exit()
#
#A = sp.Matrix([[A00, A01, A02], [A10, A11, A12], [A20, A21, A22]])
#A = A.subs(U1, rho)
#A = A.subs(U2, u*rho)
#A = A.subs(U3, E)

## Compute the Jacobian matrix A = dF/dU
A = F.jacobian(U)

#E = p/(gamma -1) + (1/2)*(U2**2/U1)
#p = (gamma - 1) * (U3 - 0.5 * U1 * U2**2/U1)
#
#A = A.subs(U1, rho)
#A = A.subs(U2, u*rho)
#A = A.subs(U3, E)
#
## Simplify the Jacobian matrix for better readability
A_simplified = sp.simplify(A)
##
#
## Display the results
#print("Conservative Variables (U):")
#sp.pprint(U, use_unicode=True)
#
#print("\nFlux Vector (F):")
#sp.pprint(F, use_unicode=True)
#
#print("\nJacobian Matrix (A = dF/dU):")
#sp.pprint(A_simplified, use_unicode=True)
