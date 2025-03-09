#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 15:35:49 2023

@author: kissami
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

matrixType="aij"
vectorType="seq"


def petsc_mat(A):
    mat = PETSc.Mat().create()
    mat.setSizes(size = A.shape)
    mat.setType(matrixType)
    mat.setUp()
    mat.assemblyBegin()
    mat.setValuesCSR(A.indptr,A.indices,A.data)
    mat.assemblyEnd()
    return mat

def petsc_vec(x):
    b = PETSc.Vec().create()
    b.setType(vectorType)
    b.setSizes(len(x))
    b.setFromOptions()
    b[...] = x[:]
    b.setUp()
    return b

A = np.loadtxt('A2.txt')
B = np.loadtxt('matrix.txt')[::,0:1]

# Create a KSP solver                                                                                                                                                                                   
ksp = PETSc.KSP().create()
ksp.setTolerances(rtol=1e-12)
ksp.setType("lgmres")
#ksp.setInitialGuessNonzero(True)                                                                                                                                                                       
pc = ksp.getPC()
pc.setType("gamg")


ksp = PETSc.KSP().create()                                                                                                                                                                              
ksp.setTolerances(rtol=1e-12)                                                                                                                                                                           
ksp.setType("lgmres")                                                                                                                                                                                   
#ksp.setGMRESRestart(2)                                                                                                                                                                                 
ksp.setInitialGuessNonzero(True)                                                                                                                                                                        
pc = ksp.getPC()                                                                                                                                                                                        
pc.setType("asm")                                                                                                                                                                                       
                

mat = petsc_mat(csr_matrix(A))
vec = petsc_mat(csr_matrix(B))                                                                                                                                                                                     
# Set the linear operator (matrix) and right-hand side vector                                                                                                                                           
ksp.setOperators(mat)                                                                                                                                                                             
ksp.setFromOptions()                                                                                                                                                                                    

z = vec.copy()#duplicate()                                                                                                                                                                              
x = np.zeros_like(B)
r = B - A @ x


ksp.matSolve(vec, z)  