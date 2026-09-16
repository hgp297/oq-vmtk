'''
Calculator for Nonlinear MDOF flexural-shear model
from Xiong et al. (2016)
'''

import numpy as np
import pandas as pd
from openquake.vmtk.units import units
from scipy.optimize import least_squares
from math import sin, sinh, cos, cosh

def calculate_parameters(T_n, H):
    '''
    Calculate NMFS parameters as detailed in Xiong et al. (2016)
    
    '''
    T_1 = T_n[0]
    T_2 = T_n[-1]

    # solve the Miranda & Taghavi approximate NMFS parameters (2005)
    # system of equations

    # have an initial guess that separates gamma_1 and gamma_2 (eigenvalues)
    initial_guess = [1.875, 4.694, 2.0]
    solution = least_squares(
        flexural_shear_eigen,
        initial_guess,
        args=(T_1, T_2),
        bounds=([1e-6, 1e-6, 0.0],
                [np.inf, np.inf, np.inf])
    )

    # assert that solution exists
    if solution.status != 1:
        raise RuntimeError(solution.message)

    if not np.allclose(solution.fun, 0, atol=1e-8):
        raise RuntimeError("Solution has significant residuals")

    gamma_1, gamma_2, alpha_0 = solution.x

def flexural_shear_eigen(parameters, T_1, T_2):
    '''
    From Approximate Floor Accelerations Demands in Multistory Buildings Formulation
    (Miranda & Taghavi 2005).

    Parameters
    --------------
    parameters: list
        gamma_1: eigenvalue associated with 1st mode of vibration
        gamma_2: eigenvalue associated with 2nd mode of vibration
        alpha_0: flexural-shear stiffness ratio, nondimensional

    T_1: float:
        1st mode vibration period
    T_2: float
        2nd mode vibration period

    Returns 
    -------------
    list: 
        char_eq_1: float
            1st mode characteristic equation (paper Equation 24)
        char_eq_2: float
            2nd mode characteristic equation (paper Equation 24)
        period_ratio_eqn:
            Ratio between fundamental period and higher (second mode) period
            (paper Equation 25)

    '''
    gamma_1 = parameters[0]
    gamma_2 = parameters[1]
    alpha_0 = parameters[-1]

    period_ratio_eqn = gamma_1/gamma_2 * ((gamma_1**2 + alpha_0**2)/
                                          (gamma_2**2 + alpha_0**2))**0.5 - (T_2 / T_1)
    char_eq_1 = (2 + 
                 (2 + alpha_0**4 / (gamma_1**2*(gamma_1**2 + alpha_0**2)))*
                 cos(gamma_1) * cosh((alpha_0**2 + gamma_1**2)**0.5) +
                 alpha_0**2 / (gamma_1*(gamma_1**2 + alpha_0**2)**0.5)*
                 sin(gamma_1) * sinh((alpha_0**2 + gamma_1**2)**0.5))
    
    char_eq_2 = (2 + 
                 (2 + alpha_0**4 / (gamma_2**2*(gamma_2**2 + alpha_0**2)))*
                 cos(gamma_2) * cosh((alpha_0**2 + gamma_2**2)**0.5) +
                 alpha_0**2 / (gamma_2*(gamma_2**2 + alpha_0**2)**0.5)*
                 sin(gamma_2) * sinh((alpha_0**2 + gamma_2**2)**0.5))

    return [period_ratio_eqn, char_eq_1, char_eq_2]

