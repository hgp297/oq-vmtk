'''
Calculator for Nonlinear MDOF flexural-shear model
from Xiong et al. (2016)
'''

import numpy as np
import pandas as pd
from openquake.vmtk.units import units
from scipy.optimize import least_squares
from math import sin, sinh, cos, cosh
import warnings

def calculate_bilinear_displacement(V_j, h_j, GA):
    '''
    Calculate displacement capacity once shear capacity 
    and shear stiffness are given using

    delta_j = V_j * h_j / GA 

    Can be calculated for 
    _d : design
    _y : yield

    This is valid for bilinear models, in which the design, yield,
    and peak points are all defined to be in the "elastic" range

    Parameters
    ----------
    V_j : np.array(number of stories)
        Strength of each floor. 

    h_j : np.array(number_of_stories)
        Height of each story in meters

    GA : float
        Whole building shear stiffness calculated from elastic parameters
        outlined by Xiong et al., Equation 1-6

    Returns
    -------
    delta_j: np.array(number_of_stories)
        Displacement capacity of each story
    '''

    return V_j * h_j / GA

def calculate_trilinear_peak_displacement(mu, Omega_p, delta_y):
    '''
    Calculate peak displacement of the trilinear capacity curve
    using Hazus ductility factors

    delta_j = V_j * h_j / GA 

    Can be calculated for 
    _d : design
    _y : yield

    This is valid for trilinear models, the peak point is at a 
    post-yield regime

    Parameters
    ----------
    mu : float
        HAZUS ductility factor

    Omega_p : float
        peak overstrength, ratio between V_p and V_y (Hazus lambda)

    delta_y : np.array(number_of_stories)
        yield displacement of each story

    Returns
    -------
    delta_p: np.array(number_of_stories)
        Peak displacement of each story
    '''

    return mu * Omega_p * delta_y


def calculate_shear_stiffness(T_n, h_j, W_j):
    '''
    Calculate shear stiffness GA as detailed in Xiong et al. (2016)
    
    Parameters
    ----------
    T_n : np.array(2)
        First and second natural periods of the building

    h_j : np.array(number_of_stories)
        Height of each story in meters

    W_j : np.array(number_of_stories)
        Weight of each story in N

    V_j : np.array(number of stories)
        Design shear of each floor, corresponding to the NBCC distribution
        of base shear, in N

    Returns
    -------
    Inventory.df : stores the raw survey as a DataFrame
    '''
    T_1 = T_n[0]
    T_2 = T_n[-1]

    # solve the Miranda & Taghavi approximate NMFS parameters (2005)
    # system of equations

    # have an initial guess that separates gamma_1 and gamma_2 (eigenvalues)
    # stiffer shear walls/concrete structures
    if T_2/T_1 <= 0.29:
        initial_guess = [1.875, 4.694, 2.0]
        solution = least_squares(
            flexural_shear_eigen,
            initial_guess,
            args=(T_1, T_2),
            bounds=([1e-6, 1e-6, 0.0],
                    [np.inf, np.inf, np.inf])
        )
    else:
        initial_guess = [1.875, 4.694, 15.0]
        solution = least_squares(
            flexural_shear_eigen,
            initial_guess,
            args=(T_1, T_2),
            bounds=([1e-6, 1e-6, 5.0],
                    [np.inf, np.inf, 20.0])
        )

    # assert that solution exists
    if solution.status == 0:
        raise RuntimeError(solution.message)

    if not np.allclose(solution.fun, 0, atol=1e-8):
        print(T_2/T_1)
        warnings.warn("Solution has significant residuals")

    gamma_1, gamma_2, alpha_0 = solution.x

    # unit density of each floor (kg/m)
    rho_j = W_j / h_j / units.g
    rho = np.mean(rho_j)

    # building height
    H_bldg = np.sum(h_j)

    # Xiong Equation 5 & 6
    omega_1_sq = 2 * units.pi / T_1
    EI_flexural = (omega_1_sq * rho * (H_bldg**4) / 
                   (gamma_1**2*(gamma_1**2 + alpha_0**2)))

    GA_shear = (alpha_0 / H_bldg)**2 * EI_flexural

    return GA_shear

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

