'''
Calculator for Nonlinear MDOF flexural-shear model
from Xiong et al. (2016)
'''

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from openquake.vmtk.units import units
from scipy.optimize import least_squares
from math import sin, sinh, cos, cosh
import warnings
from typing import Callable, NamedTuple

def calculate_elastic_displacement(V_j, h_j, GA):
    '''
    Calculate displacement capacity once shear capacity 
    and shear stiffness are given using

    delta_j = V_j * h_j / GA 

    Can be calculated for 
    _d : design
    _y : yield

    This is only valid for pre-yield parameters in the elastic
    range. Do not use this to calculate peak/ultimate displacements

    Parameters
    ----------
    V_j : np.array(number of stories)
        Strength of each floor, units of kN

    h_j : np.array(number_of_stories)
        Height of each story in meters

    GA : float
        Whole building shear stiffness calculated from elastic parameters
        outlined by Xiong et al., Equation 1-6, units of kN

    Returns
    -------
    delta_j: np.array(number_of_stories)
        Displacement capacity of each story, units of m
    '''

    return V_j * units.kN * h_j * units.m / (GA * units.kN)

def determine_sdof_stiffness(nst, T_eff, m_0, is_sos=False):
    """
    Determine k_0, the first mode stiffness corresponding to the
    eigenproblem 

    k_0 [A] = omega_0^2 m_0 [I]

    Assumes that mass is distributed uniform, except roof, which has 75% 
    mass (modified I). Stiffness is distributed uniformly.

    Parameters
    ----------
    nst: int
        Number of stories
    T_eff: int
        Effective pre-yield period, considering post-cracking pre-yield
        behavior in RC/masonry and NBC underprediction in SMF.
        Basis is found in Saatcioglu and Humar (2003).
    m_0: float
        Mass from actual estimate, redistributed such that total mass 
        is the same, but distribution follows the [1 1 1 0.75] pattern.
    is_sos : bool, optional
        True for soft-storey buildings. Softens the ground-floor
        stiffness used to derive the mode shape. Default False.

    Returns
    ----------
    k0: float
        Stiffness
    """
    I_mat = np.identity(nst)
    if nst > 1:
        I_mat[-1, -1] = 0.75
    
    A_mat = np.zeros((nst, nst))
    np.fill_diagonal(A_mat, 2)
    A_mat[-1, -1] = 1
    if is_sos:
        A_mat[0, 0] = 1.20
    for i in range(nst - 1):
        A_mat[i, i + 1] = A_mat[i + 1, i] = -1

    _, eigenvectors = eigh(A_mat, I_mat)
    phi = eigenvectors[:, 0]
    phi = phi / phi[-1]

    lam = (phi @ I_mat @ phi) / (phi @ A_mat @ phi)

    k_0 = lam * 4 * units.pi**2 * m_0 / (T_eff**2)
    return k_0


# def triangular_ritz_displacement(nst, Vb, )


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
        initial_guess = [1.875, 4.694, 20.0]
        solution = least_squares(
            flexural_shear_eigen,
            initial_guess,
            args=(T_1, T_2),
            bounds=([1e-6, 1e-6, 5.0],
                    [np.inf, np.inf, 50.0])
        )

    # assert that solution exists
    if solution.status == 0:
        raise RuntimeError(solution.message)

    if not np.allclose(solution.fun, 0, atol=1e-8):
        warnings.warn("Solution has significant residuals")

    gamma_1, gamma_2, alpha_0 = solution.x

    # unit density of each floor (1e3 kg/m)
    rho_j = W_j*units.kN / h_j*units.m / units.g
    rho = np.mean(rho_j)

    # building height
    H_bldg = np.sum(h_j)

    # Xiong Equation 5 & 6
    omega_1 = (2 * units.pi / T_1)
    EI_flexural = (omega_1**2 * rho * (H_bldg**4) / 
                   (gamma_1**2*(gamma_1**2 + alpha_0**2))) # units of kN m^2

    GA_shear = (alpha_0 / H_bldg)**2 * EI_flexural # units of kN

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

    period_ratio_eqn = (T_2 / T_1) - (gamma_1/gamma_2) * ((gamma_1**2 + alpha_0**2)/
                                          (gamma_2**2 + alpha_0**2))**0.5
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

class CurveBuilder(NamedTuple):
    shear: Callable   # (Vy, Vp, c)    -> (n_story, n_pts) normalized shear
    drift: Callable   # (uy, up, uu, upc)  -> (n_story, n_pts) drift (ratio or m)


# default curve is trilinear: yield, peak, postcap
def _shear_default(Vy, Vp, c):
    return np.column_stack([Vy, Vp, c * Vp])

def _drift_default(uy, up, uu, upc, Vy=None, Vp=None, c=None):
    return np.column_stack([uy, up, upc])

# ductile, elastic until "effective" yield, then perfectly plastic
# until halfway between ultimate and yield, then 0.8Vmax
def _shear_ductile_steel(Vy, Vp, c):
    '''EPP-style eff_yield_peak-peak-ultimate model'''
    return np.column_stack([Vp, Vp, 0.8*Vp, c * Vp])

def _drift_ductile_steel(uy, up, uu, upc, Vy, Vp, c=None):
    '''EPP-style eff_yield_peak-peak-ultimate model'''
    uy_eff = Vp/Vy*uy
    mu_u = uu/uy
    # FEMA 351 shows that even Pre-Northridge buildings have 
    # ultimate-to-yield ratio of about 2.5-3.0. Our current values
    # are 2.07 around Northridge, and 3.0 post-Northridge. It is then
    # reasonable to provide some nominal 2.0 ductility for pre-northridge
    # cases that bottom out at 1.0. This value is the moderate-extensive 
    # damage state for pre-northridge in FEMA 351
    mu_u = np.maximum(mu_u, 2.0)
    uu_eff = mu_u * uy_eff
    mu_pc = upc/uy
    upc_eff = np.maximum(mu_pc * uy_eff, 1.01*uu_eff)
    return np.column_stack([uy_eff, 0.5*(uy_eff + uu_eff), uu_eff, upc_eff])

def _shear_cfs(Vy, Vp, c):
    '''yield-peak-ultimate model'''
    return np.column_stack([Vy, Vp, 0.8*Vp, c * Vp])

def _drift_cfs(uy, up, uu, upc, Vy=None, Vp=None, c=None):
    '''yield-peak-ultimate model'''
    return np.column_stack([uy, 0.5*(uy + uu), uu, upc])


def _shear_ductile_cmf(Vy, Vp, c):
    '''EPP-style eff_yield_peak-peak-ultimate model'''
    return np.column_stack([Vp, Vp, 0.8*Vp, c * Vp])

def _drift_ductile_cmf(uy, up, uu, upc, Vy, Vp, c=None):
    '''EPP-style eff_yield_peak-peak-ultimate model'''
    uy_eff = Vp/Vy*uy
    mu_u = uu/uy
    uu_eff = mu_u * uy_eff
    mu_pc = upc/uy
    upc_eff = 1.2*uu_eff
    return np.column_stack([uy_eff, 0.5*(uy_eff + uu_eff), uu_eff, upc_eff])



def _shear_wood(Vy, Vp, c):
    '''yield-peak-ultimate model'''
    return np.column_stack([Vy, Vp, 0.8*Vp, c * Vp])

def _drift_wood(uy, up, uu, upc, Vy=None, Vp=None, c=None):
    '''yield-peak-ultimate model'''
    return np.column_stack([uy, 0.5*(uy + uu), uu, upc])

def _drift_wpb(uy, up, uu, upc, Vy=None, Vp=None, c=None):
    '''Compared to wlf, allow more post-cap ductility'''
    return np.column_stack([uy, 0.5*(uy + uu), uu, 2*uu])

def _shear_sw(Vy, Vp, c):
    '''shear-wall yield-peak-dropoff model'''
    return np.column_stack([Vy, Vp, c * Vp, c * Vp])

def _drift_sw(uy, up, uu, upc, Vy=None, Vp=None, c=None):
    '''shear-wall yield-peak-dropoff model'''
    
    x = 0.15*uu/(0.2*Vp)*(1-c)*Vp
    uplateau = 0.85*uu + x
    return np.column_stack([uy, 0.85*uu, uplateau, 3*uplateau])

DEFAULT_BUILDER = CurveBuilder(_shear_default, _drift_default)

CAPACITY_BUILDERS = {
    "WLF": CurveBuilder(_shear_wood, _drift_wood),
    "WLF-P9": CurveBuilder(_shear_wood, _drift_wood),
    "WPB": CurveBuilder(_shear_wood, _drift_wpb),
    "SMF": CurveBuilder(_shear_ductile_steel, _drift_ductile_steel),
    "SBF": CurveBuilder(_shear_ductile_steel, _drift_ductile_steel),
    "SLF": CurveBuilder(_shear_cfs, _drift_cfs),
    "CFS1": CurveBuilder(_shear_cfs, _drift_cfs),
    "CFS2": CurveBuilder(_shear_cfs, _drift_cfs),
    "CMF": CurveBuilder(_shear_ductile_cmf, _drift_ductile_cmf),
    "PCF2": CurveBuilder(_shear_ductile_cmf, _drift_ductile_cmf),
    "CSW": CurveBuilder(_shear_sw, _drift_sw),
    "PCW": CurveBuilder(_shear_sw, _drift_sw),
    "PCF1": CurveBuilder(_shear_sw, _drift_sw),
    "SCW": CurveBuilder(_shear_sw, _drift_sw),
}


def _get_builder(lfrs) -> CurveBuilder:
    """
    Look up builder by LFRS label
    fall back to default if unsupported."""
    key = str(lfrs).strip().upper() if lfrs is not None else None
    return CAPACITY_BUILDERS.get(key, DEFAULT_BUILDER)


def build_capacity_curves(lfrs, Vyj, Vpj, c, uyj, upj, uuj, upcj):
    """
    Build all four stacked capacity curves for one building.

    lfrs : tuple (lfrs_dir0, lfrs_dir1)
    Vyj, Vpj, c, uyj, upj, upcj, ... : tuples of per-direction arrays
    W_j : story weights (kN), same for both directions
    Returns dict of 4 tuples, each (dir0_array, dir1_array).
    """
    out = {"story_normalized_shear": [], "story_forces_kN": [],
           "story_drift_ratio_capacity": [], "story_drift_capacity_m": []}

    for d in range(2):
        b = _get_builder(lfrs[d])
        shear = b.shear(Vyj[d], Vpj[d], c[d])
        out["story_normalized_shear"].append(shear)
        out["story_drift_ratio_capacity"].append(b.drift(uyj[d], upj[d], uuj[d], upcj[d],
                                                         Vyj[d], Vpj[d], c[d]))

    return {k: tuple(v) for k, v in out.items()}
