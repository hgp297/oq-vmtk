'''
Estimation of properties from historical NBCC as well as the 
Level 3 Seismic Evaluation Guidelines.

'''
import numpy as np
import pandas as pd
from openquake.vmtk.units import units
from scipy.interpolate import RegularGridInterpolator

NBCC_YEARS = np.array([1941, 1953, 1960, 1965, 1970, 1975, 1977, 1980, 1985, 1990, 1995, 2005, 2010, 2015, 2020, 2025])

def determine_effective_nbcc_year(original_year_series, seismic_upgrade_year_series):
    '''
    Determine effective code year for the building. It is the later of
    original code-year of construction or code-year of a renovation
    that contains significant seismic upgrades, rounded down to the 
    latest NBCC edition of the time.

    Pre-code buildings are assumed to meet the requirements of the 1941 NBCC edition.

    Parameters
    ----------
    original_year_series: pd.Series
        Code year of the original building construction

    seismic_upgrade_year_series: pd.Series
        Actual year of latest major upgrade

    Returns
    -------
    pd.Series of the effective code year.
    '''
    nbcc_code_year_original = original_year_series.fillna(1941)
    seismic_upgrade_year = seismic_upgrade_year_series.fillna(1941)

    effective_construction_year = np.maximum(nbcc_code_year_original, seismic_upgrade_year)

    def latest_code_year(effective_construction_year: pd.Series) -> pd.Series:
        idx = np.searchsorted(NBCC_YEARS, effective_construction_year.to_numpy(),
                              side='right')
        return pd.Series(
            NBCC_YEARS[idx-1],
            index=effective_construction_year.index,
        )

    return latest_code_year(effective_construction_year)

def determine_period(row):
    '''
    Function to estimate fundamental period of the building. According to
    the Level 3 Seismic Evaluation Guideline, the fundamental period is to 
    be calculated using the NBCC 2025 Equation. 

    Parameters
    ------------------
    row: pd.Series
        Current analysis building
    methodology_year: int
        Year of methodology of the distribution function.
    shear_field: str
        Field of the pd.Series representing the base shear to be distributed. 
        Default is the NBCC factored base shear, representing the SEG's
        best estimate of the design code of the time.
    weight_array: np.array
        Array of the weight distribution.
        Default is None, which will generate a generic distribution

    row["Seismic Force Resisting System in the North-South Direction"]: str
        modern-classification of the n-s lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Seismic Force Resisting System in the East-West Direction"]: str
        modern-classification of the e-w lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies
    
    row["Building Height (Total Height Above Ground (m) to Roof Slab)"]: numeric
        Building height in metres. If not provided, will be estimated with 3.5m stories. 

    row["Floors Above Grade"]: numeric
        number of stories above grade

    Returns
    tuple:
        (list, list) first two periods in the n-s and e-w directions

    '''
    number_of_stories = int(row["Floors Above Grade"])
    bldg_height = row["Building Height (Total Height Above Ground (m) to Roof Slab)"]*units.m
    lfrs_ns = row["Seismic Force Resisting System in the North-South Direction"]
    lfrs_ew = row["Seismic Force Resisting System in the East-West Direction"]

    # estimate building height array
    if np.isnan(bldg_height):
        h_n = 3.5 * units.m * number_of_stories
    else:
        h_n = bldg_height * units.m
        h_ix = h_n / number_of_stories

    def calc_periods(lfrs, h_n):
        return determine_period_post_1995(lfrs=lfrs, h_n=h_n)

    T_ns = calc_periods(lfrs_ns, h_n)
    T_ew = calc_periods(lfrs_ew, h_n)

    return T_ns, T_ew


def determine_code_strength(row, **kwargs):
    '''
    Dispatcher function to redirect to the correct code calculation
    function based on the "effective_nbcc_year" function.

    Dispatcher is located at global variable NBCC_VS_CALCULATORS

    Returns
    -------

    tuple: (vs_ns, vs_ew)
    '''
    try:
        return NBCC_VS_CALCULATORS[row["effective_nbcc_year"]](row, **kwargs)
    except KeyError:
        raise ValueError(
            f"Unsupported code year: {row['effective_nbcc_year']}"
        )

def get_code_bins(lfrs):
    '''
    Definitions for pre-, low-, and moderate code match that 
    of the 2025 UBC Seismic Risk study performed by Arup. Definitions
    of benchmark high-code are set by the Seismic Evaluation Guidelines

    Parameters
    --------------------
    lfrs: string
        modern-classification of the lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    Returns
    --------------------
    list:
        bins for the years of "low-code", "moderate-code", and "high-code"
        classification cutoffs.
    '''
    concrete_buildings = ["SCW", "CMF", "CSW", "CIW", "PCW", "PCF1", "PCF2"]
    lfrs_benchmark = lfrs.map(BENCHMARK_YEAR)
    concrete_mask = lfrs.isin(concrete_buildings)

    return [
        1970,
        np.where(concrete_mask, 1985, 1992),
        lfrs_benchmark+1
    ]
        

    
# correspond to worst-case load factor for lateral E loads
def factor_lateral_earthquake_load(row):
    '''
    Return the equivalent limit-state design load factor for the LFRS depending on the code-year.
    
    Params
    --------
    row['effective_nbcc_year']: int
        year of code design
    
    row["Seismic Force Resisting System in the North-South Direction"]: str
        modern-classification of the n-s lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Seismic Force Resisting System in the East-West Direction"]: str
        modern-classification of the e-w lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies
    
    row['original_nbcc_unfactored_Vd']: tuple
        Unfactored base shear calculated from nbcc.determine_code_strength
        (vs_ns, vs_ew)

    Returns
    -------
    tuple
        Factored base shear in each direction (ns, ew)
    '''

    code_year = row["effective_nbcc_year"]
    vs_ns, vs_ew = row["original_nbcc_unfactored_Vd"]
    lfrs_ns = row["Seismic Force Resisting System in the North-South Direction"]
    lfrs_ew = row["Seismic Force Resisting System in the East-West Direction"]

    def lookup_load_factor(lfrs):
        is_concrete = lfrs in ['SCW', 'CMF', 'CSW', 'CIW', 'PCW', 'PCF1', 'PCF2']
        return {
            1941: 2.0, # working stress design, estimated from reinforcing steel stress limited to 50% of yield 
            1953: 2.0, # working stress design
            1960: 2.0, # working stress design
            1965: 1.35 if is_concrete else 2.0, # ultimate strength design allowed as alternative, based on ACI 1963
            1970: 1.80 if is_concrete else 2.0, # ultimate strength design allowed as alternative, based on ACI 1963
            1975: 1.80 if is_concrete else 1.50, # ultimate strength from concrete CSA allowed (1.8 worst), but limit state introduced
            1977: 1.80 if is_concrete else 1.50, # limit state design
            1980: 1.80 if is_concrete else 1.50, # limit state design
            1985: 1.50, # CSA concrete adopts limit state design in 1984
            1990: 1.0, # reduced load factor to acknowledge extreme event
            1995: 1.0, 
            2005: 1.0, 
            2010: 1.0, 
            2015: 1.0, 
            2020: 1.0,
            2025: 1.0
        }[code_year]

    load_factor_ns = lookup_load_factor(lfrs_ns)
    load_factor_ew = lookup_load_factor(lfrs_ew)


    return load_factor_ns * vs_ns, load_factor_ew * vs_ew

def adjust_base_shear_capacity_SEG(row):
    '''
    Return estimated minimum base shear capacity based on the 2025 Level 3
    Seismic Evaluation Guidelines. This accounts for the variety of "load
    factor" methods (ultimate, limit states) in Canada for previous versions.
    
    Params
    --------
    row['effective_nbcc_year']: int
        year of code design
    
    row["Seismic Force Resisting System in the North-South Direction"]: str
        modern-classification of the n-s lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Seismic Force Resisting System in the East-West Direction"]: str
        modern-classification of the e-w lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies
    
    row['original_nbcc_unfactored_Vd']: tuple
        Unfactored base shear calculated from nbcc.determine_code_strength
        (vs_ns, vs_ew)

    Returns
    -------
    tuple
        Factored base shear in each direction (ns, ew)
    '''

    code_year = row["effective_nbcc_year"]
    vs_ns, vs_ew = row["original_nbcc_unfactored_Vd"]
    lfrs_ns = row["Seismic Force Resisting System in the North-South Direction"]
    lfrs_ew = row["Seismic Force Resisting System in the East-West Direction"]

    def adjusted_earthquake_load_factor(lfrs):
        is_ultimate_design = lfrs in ['SCW', 'CMF', 'CSW', 'CIW', 'PCW', 'PCF1', 'PCF2']
        return {
            1941: 1.0, # working stress design, estimated from reinforcing steel stress limited to 50% of yield 
            1953: 1.0, # working stress design
            1960: 1.0, # working stress design
            1965: 1.35, # SEG does not yet split for ultimate vs. limit state
            1970: 1.35,
            1975: 1.35 if is_ultimate_design else 1.05, # ultimate strength from concrete CSA allowed (1.8 worst), but limit state introduced
            1977: 1.35 if is_ultimate_design else 1.05, # limit state design
            1980: 1.35 if is_ultimate_design else 1.05, # limit state design
            1985: 1.35 if is_ultimate_design else 1.05, # limit state design
            1990: 1.0, # reduced load factor to acknowledge extreme event
            1995: 1.0, 
            2005: 1.0, 
            2010: 1.0, 
            2015: 1.0, 
            2020: 1.0,
            2025: 1.0
        }[code_year]

    alpha_q_ns = adjusted_earthquake_load_factor(lfrs_ns)
    alpha_q_ew = adjusted_earthquake_load_factor(lfrs_ew)


    return alpha_q_ns * vs_ns, alpha_q_ew * vs_ew

def vs_nbcc_1941(row, seismic_hazard_params):
    '''
    Calculate the lateral force coefficient based on NBC1941, as outlined
    in Appendix H.

    The coefficient is NOT yet multiplied with the building weight. Seismic
    weight W is intended to be 1.0D + 0.5L.

    In Table 1, the highest controlling coefficient is if the system is 
    a bearing wall, followed by conditions based on soil bearing strength.

    The strength here is taken as the controlling condition between the two.

    Parameters
    ----------
    row: pd.Series
        row of the inventory df

    row["Site Class"]: str
        modern-assessed site class of the building

    row["Seismic Force Resisting System in the North-South Direction"]: str
        modern-classification of the n-s lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Seismic Force Resisting System in the East-West Direction"]: str
        modern-classification of the e-w lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    seismic_hazard_params: dict
        This edition of NBCC does not recognize location-specific hazard

    Returns
    -------
    vs_ns: float
        Lateral force coefficient in the n-s
    vs_ew: float
        Lateral force coefficient in the e-w
    '''
    site_class = row["Site Class"]
    lfrs_ns = row["Seismic Force Resisting System in the North-South Direction"]
    lfrs_ew = row["Seismic Force Resisting System in the East-West Direction"]

    # 1941 "weak-site" cut-off: 2000 psf soil bearing strength (q_u)
    # rough Coduto estimation: q_u ~ 6 s_u, s_u = undrained shear strength
    # s_u cutoff = 2000/6 = 333 psf = 16 kPa
    # only site with s_u < 40 kPa is F
    stronger_soil_bearing_sites = ['A', 'B', 'C', 'D', 'E']

    # assume that all "frame-with-wall" system have a non load-bearing wall
    bearing_wall_systems = ['CSW', 'PCW', 'RML', 'RMC', 'URM']

    # Use Table H1 with item corresponding to 
    # "the building as a whole"
    def flowchart_1941(lfrs, site_class):
        if site_class not in stronger_soil_bearing_sites:
            return 0.04
        else:
            return 0.02
            
    '''
    def flowchart_1941(lfrs, site_class):
        if lfrs in bearing_wall_systems:
            return 0.05
        elif site_class not in stronger_soil_bearing_sites:
            return 0.04
        else:
            return 0.02
    '''
    vs_ns = flowchart_1941(lfrs_ns, site_class)
    vs_ew = flowchart_1941(lfrs_ew, site_class)

    # unsupported features:
    # cantilever-style walls
    # ornamentations
    # towers, tanks, chimneys, smokestacks, penthouses

    return np.float64(vs_ns), np.float64(vs_ew)

def vs_nbcc_1953(row, seismic_hazard_params):
    '''
    Calculate the lateral force coefficient based on NBC1953, as outlined
    in Section 4.1.2.9

    The coefficient is NOT yet multiplied with the building weight. Vs is taken
    as the sum of floor forces calculated using Table 4.1.2.

    Parameters
    ----------
    row: pd.Series
        row of the inventory df

    row["Seismic Force Resisting System in the North-South Direction"]: str
        modern-classification of the n-s lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Seismic Force Resisting System in the East-West Direction"]: str
        modern-classification of the e-w lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Floors Above Grade"]: numeric
        number of stories above grade

    seismic_hazard_params: dict
        1953 used seismic zone as the location-specific parameter.
        Thus, the hazard dictionary
            requires the following inputs:
        
        seismic_zone: int
            Seismic Zone as determined by the 1952 map in Sec 2.11 of NBC 1953
            Zone 3: Western British Columbia (Victoria, Vancouver), St. Lawrence River
            Valley (Quebec City, Montreal), Ottawa River Valley (Ottawa)
            Zone 2: Maritime (Eastern Newfoundland, New Brunswick, Nova Scotia, PEI)
            Canadian Rockies (Calgary)
            Zone 1: Prairies (Edmonton, Saskatoon, Winnipeg, Regina), Great Lakes 
            (Toronto GTA)
            Zone 0: Upper Ontario, Upper Quebec

    TODO: determine zone from lat lon

    Returns
    -------
    vs_ns: float
        Lateral force coefficient in the n-s
    vs_ew: float
        Lateral force coefficient in the e-w
    '''
    number_of_stories = row["Floors Above Grade"]
    lfrs_ns = row["Seismic Force Resisting System in the North-South Direction"]
    lfrs_ew = row["Seismic Force Resisting System in the East-West Direction"]
    
    seismic_zone = seismic_hazard_params['seismic_zone']

    '''
    # assume that all "frame-with-wall" system have a non load-bearing wall
    # bearing walls, non-bearing walls, free-standing masonry walls
    bearing_wall_systems = ['CSW', 'PCW', 'RML', 'RMC', 'URM',
                            'SCW', 'SIW', 'CIW', 'CFS1', 'CFS2']

    def flowchart_1953(lfrs, number_of_stories):
        if lfrs in bearing_wall_systems:
            return 0.05
        else:
            stories = np.arange(0, number_of_stories)
            return np.sum(0.15/(stories+4.5))
    '''

    # Use Table 4.1.2 with item corresponding to 
    # "the structure as a whole", meaning that 
    # N is the number of stories in total

    def flowchart_1953(lfrs, number_of_stories):
        return 0.15/(number_of_stories+4.5)
    
    vs_ns = flowchart_1953(lfrs_ns, number_of_stories)
    vs_ew = flowchart_1953(lfrs_ew, number_of_stories)

    # unsupported features:
    # cantilever-style walls
    # ornamentations
    # towers, tanks, chimneys, smokestacks, penthouses

    if seismic_zone == 3:
        return np.float64(4*vs_ns), np.float64(4*vs_ew)
    elif seismic_zone == 2:
        return np.float64(2*vs_ns), np.float64(2*vs_ew)
    elif seismic_zone == 1:
        return np.float64(vs_ns), np.float64(vs_ew)
    else:
        return np.float64(0.0), np.float64(0.0)

def vs_nbcc_1965(row, seismic_hazard_params):
    '''
    Calculate the lateral force coefficient based on NBC1965, as outlined
    in Section 4.1.3.15.

    The coefficient is NOT yet multiplied with the building weight. Vs is taken
    from Sentence (4) of the section above.

    Mitchell et al. (2010) states that W is 1.0D+ 0.25S+ storage L. However,
    concrete structures allowed for load factors with 
    U = 1.35*(D + L + E)

    Distribution is later available in the same section.

    Parameters
    ----------
    row: pd.Series
        row of the inventory df

    row["Seismic Force Resisting System in the North-South Direction"]: str
        modern-classification of the n-s lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Seismic Force Resisting System in the East-West Direction"]: str
        modern-classification of the e-w lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Floors Above Grade"]: numeric
        number of stories above grade

    row['"Original" Building Importance Factor Ie']
        Importance factor assigned to the building in its original design

    row["Site Class"]: str
        modern-assessed site class of the building

    seismic_hazard_params: dict
        The hazard dictionary requires the following inputs:
        
        seismic_zone: int
            Seismic Zone as determined by the 1952 map in Sec 2.11 of NBC 1953
            Zone 3: Western British Columbia (Victoria, Vancouver), St. Lawrence River
            Valley (Quebec City, Montreal), Ottawa River Valley (Ottawa)
            Zone 2: Maritime (Eastern Newfoundland, New Brunswick, Nova Scotia, PEI)
            Canadian Rockies (Calgary)
            Zone 1: Prairies (Edmonton, Saskatoon, Winnipeg, Regina), Great Lakes 
            (Toronto GTA)
            Zone 0: Upper Ontario, Upper Quebec

    Returns
    -------
    vs_ns: float
        Lateral force coefficient in the n-s
    vs_ew: float
        Lateral force coefficient in the e-w
    '''
    number_of_stories = row["Floors Above Grade"]
    importance_factor = row['"Original" Building Importance Factor Ie']
    site_class = row["Site Class"]
    lfrs_ns = row["Seismic Force Resisting System in the North-South Direction"]
    lfrs_ew = row["Seismic Force Resisting System in the East-West Direction"]

    seismic_zone = seismic_hazard_params['seismic_zone']

    # "buildings framed in steel or reinforced concrete with moment resisting
    # connections, with floors sufficiently strong and stiff to distribute 
    # lateral forces among structural elements of varying flexibility, and
    # in which the frame alone is able to carry 50 per cent off the design shears
    # or in which shear walls are adequately reinforced to carry design shear
    # forces in a ductile fashion"

    # 1965 Foundation factor depends on "compressible" soil
    compressible_soil_sites = ['E', 'F']

    def flowchart_1965(lfrs, site_class):

        C_factor = C_TABLE_1965(lfrs)

        I_factor = importance_factor

        if site_class in compressible_soil_sites:
            F_factor = 1.5
        else:
            F_factor = 1.0

        S_factor = 0.25/(9+number_of_stories)

        return C_factor * I_factor * F_factor * S_factor

    vs_ns = flowchart_1965(lfrs_ns, site_class)
    vs_ew = flowchart_1965(lfrs_ew, site_class)

    # unsupported features:
    # cantilever-style walls
    # ornamentations
    # towers, tanks, chimneys, smokestacks, penthouses

    # in 1965, this was renamed to R factor
    if seismic_zone == 3:
        return np.float64(4*vs_ns), np.float64(4*vs_ew)
    elif seismic_zone == 2:
        return np.float64(2*vs_ns), np.float64(2*vs_ew)
    elif seismic_zone == 1:
        return np.float64(vs_ns), np.float64(vs_ew)
    else:
        return np.float64(0.0), np.float64(0.0)

    # TODO: 1965 NBCC required that if the building had significant
    # torsional irregularity, design computed torsional moment would
    # be doubled

def vs_nbcc_1970(row, seismic_hazard_params):
    '''
    Calculate the lateral force coefficient based on NBC1970, as outlined
    in Section 4.1.7.

    The coefficient is NOT yet multiplied with the building weight. Vs is taken
    from Sentence (4) of the section above.

    Mitchell et al. (2010) states that W is 1.0D+ 0.25S+ storage L. However,
    concrete structures allowed for load factors with 
    U = 1.35*(D + L + E)

    Distribution is later available in the same section.

    Parameters
    ----------
    row: pd.Series
        row of the inventory df

    row["Seismic Force Resisting System in the North-South Direction"]: str
        modern-classification of the n-s lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Seismic Force Resisting System in the East-West Direction"]: str
        modern-classification of the e-w lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Ground Floor Plan Area (sq.m.)"]: numeric
        Ground floor plan area in square metres, to identify dimension length
    
    row["Building Height (Total Height Above Ground (m) to Roof Slab)"]: numeric
        Building height in metres. If not provided, will be estimated with 3.5m stories. 

    row["Floors Above Grade"]: numeric
        number of stories above grade

    row['"Original" Building Importance Factor Ie']
        Importance factor assigned to the building in its original design

    row["Site Class"]: str
        modern-assessed site class of the building

    seismic_hazard_params: dict
        The hazard dictionary requires the following inputs:
        
        seismic_zone: int
        Seismic region as determined by the map in the Table of Climactic Data of NBC 1970
            Zone 3: Western British Columbia (Victoria, Vancouver), upper St. Lawrence River
            Valley (Quebec City)
            Zone 2: Maritime (Eastern Newfoundland, New Brunswick, Nova Scotia, PEI)
            Ottawa River Valley (Ottawa), lower St. Lawrence River (Montreal)
            Zone 1:  Great Lakes (Toronto GTA), Upper Quebec, Eastern British Columbia  
            Zone 0: Upper Ontario, Canadian Rockies (Calgary), Prairies (Edmonton, 
            Saskatoon, Winnipeg, Regina),

    Returns
    -------
    vs_ns: float
        Lateral force coefficient in the n-s
    vs_ew: float
        Lateral force coefficient in the e-w
    '''
    number_of_stories = row["Floors Above Grade"]
    importance_factor = row['"Original" Building Importance Factor Ie']
    site_class = row["Site Class"]
    plan_area = row["Ground Floor Plan Area (sq.m.)"]*units.m2
    bldg_height = row["Building Height (Total Height Above Ground (m) to Roof Slab)"]*units.m
    lfrs_ns = row["Seismic Force Resisting System in the North-South Direction"]
    lfrs_ew = row["Seismic Force Resisting System in the East-West Direction"]

    seismic_zone = seismic_hazard_params['seismic_zone']

    # TODO: temporarily stand-in square building
    D_ft = (plan_area**0.5)/units.ft # divide to go from m to ft

    # TODO: temporarily estimate bldg_height if not available
    # estimate as 3.5m stories
    if np.isnan(bldg_height):
        h_n_ft = 3.5/units.ft * number_of_stories
    else:
        h_n_ft = bldg_height/units.ft

    # "A ductile moment-resisting space frame is a space frame that is designed to resist
    # all the specified seismic forces and that, in addition, has adequate ductility or
    # energy-absorptive capacity."
    # TODO: request review on this
    # ductile systems weren't provided until 1973
    ductile_moment_frames = ["SMF", "CMF"]

    # consisting of a complete ductile moment resisting space frame and shear walls
    # 1) resist total lateral force in accordance with their rigidity
    # 2) shear walls resist total lateral force independent of ductile MF
    # 3) MF resist at least 25% of required lateral force
    dual_systems = ["SCW", "PCF1", "CFS1"]

    # 1970 Foundation factor depends on "compressible" soil
    compressible_soil_sites = ['E', 'F']

    # D_ft could change per-direction
    def flowchart_1970(lfrs, D_ft):
        K_factor = K_TABLE_1970(lfrs)

        if lfrs in ductile_moment_frames:
            T_period = 0.1 * number_of_stories
        elif lfrs in dual_systems:
            T_period = 0.05*h_n_ft/(D_ft**0.5)
        else:
            T_period = 0.05*h_n_ft/(D_ft**0.5)

        if number_of_stories <= 3:
            C_factor = 0.10
        else:
            C_factor = np.minimum(0.05/(T_period**(1/3)), 0.10)

        I_factor = importance_factor

        if site_class in compressible_soil_sites:
            F_factor = 1.5
        else:
            F_factor = 1.0

        return K_factor * C_factor * I_factor * F_factor / 4

    
    vs_ns = flowchart_1970(lfrs_ns, D_ft)
    vs_ew = flowchart_1970(lfrs_ew, D_ft)

    # unsupported features:
    # cantilever-style walls
    # ornamentations
    # towers, tanks, chimneys, smokestacks, penthouses
    # floors and roofs acting as diaphragms

    # in 1965, this was renamed to R factor
    if seismic_zone == 3:
        return np.float64(4*vs_ns), np.float64(4*vs_ew)
    elif seismic_zone == 2:
        return np.float64(2*vs_ns), np.float64(2*vs_ew)
    elif seismic_zone == 1:
        return np.float64(vs_ns), np.float64(vs_ew)
    else:
        return np.float64(0.0), np.float64(0.0)

'''
NOTE: 
walls designed in accordance with the older codes (1975 to 1995)
are likely to lack sufficient shear capacity over their height as
well as flexural strength above the plastic hinge region.

Ghorbanirenani et al. (2009)
'''
def vs_nbcc_1975(row, seismic_hazard_params):
    '''
    Calculate the lateral force coefficient based on NBC1975, as outlined
    in Section 4.1.9.

    The coefficient is NOT yet multiplied with the building weight. Vs is taken
    from Sentence (4) of the section above.

    Mitchell et al. (2010) states that W is 1.0D+ 0.25S+ storage L. However,
    concrete structures allowed for load factors with 
    U = 1.35*(D + L + E)

    Distribution is later available in the same section.

    This function is also used for the 1980 edition, as the provisions were
    nearly identical except for an update to SI units and the S_factor.

    Parameters
    ----------
    row: pd.Series
        row of the inventory df

    row["Seismic Force Resisting System in the North-South Direction"]: str
        modern-classification of the n-s lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Seismic Force Resisting System in the East-West Direction"]: str
        modern-classification of the e-w lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Ground Floor Plan Area (sq.m.)"]: numeric
        Ground floor plan area in square metres, to identify dimension length
    
    row["Building Height (Total Height Above Ground (m) to Roof Slab)"]: numeric
        Building height in metres. If not provided, will be estimated with 3.5m stories. 

    row["Floors Above Grade"]: numeric
        number of stories above grade

    row['"Original" Building Importance Factor Ie']
        Importance factor assigned to the building in its original design

    row["Site Class"]: str
        modern-assessed site class of the building

    row["effective_code_year"]: int
        Effective NBCC year. If = 1980, redirect to correct S_factor

    seismic_hazard_params: dict
        The hazard dictionary requires the following inputs:
        
        seismic_zone: int
        Seismic region as determined by the map in the Table of Climactic Data of NBC 1970
            Zone 3: Western British Columbia (Victoria, Vancouver), upper St. Lawrence River
            Valley (Quebec City)
            Zone 2: Maritime (Eastern Newfoundland, New Brunswick, Nova Scotia, PEI)
            Ottawa River Valley (Ottawa), lower St. Lawrence River (Montreal)
            Zone 1:  Great Lakes (Toronto GTA), Upper Quebec, Eastern British Columbia  
            Zone 0: Upper Ontario, Canadian Rockies (Calgary), Prairies (Edmonton, 
            Saskatoon, Winnipeg, Regina)

    Returns
    -------
    vs_ns: float
        Lateral force coefficient in the n-s
    vs_ew: float
        Lateral force coefficient in the e-w
    '''
    number_of_stories = row["Floors Above Grade"]
    importance_factor = row['"Original" Building Importance Factor Ie']
    site_class = row["Site Class"]
    plan_area = row["Ground Floor Plan Area (sq.m.)"]*units.m2
    bldg_height = row["Building Height (Total Height Above Ground (m) to Roof Slab)"]*units.m
    lfrs_ns = row["Seismic Force Resisting System in the North-South Direction"]
    lfrs_ew = row["Seismic Force Resisting System in the East-West Direction"]

    # redirect to 1980 S_factor update
    code_year = row["effective_nbcc_year"]

    seismic_zone = seismic_hazard_params['seismic_zone']

    # TODO: temporarily stand-in square building
    D_ft = (plan_area**0.5)/units.ft # divide to go from m to ft

    # TODO: temporarily estimate bldg_height if not available
    # estimate as 3.5m stories
    if np.isnan(bldg_height):
        h_n_ft = 3.5/units.ft * number_of_stories
    else:
        h_n_ft = bldg_height/units.ft

    
    # in 1975, a design ground acceleration was assigned to each zone
    climactic_table_A = {
        0: 0.0,
        1: 0.02,
        2: 0.04, 
        3: 0.08
    }

    # "A ductile moment-resisting space frame is a space frame that is designed to resist
    # all the specified seismic forces and that, in addition, has adequate ductility or
    # energy-absorptive capacity."
    ductile_moment_frames = ["SMF", "CMF"]

    # TODO: request review on this
    # rock and very stiff soils
    rock_sites = ['A', 'B', 'C']
    # stiff soils
    soft_or_compact_soil_sites = ['D', 'E']
    # 1970 Foundation factor depends on "compressible" soil
    very_soft_soil_sites = ['F']

    # D_ft could change per-direction
    def flowchart_1975(lfrs, D_ft):
        # acceleration
        A_factor = climactic_table_A[seismic_zone]

        # system-specific force reductions
        K_factor = K_TABLE_1975(lfrs, number_of_stories)

        # period estimation
        if lfrs in ductile_moment_frames:
            T_period = 0.1 * number_of_stories
        else:
            T_period = 0.05*h_n_ft/(D_ft**0.5)

        # seismic response factor
        if int(code_year) == 1980:
            S_factor = np.minimum(0.5/(T_period**(1/2)), 1.0)
        else:
            S_factor = np.minimum(0.5/(T_period**(1/3)), 1.0)
        

        # importance factor
        I_factor = importance_factor

        # soil factor
        if site_class in very_soft_soil_sites:
            F_factor = 1.5
        elif site_class in soft_or_compact_soil_sites:
            F_factor = 1.3
        elif site_class in rock_sites:
            F_factor = 1.0
        else:
            F_factor = 1.5

        return A_factor * S_factor * K_factor * I_factor * F_factor

    
    vs_ns = flowchart_1975(lfrs_ns, D_ft)
    vs_ew = flowchart_1975(lfrs_ew, D_ft)

    # unsupported features:
    # cantilever-style walls
    # ornamentations
    # towers, tanks, chimneys, smokestacks, penthouses
    # floors and roofs acting as diaphragms

    # in 1975, a design ground acceleration was assigned to each zone
    # A*S was calibrated to be about 20% less than previous 1/4*R*C
    return vs_ns, vs_ew

    # TODO: 1975 NBCC required that if the building had significant
    # torsional irregularity, design computed torsional moment would
    # be doubled    

def vs_nbcc_1985(row, seismic_hazard_params):
    '''
    Calculate the lateral force coefficient based on NBC1985, as outlined
    in Section 4.1.9. 

    The coefficient is NOT yet multiplied with the building weight. 

    Distribution is later available in the same section.

    Parameters
    ----------
    row: pd.Series
        row of the inventory df

    row["Seismic Force Resisting System in the North-South Direction"]: str
        modern-classification of the n-s lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Seismic Force Resisting System in the East-West Direction"]: str
        modern-classification of the e-w lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Ground Floor Plan Area (sq.m.)"]: numeric
        Ground floor plan area in square metres, to identify dimension length
    
    row["Building Height (Total Height Above Ground (m) to Roof Slab)"]: numeric
        Building height in metres. If not provided, will be estimated with 3.5m stories. 

    row["Floors Above Grade"]: numeric
        number of stories above grade

    row['"Original" Building Importance Factor Ie']
        Importance factor assigned to the building in its original design

    row["Site Class"]: str
        modern-assessed site class of the building

    seismic_hazard_params: Dictionary
        1985 NBCC introduced finer control for hazard, acknowledging differences in spectral
        shape and probabilistically aiming for a 495-year return period. Thus, the hazard dictionary
        requires the following inputs:
        - Za: seismic acceleration zone
        - Zv: seismic velocity zone
        - v: zonal velocity ratio
        (zonal acceleration ratio is implicitly used in S factor)


    Returns
    -------
    vs_ns: float
        Lateral force coefficient in the n-s
    vs_ew: float
        Lateral force coefficient in the e-w
    '''
    number_of_stories = row["Floors Above Grade"]
    importance_factor = row['"Original" Building Importance Factor Ie']
    site_class = row["Site Class"]
    plan_area = row["Ground Floor Plan Area (sq.m.)"]*units.m2
    bldg_height = row["Building Height (Total Height Above Ground (m) to Roof Slab)"]*units.m
    lfrs_ns = row["Seismic Force Resisting System in the North-South Direction"]
    lfrs_ew = row["Seismic Force Resisting System in the East-West Direction"]

    Za = seismic_hazard_params['Za']
    Zv = seismic_hazard_params['Zv']
    v_ratio = seismic_hazard_params['v_ratio']

    # exception statement
    if (Zv == 0) and (Za > 0):
        Zv = 1
        v_ratio = 0.05


    # TODO: temporarily estimator
    # if moment frame, infill, masonry, dual system
    # D_s_ft is length of the building (square assumption)
    # else, roughly estimate D_s as about 1/6 of the length
    # length of just the LFRS
    whole_length_systems = ["SMF", "CMF", "SCW", "PCF1", "CFS1", "CFS2",
                            "RML", "RMC", "SLF", "PCF2", "URM", "SIW", "CIW"]

    if lfrs_ns in whole_length_systems:
        D_s_ns = (plan_area**0.5)
    else:
        D_s_ns = (plan_area**0.5)/6

    if lfrs_ew in whole_length_systems:
        D_s_ew = (plan_area**0.5)
    else:
        D_s_ew = (plan_area**0.5)/6

    # TODO: temporarily estimate bldg_height if not available
    # estimate as 3.5 m stories
    if np.isnan(bldg_height):
        h_n = 3.5*units.m * number_of_stories
    else:
        h_n = bldg_height * units.m

    # "A ductile moment-resisting space frame is a space frame that is designed to resist
    # all the specified seismic forces and that, in addition, has adequate ductility or
    # energy-absorptive capacity."
    ductile_moment_frames = ["SMF", "CMF"]


    # TODO: request review on this
    # rock and very stiff soils
    rock_sites = ['A', 'B', 'C']
    # stiff soils
    soft_or_compact_soil_sites = ['D', 'E']
    # 1970 Foundation factor depends on "compressible" soil
    very_soft_soil_sites = ['F']

    # D_s could change per-direction
    def flowchart_1985(lfrs, D_s):

        # system-specific force reductions
        K_factor = K_TABLE_1975(lfrs, number_of_stories)

        # period estimation
        if lfrs in ductile_moment_frames:
            T_period = 0.1 * number_of_stories
        else:
            T_period = 0.09*h_n/(D_s**0.5)

        # Table 4.1.9.A for S_factor calculation
        # seismic response factor
        if T_period < 0.25:
            if Za/Zv > 1.0:
                S_factor = 0.62
            elif Za/Zv == 1.0:
                S_factor = 0.44
            else:
                S_factor = 0.31
        elif (T_period < 0.50) and (T_period >= 0.25):
            if Za/Zv > 1.0:
                S_factor = 0.62 - 1.23*(T_period - 0.25)
            elif Za/Zv == 1.0:
                S_factor = 0.44 - 0.50*(T_period - 0.25)
            else:
                S_factor = 0.31
        else:
            S_factor = 0.22/(T_period**0.5)

        # importance factor
        I_factor = importance_factor

        # soil factor
        if site_class in very_soft_soil_sites:
            F_factor = 1.5
        elif site_class in soft_or_compact_soil_sites:
            F_factor = 1.3
        elif site_class in rock_sites:
            F_factor = 1.0
        else:
            F_factor = 1.5

        # Sentence 10
        F_times_S = F_factor * S_factor
        if Za <= Zv:
            F_times_S = np.minimum(F_times_S, 0.44)
        else:
            F_times_S = np.minimum(F_times_S, 0.62)

        return v_ratio * F_times_S * K_factor * I_factor

    
    vs_ns = flowchart_1985(lfrs_ns, D_s_ns)
    vs_ew = flowchart_1985(lfrs_ew, D_s_ew)

    # unsupported features:
    # cantilever-style walls
    # ornamentations
    # towers, tanks, chimneys, smokestacks, penthouses
    # floors and roofs acting as diaphragms

    return vs_ns, vs_ew

    # TODO: 1985 NBCC required that if the building had significant
    # torsional irregularity, dynamic analysis is required to design
    
def vs_nbcc_1990(row, seismic_hazard_params):
    '''
    Calculate the lateral force coefficient based on NBC1990, as outlined
    in Section 4.1.9. 

    The coefficient is NOT yet multiplied with the building weight. 

    Distribution is later available in the same section.

    Parameters
    ----------
    row: pd.Series
        row of the inventory df

    row["Seismic Force Resisting System in the North-South Direction"]: str
        modern-classification of the n-s lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Seismic Force Resisting System in the East-West Direction"]: str
        modern-classification of the e-w lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Ground Floor Plan Area (sq.m.)"]: numeric
        Ground floor plan area in square metres, to identify dimension length
    
    row["Building Height (Total Height Above Ground (m) to Roof Slab)"]: numeric
        Building height in metres. If not provided, will be estimated with 3.5m stories. 

    row["Floors Above Grade"]: numeric
        number of stories above grade

    row['"Original" Building Importance Factor Ie']
        Importance factor assigned to the building in its original design

    row["Site Class"]: str
        modern-assessed site class of the building

    seismic_hazard_params: Dictionary
        1990 NBCC introduced finer control for hazard, acknowledging differences in spectral
        shape and probabilistically aiming for a 495-year return period. Thus, the hazard dictionary
        requires the following inputs:
        - Za: seismic acceleration zone
        - Zv: seismic velocity zone
        - v: zonal velocity ratio
        (zonal acceleration ratio is implicitly used in S factor)

    Returns
    -------
    vs_ns: float
        Lateral force coefficient in the n-s
    vs_ew: float
        Lateral force coefficient in the e-w
    '''
    number_of_stories = row["Floors Above Grade"]
    importance_factor = row['"Original" Building Importance Factor Ie']
    site_class = row["Site Class"]
    plan_area = row["Ground Floor Plan Area (sq.m.)"]*units.m2
    bldg_height = row["Building Height (Total Height Above Ground (m) to Roof Slab)"]*units.m
    lfrs_ns = row["Seismic Force Resisting System in the North-South Direction"]
    lfrs_ew = row["Seismic Force Resisting System in the East-West Direction"]
    
    Za = seismic_hazard_params['Za']
    Zv = seismic_hazard_params['Zv']
    v_ratio = seismic_hazard_params['v_ratio']

    # exception statement
    if (Zv == 0) and (Za > 0):
        Zv = 1
        v_ratio = 0.05


    # TODO: temporarily estimator
    # if moment frame, infill, masonry, dual system
    # D_s_ft is length of the building (square assumption)
    # else, roughly estimate D_s as about 1/6 of the length
    # length of just the LFRS
    whole_length_systems = ["SMF", "CMF", "SCW", "PCF1", "CFS1", "CFS2",
                            "RML", "RMC", "SLF", "PCF2", "URM", "SIW", "CIW"]

    if lfrs_ns in whole_length_systems:
        D_s_ns = (plan_area**0.5)
    else:
        D_s_ns = (plan_area**0.5)/6

    if lfrs_ew in whole_length_systems:
        D_s_ew = (plan_area**0.5)
    else:
        D_s_ew = (plan_area**0.5)/6

    # TODO: temporarily estimate bldg_height if not available
    # estimate as 3.5m stories
    if np.isnan(bldg_height):
        h_n = 3.5 * units.m * number_of_stories
    else:
        h_n = bldg_height * units.m

    # "A ductile moment-resisting space frame is a space frame that is designed to resist
    # all the specified seismic forces and that, in addition, has adequate ductility or
    # energy-absorptive capacity."
    # TODO: request review on this
    ductile_moment_frames = ["SMF", "CMF"]

    # TODO: request review on this
    # rock and very stiff soils
    rock_sites = ['A', 'B', 'C']
    # soil sites
    soft_or_compact_soil_sites = ['D', 'E']
    # very loose and loose coarse soils > 15m
    very_soft_soil_sites = ['F']

    flexure_controlled = number_of_stories > 3

    # D_s could change per-direction
    def flowchart_1990(lfrs, D_s):

        R_factor = R_TABLE_1990(lfrs, flexure_controlled)

        # period estimation
        if lfrs in ductile_moment_frames:
            T_period = 0.1 * number_of_stories
        else:
            T_period = 0.09*h_n/(D_s**0.5)

        # Table 4.1.9.A for S_factor calculation
        # seismic response factor
        if T_period < 0.25:
            if Za/Zv > 1.0:
                S_factor = 4.2
            elif Za/Zv == 1.0:
                S_factor = 3.0
            else:
                S_factor = 2.1
        elif (T_period < 0.50) and (T_period >= 0.25):
            if Za/Zv > 1.0:
                S_factor = 4.2 - 8.4*(T_period - 0.25)
            elif Za/Zv == 1.0:
                S_factor = 3.0 - 3.6*(T_period - 0.25)
            else:
                S_factor = 2.1
        else:
            S_factor = 1.5/(T_period**0.5)

        # importance factor
        I_factor = importance_factor

        # soil factor
        if site_class in very_soft_soil_sites:
            F_factor = 1.5
        elif site_class in soft_or_compact_soil_sites:
            F_factor = 1.3
        elif site_class in rock_sites:
            F_factor = 1.0
        else:
            F_factor = 2.0

        # Sentence 11
        F_times_S = F_factor * S_factor
        if Za <= Zv:
            F_times_S = np.minimum(F_times_S, 3.0)
        else:
            F_times_S = np.minimum(F_times_S, 4.2)

        # calibration factor (to previous code values)
        # meant to keep base shear consistent for buildings
        # constructed well with correct R
        U_factor = 0.6

        return v_ratio * F_times_S * I_factor / R_factor * U_factor

    
    vs_ns = flowchart_1990(lfrs_ns, D_s_ns)
    vs_ew = flowchart_1990(lfrs_ew, D_s_ew)

    # unsupported features:
    # cantilever-style walls
    # ornamentations
    # towers, tanks, chimneys, smokestacks, penthouses
    # floors and roofs acting as diaphragms

    return vs_ns, vs_ew

    # TODO: 1990 NBCC required that if the building had significant
    # torsional irregularity, dynamic analysis is required to design

    # TODO: 1990 NBCC also imposed design drift limits of 0.01 for post-disaster
    # and 0.02 for other buildings
    
def vs_nbcc_1995(row, seismic_hazard_params):
    '''
    Calculate the lateral force coefficient based on NBC1995, as outlined
    in Section 4.1.9. This edition is largely similar to NBC 1990, with
    a few minor changes in R factor classification and T_period estimation.

    The coefficient is NOT yet multiplied with the building weight. 

    Distribution is later available in the same section.

    Parameters
    ----------
    row: pd.Series
        row of the inventory df

    row["Seismic Force Resisting System in the North-South Direction"]: str
        modern-classification of the n-s lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Seismic Force Resisting System in the East-West Direction"]: str
        modern-classification of the e-w lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Ground Floor Plan Area (sq.m.)"]: numeric
        Ground floor plan area in square metres, to identify dimension length
    
    row["Building Height (Total Height Above Ground (m) to Roof Slab)"]: numeric
        Building height in metres. If not provided, will be estimated with 3.5m stories. 

    row["Floors Above Grade"]: numeric
        number of stories above grade

    row['"Original" Building Importance Factor Ie']
        Importance factor assigned to the building in its original design

    row["Site Class"]: str
        modern-assessed site class of the building

    seismic_hazard_params: Dictionary
        1990 NBCC introduced finer control for hazard, acknowledging differences in spectral
        shape and probabilistically aiming for a 495-year return period. Thus, the hazard dictionary
        requires the following inputs:
        - Za: seismic acceleration zone
        - Zv: seismic velocity zone
        - v: zonal velocity ratio
        (zonal acceleration ratio is implicitly used in S factor)

    Returns
    -------
    vs_ns: float
        Lateral force coefficient in the n-s
    vs_ew: float
        Lateral force coefficient in the e-w
    '''
    number_of_stories = row["Floors Above Grade"]
    importance_factor = row['"Original" Building Importance Factor Ie']
    site_class = row["Site Class"]
    plan_area = row["Ground Floor Plan Area (sq.m.)"]*units.m2
    bldg_height = row["Building Height (Total Height Above Ground (m) to Roof Slab)"]*units.m
    lfrs_ns = row["Seismic Force Resisting System in the North-South Direction"]
    lfrs_ew = row["Seismic Force Resisting System in the East-West Direction"]
    code_year = row["effective_nbcc_year"]
    
    Za = seismic_hazard_params['Za']
    Zv = seismic_hazard_params['Zv']
    v_ratio = seismic_hazard_params['v_ratio']

    # exception statement
    if (Zv == 0) and (Za > 0):
        Zv = 1
        v_ratio = 0.05


    # TODO: temporarily estimator
    # if moment frame, infill, masonry, dual system
    # D_s is length of the building (square assumption)
    # else, roughly estimate D_s as about 1/6 of the length
    # length of just the LFRS
    whole_length_systems = ["SMF", "CMF", "SCW", "PCF1", "CFS1", "CFS2",
                            "RML", "RMC", "SLF", "PCF2", "URM", "SIW", "CIW"]

    if lfrs_ns in whole_length_systems:
        D_s_ns = (plan_area**0.5)
    else:
        D_s_ns = (plan_area**0.5)/6

    if lfrs_ew in whole_length_systems:
        D_s_ew = (plan_area**0.5)
    else:
        D_s_ew = (plan_area**0.5)/6

    # TODO: temporarily estimate bldg_height if not available
    # estimate as 3.5m stories
    if np.isnan(bldg_height):
        h_n = 3.5 * units.m * number_of_stories
    else:
        h_n = bldg_height * units.m

    # TODO: request review on this
    # rock and very stiff soils
    rock_sites = ['A', 'B', 'C']
    # soil sites
    soft_or_compact_soil_sites = ['D', 'E']
    # very loose and loose coarse soils > 15m
    very_soft_soil_sites = ['F']

    flexure_controlled = number_of_stories > 3
    built_after_1995 = code_year >= 1995

    # D_s could change per-direction
    def flowchart_1995(lfrs, D_s):

        R_factor = R_TABLE_1990(lfrs, flexure_controlled, built_after_1995=built_after_1995)

        # period estimation, using the more detailed moment-frame
        if lfrs == 'SMF':
            T_period = 0.085*(h_n**0.75)
        elif lfrs == 'CMF':
            T_period = 0.075*(h_n**0.75)
        else:
            T_period = 0.09*h_n/(D_s**0.5)

        # Table 4.1.9.A for S_factor calculation
        # seismic response factor
        if T_period < 0.25:
            if Za/Zv > 1.0:
                S_factor = 4.2
            elif Za/Zv == 1.0:
                S_factor = 3.0
            else:
                S_factor = 2.1
        elif (T_period < 0.50) and (T_period >= 0.25):
            if Za/Zv > 1.0:
                S_factor = 4.2 - 8.4*(T_period - 0.25)
            elif Za/Zv == 1.0:
                S_factor = 3.0 - 3.6*(T_period - 0.25)
            else:
                S_factor = 2.1
        else:
            S_factor = 1.5/(T_period**0.5)

        # importance factor
        I_factor = importance_factor

        # soil factor
        if site_class in very_soft_soil_sites:
            F_factor = 1.5
        elif site_class in soft_or_compact_soil_sites:
            F_factor = 1.3
        elif site_class in rock_sites:
            F_factor = 1.0
        else:
            F_factor = 2.0

        # Sentence 11
        F_times_S = F_factor * S_factor
        if Za <= Zv:
            F_times_S = np.minimum(F_times_S, 3.0)
        else:
            F_times_S = np.minimum(F_times_S, 4.2)

        # calibration factor (to previous code values)
        # meant to keep base shear consistent for buildings
        # constructed well with correct R
        U_factor = 0.6

        return v_ratio * F_times_S * I_factor / R_factor * U_factor

    
    vs_ns = flowchart_1995(lfrs_ns, D_s_ns)
    vs_ew = flowchart_1995(lfrs_ew, D_s_ew)

    # unsupported features:
    # cantilever-style walls
    # ornamentations
    # towers, tanks, chimneys, smokestacks, penthouses
    # floors and roofs acting as diaphragms

    return vs_ns, vs_ew

    # TODO: 1995 NBCC required that if the building had significant
    # torsional irregularity, dynamic analysis is required to design

    # TODO: 1995 NBCC also imposed design drift limits of 0.01 for post-disaster
    # and 0.02 for other buildings

def vs_nbcc_2005(row, seismic_hazard_params):
    '''
    Calculate the lateral force coefficient based on NBC 2005, as outlined
    in Section 4.1.8.

    The coefficient is NOT yet multiplied with the building weight. 

    Distribution is later available in the same section.

    Parameters
    ----------
    row: pd.Series
        row of the inventory df

    row["Seismic Force Resisting System in the North-South Direction"]: str
        modern-classification of the n-s lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Seismic Force Resisting System in the East-West Direction"]: str
        modern-classification of the e-w lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Ground Floor Plan Area (sq.m.)"]: numeric
        Ground floor plan area in square metres, to identify dimension length
    
    row["Building Height (Total Height Above Ground (m) to Roof Slab)"]: numeric
        Building height in metres. If not provided, will be estimated with 3.5m stories. 

    row["Floors Above Grade"]: numeric
        number of stories above grade

    row['"Original" Building Importance Factor Ie']
        Importance factor assigned to the building in its original design

    row["Site Class"]: str
        modern-assessed site class of the building

    seismic_hazard_params: Dictionary
        2005 NBCC introduced a site-specific response spectral acceleration table
        meant to implement a uniform hazard spectrum aimed at a 2% in 50 year (2475 rp)
        hazard.
        - Sa_0p2, Sa_0p5, Sa_1p0, Sa_2p0: 5% damped Sa at periods
        - Sa_pga: peak ground acceleration

    Returns
    -------
    vs_ns: float
        Lateral force coefficient in the n-s
    vs_ew: float
        Lateral force coefficient in the e-w
    '''
    number_of_stories = row["Floors Above Grade"]
    importance_factor = row['"Original" Building Importance Factor Ie']
    site_class = row["Site Class"]
    bldg_height = row["Building Height (Total Height Above Ground (m) to Roof Slab)"]*units.m
    lfrs_ns = row["Seismic Force Resisting System in the North-South Direction"]
    lfrs_ew = row["Seismic Force Resisting System in the East-West Direction"]
    code_year = row["effective_nbcc_year"]

    Sa_0p2 = seismic_hazard_params['Sa_0p2']
    Sa_0p5 = seismic_hazard_params['Sa_0p5']
    Sa_1p0 = seismic_hazard_params['Sa_1p0']
    Sa_2p0 = seismic_hazard_params['Sa_2p0']
    Sa_pga = seismic_hazard_params['Sa_pga']

    # lookup Fa using Sa(0.2) and site class
    # site class F needs in-depth geotechnical study
    Fa = np.interp(
        Sa_0p2, FA_TABLE_2005.index, FA_TABLE_2005[site_class]
    )

    # lookup Fv using Sa(1.0) and site class
    Fv = np.interp(
        Sa_1p0, FV_TABLE_2005.index, FV_TABLE_2005[site_class]
    )


    # TODO: temporarily estimate bldg_height if not available
    # estimate as 3.5m stories
    if np.isnan(bldg_height):
        h_n = 3.5 * units.m * number_of_stories
    else:
        h_n = bldg_height * units.m

    # 1990 has an additional category for very soft fine-grained soils >15m
    # assumed that it wouldn't be here unless a flag is raised

    # SMF and SBF values are assuming ductile frames
    # a lesser, intermediate "nominal" ductility form is 
    # also available 

    # all frames not explicitly called out to be moment
    # frames are assumed to be a nominal ductility one
    # i.e. precast = nominal ductility RC frame
    built_after_1995 = code_year >= 1995

    # ductility
    # ductile coupled walls is assumed to be classified as "CSW", which is ductile shear wall as it is the more conservative one
    # it is assumed that RM construction 1995 and after are "nominal ductility"
    ductility_lookup = {
        "WLF-P9": 'conventional', # assuming CAN/CSA-O86.1-M compliant, conventional construction
        "WLF": 'conventional', # assuming CAN/CSA-O86.1-M compliant
        "WPB": 'moderate', # assuming CAN/CSA-O86.1-M compliant, moment-frame like
        "SMF": 'ductile', # ductile mrf
        "SBF": 'ductile', # ductile braced frame
        "SLF": 'ductile', # ductile mrf
        "SCW": 'moderate', # assuming wall controls, designed to moderate ductility
        "SIW": 'moderate', # assuming wall controls, designed to moderate ductility
        "CMF": 'ductile', # ductile mrf
        "CSW": 'ductile', # ductile shear wall
        "CIW": 'moderate', # assuming wall controls, designed to moderate ductility
        "PCW": 'moderate', # moderate ductility
        "PCF1": 'moderate', # assuming wall controls, designed to moderate ductility
        "PCF2": 'moderate', # nominal ductility frame
        "RML": 'moderate' if built_after_1995 else 'conventional', # "nominal ductility" RM depending on construction year
        "RMC": 'moderate' if built_after_1995 else 'conventional', # "nominal ductility" RM depending on construction year
        "URM": 'conventional',
        "CFS1":'conventional', # non-ductile steel frame assumed, other category
        "CFS2": 'conventional',
    }


    def flowchart_2005(lfrs):
        ductility_level = ductility_lookup[lfrs]
        R_d = R_D_TABLE(lfrs, ductility_level=ductility_level)
        R_o = R_O_TABLE(lfrs, ductility_level=ductility_level)

        # period estimation, using the more detailed moment-frame
        T_period = determine_period_post_1995(lfrs=lfrs, h_n=h_n)

        # 4.1.8.4 Sentence 6, design spectral acceleration
        T_anchor = np.array([0.2, 0.5, 1.0, 2.0, 4.0])
        S_T_functions = np.array([
            Fa * Sa_0p2,
            np.minimum(Fv * Sa_0p5, Fa * Sa_0p2),
            Fv * Sa_1p0,
            Fv * Sa_2p0,
            Fv * Sa_2p0 / 2
        ])

        S_Tperiod = np.interp(T_period, T_anchor, S_T_functions)
        S_0p2 = np.interp(0.2, T_anchor, S_T_functions)
        S_2p0 = np.interp(2.0, T_anchor, S_T_functions)

        # M_v Table 4.1.8.11
        def M_v_decision_tree(Sa_ratio, lfrs_name):
            '''
            Given Sa(0.2)/Sa(2.0) and the LFRS
            
            Return the two bound values for T_a < 1.0 and T_a > 2.0
            '''
            if Sa_ratio < 8.0:
                if lfrs_name in ['SMF', 'CMF']:
                    return np.array([1.0, 1.0])
                elif lfrs_name in ['SBF']:
                    return np.array([1.0, 1.0])
                else:
                    return np.array([1.0, 1.2])
            else:
                if lfrs_name in ['SMF', 'CMF']:
                    return np.array([1.0, 1.2])
                elif lfrs_name in ['SBF']:
                    return np.array([1.0, 1.5])
                else:
                    return np.array([1.0, 2.5])

        
        Mv_bounds = M_v_decision_tree(Sa_0p2/Sa_2p0, lfrs)
        M_v = np.interp(T_period, [1.0, 2.0], Mv_bounds)
        
        # importance factor
        I_factor = importance_factor

        V_b = S_Tperiod * M_v * I_factor / (R_d * R_o)
        V_min = S_2p0 * M_v * I_factor / (R_d * R_o)

        if R_d >= 1.5:
            V_max = 2/3* S_0p2 * I_factor / (R_d * R_o)
            V_b = np.minimum(V_b, V_max)

        return np.maximum(V_b, V_min)

    
    vs_ns = flowchart_2005(lfrs_ns)
    vs_ew = flowchart_2005(lfrs_ew)

    # unsupported features:
    # cantilever-style walls
    # ornamentations
    # towers, tanks, chimneys, smokestacks, penthouses
    # floors and roofs acting as diaphragms

    return vs_ns, vs_ew

    # TODO: 2005 NBCC simplified torsional effects 

    # TODO: 2005 NBCC has the same 0.01 0.02 drift limits, but
    # required that these be inelastic drift limits (elastic_dx*R_d*R_o/I_e)


def vs_nbcc_2010(row, seismic_hazard_params):
    '''
    Calculate the lateral force coefficient based on NBC 2010, as outlined
    in Section 4.1.8.

    The coefficient is NOT yet multiplied with the building weight. 

    Distribution is later available in the same section.

    Parameters
    ----------
    row: pd.Series
        row of the inventory df

    row["Seismic Force Resisting System in the North-South Direction"]: str
        modern-classification of the n-s lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Seismic Force Resisting System in the East-West Direction"]: str
        modern-classification of the e-w lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Ground Floor Plan Area (sq.m.)"]: numeric
        Ground floor plan area in square metres, to identify dimension length
    
    row["Building Height (Total Height Above Ground (m) to Roof Slab)"]: numeric
        Building height in metres. If not provided, will be estimated with 3.5m stories. 

    row["Floors Above Grade"]: numeric
        number of stories above grade

    row['"Original" Building Importance Factor Ie']
        Importance factor assigned to the building in its original design

    row["Site Class"]: str
        modern-assessed site class of the building

    seismic_hazard_params: Dictionary
        2005 NBCC introduced a site-specific response spectral acceleration table
        meant to implement a uniform hazard spectrum aimed at a 2% in 50 year (2475 rp)
        hazard.
        - Sa_0p2, Sa_0p5, Sa_1p0, Sa_2p0: 5% damped Sa at periods
        - Sa_pga: peak ground acceleration

    Returns
    -------
    vs_ns: float
        Lateral force coefficient in the n-s
    vs_ew: float
        Lateral force coefficient in the e-w
    '''
    number_of_stories = row["Floors Above Grade"]
    importance_factor = row['"Original" Building Importance Factor Ie']
    site_class = row["Site Class"]
    bldg_height = row["Building Height (Total Height Above Ground (m) to Roof Slab)"]*units.m
    lfrs_ns = row["Seismic Force Resisting System in the North-South Direction"]
    lfrs_ew = row["Seismic Force Resisting System in the East-West Direction"]
    code_year = row["effective_nbcc_year"]

    Sa_0p2 = seismic_hazard_params['Sa_0p2']
    Sa_0p5 = seismic_hazard_params['Sa_0p5']
    Sa_1p0 = seismic_hazard_params['Sa_1p0']
    Sa_2p0 = seismic_hazard_params['Sa_2p0']
    Sa_pga = seismic_hazard_params['Sa_pga']

    # lookup Fa using Sa(0.2) and site class
    # site class F needs in-depth geotechnical study
    Fa = np.interp(
        Sa_0p2, FA_TABLE_2005.index, FA_TABLE_2005[site_class]
    )

    # lookup Fv using Sa(1.0) and site class
    Fv = np.interp(
        Sa_1p0, FV_TABLE_2005.index, FV_TABLE_2005[site_class]
    )


    # TODO: temporarily estimate bldg_height if not available
    # estimate as 3.5m stories
    if np.isnan(bldg_height):
        h_n = 3.5 * units.m * number_of_stories
    else:
        h_n = bldg_height * units.m

    # SMF and SBF values are assuming ductile frames
    # a lesser, intermediate "nominal" ductility form is 
    # also available 

    # all frames not explicitly called out to be moment
    # frames are assumed to be a nominal ductility one
    # i.e. precast = nominal ductility RC frame
    built_after_1995 = code_year >= 1995

    # ductility
    # 2010 added cold-formed steel category (and BRBs)
    ductility_lookup = {
            "WLF-P9": 'conventional', # assuming CAN/CSA-O86.1-M compliant, conventional construction
            "WLF": 'conventional', # assuming CAN/CSA-O86.1-M compliant
            "WPB": 'moderate', # assuming CAN/CSA-O86.1-M compliant, moment-frame like
            "SMF": 'ductile', # ductile mrf
            "SBF": 'ductile', # ductile braced frame
            "SLF": 'ductile', # ductile mrf
            "SCW": 'moderate', # assuming wall controls, designed to moderate ductility
            "SIW": 'moderate', # assuming wall controls, designed to moderate ductility
            "CMF": 'ductile', # ductile mrf
            "CSW": 'ductile', # ductile shear wall
            "CIW": 'moderate', # assuming wall controls, designed to moderate ductility
            "PCW": 'moderate', # moderate ductility
            "PCF1": 'moderate', # assuming wall controls, designed to moderate ductility
            "PCF2": 'moderate', # nominal ductility frame
            "RML": 'moderate' if built_after_1995 else 'conventional', # "nominal ductility" RM depending on construction year
            "RMC": 'moderate' if built_after_1995 else 'conventional', # "nominal ductility" RM depending on construction year
            "URM": 'conventional',
            "CFS1": 'ductile', # non-ductile steel frame assumed, other category
            "CFS2": 'ductile',
        }

    # ductile coupled walls is assumed to be classified as "CSW", which is ductile shear wall as it is the more conservative one
    # it is assumed that RM construction 1995 and after are "nominal ductility"

    def flowchart_2010(lfrs):
        ductility_level = ductility_lookup[lfrs]
        R_d = R_D_TABLE(lfrs, ductility_level=ductility_level)
        R_o = R_O_TABLE(lfrs, ductility_level=ductility_level)

        # period estimation, using the more detailed moment-frame
        
        T_period = determine_period_post_1995(lfrs=lfrs, h_n=h_n)

        # 4.1.8.4 Sentence 6, design spectral acceleration
        T_anchor = np.array([0.2, 0.5, 1.0, 2.0, 4.0])
        S_T_functions = np.array([
            Fa * Sa_0p2,
            np.minimum(Fv * Sa_0p5, Fa * Sa_0p2),
            Fv * Sa_1p0,
            Fv * Sa_2p0,
            Fv * Sa_2p0 / 2
        ])

        S_Tperiod = np.interp(T_period, T_anchor, S_T_functions)
        S_0p2 = np.interp(0.2, T_anchor, S_T_functions)
        S_2p0 = np.interp(2.0, T_anchor, S_T_functions)

        # M_v Table 4.1.8.11
        def M_v_decision_tree(Sa_ratio, lfrs_name):
            '''
            Given Sa(0.2)/Sa(2.0) and the LFRS
            
            Return the two bound values for T_a < 1.0 and T_a > 2.0
            '''
            if Sa_ratio < 8.0:
                if lfrs_name in ['SMF', 'CMF']:
                    return np.array([1.0, 1.0, 1.0])
                elif lfrs_name in ['SBF']:
                    return np.array([1.0, 1.0, 1.0])
                # walls and wall-frame systems
                elif lfrs_name in ['SCW', 'SIW', 'CSW', 'CIW', 'PCW', 'PCF1', 'RML', 'RMC', 'URM', 'CFS1', 'CFS2']:
                    return np.array([1.0, 1.2, 1.6])
                else:
                    return np.array([1.0, 1.2, 1.2])
            else:
                if lfrs_name in ['SMF', 'CMF']:
                    return np.array([1.0, 1.2, 1.2])
                elif lfrs_name in ['SBF']:
                    return np.array([1.0, 1.5, 1.5])
                # walls and wall-frame systems
                elif lfrs_name in ['SCW', 'SIW', 'CSW', 'CIW', 'PCW', 'PCF1', 'RML', 'RMC', 'URM', 'CFS1', 'CFS2']:
                    return np.array([1.0, 2.2, 3.0])
                else:
                    return np.array([1.0, 2.2, 2.2])

        
        Mv_bounds = M_v_decision_tree(Sa_0p2/Sa_2p0, lfrs)
        M_v = np.interp(T_period, [1.0, 2.0, 4.0], Mv_bounds)
        

        # importance factor
        I_factor = importance_factor

        V_b = S_Tperiod * M_v * I_factor / (R_d * R_o)
        V_min = S_2p0 * M_v * I_factor / (R_d * R_o)

        if R_d >= 1.5:
            V_max = 2/3* S_0p2 * I_factor / (R_d * R_o)
            V_b = np.minimum(V_b, V_max)

        return np.maximum(V_b, V_min)

    
    vs_ns = flowchart_2010(lfrs_ns)
    vs_ew = flowchart_2010(lfrs_ew)

    # unsupported features:
    # cantilever-style walls
    # ornamentations
    # towers, tanks, chimneys, smokestacks, penthouses
    # floors and roofs acting as diaphragms

    return vs_ns, vs_ew

def vs_nbcc_2015(row, seismic_hazard_params):
    '''
    Calculate the lateral force coefficient based on NBC 2015, as outlined
    in Section 4.1.8.

    The coefficient is NOT yet multiplied with the building weight. 

    Distribution is later available in the same section.

    Parameters
    ----------
    row: pd.Series
        row of the inventory df

    row["Seismic Force Resisting System in the North-South Direction"]: str
        modern-classification of the n-s lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Seismic Force Resisting System in the East-West Direction"]: str
        modern-classification of the e-w lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Ground Floor Plan Area (sq.m.)"]: numeric
        Ground floor plan area in square metres, to identify dimension length
    
    row["Building Height (Total Height Above Ground (m) to Roof Slab)"]: numeric
        Building height in metres. If not provided, will be estimated with 3.5m stories. 

    row["Floors Above Grade"]: numeric
        number of stories above grade

    row['"Original" Building Importance Factor Ie']
        Importance factor assigned to the building in its original design

    row["Site Class"]: str
        modern-assessed site class of the building

    seismic_hazard_params: Dictionary
        2015 NBCC added more seismic parameters
        - Sa_0p2, Sa_0p5, Sa_1p0, Sa_2p0, Sa_5p0, Sa_10p0: 5% damped Sa at periods
        - Sa_pga: peak ground acceleration
        - Sa_pgv: peak ground velocity (misnomer but to keep consistency)


    Returns
    -------
    vs_ns: float
        Lateral force coefficient in the n-s
    vs_ew: float
        Lateral force coefficient in the e-w
    '''
    number_of_stories = row["Floors Above Grade"]
    importance_factor = row['"Original" Building Importance Factor Ie']
    site_class = row["Site Class"]
    bldg_height = row["Building Height (Total Height Above Ground (m) to Roof Slab)"]*units.m
    lfrs_ns = row["Seismic Force Resisting System in the North-South Direction"]
    lfrs_ew = row["Seismic Force Resisting System in the East-West Direction"]
    code_year = row["effective_nbcc_year"]

    Sa_0p2 = seismic_hazard_params['Sa_0p2']
    Sa_0p5 = seismic_hazard_params['Sa_0p5']
    Sa_1p0 = seismic_hazard_params['Sa_1p0']
    Sa_2p0 = seismic_hazard_params['Sa_2p0']
    Sa_5p0 = seismic_hazard_params['Sa_5p0']
    Sa_10p0 =seismic_hazard_params['Sa_10p0']
    Sa_pga = seismic_hazard_params['Sa_pga']
    Sa_pgv = seismic_hazard_params['Sa_pgv']

    # lookup Fa using PGA_ref and site class
    # site class F needs in-depth geotechnical study
    if Sa_0p2/Sa_pga < 2.0:
        PGA_ref = 0.8 * Sa_pga
    else:
        PGA_ref = Sa_pga

    F_0p2 = np.interp(
        PGA_ref, F_0P2_TABLE_2015.index, F_0P2_TABLE_2015[site_class]
    )
    F_0p5 = np.interp(
        PGA_ref, F_0P5_TABLE_2015.index, F_0P5_TABLE_2015[site_class]
    )
    F_1p0 = np.interp(
        PGA_ref, F_1P0_TABLE_2015.index, F_1P0_TABLE_2015[site_class]
    )
    F_2p0 = np.interp(
        PGA_ref, F_2P0_TABLE_2015.index, F_2P0_TABLE_2015[site_class]
    )
    F_5p0 = np.interp(
        PGA_ref, F_5P0_TABLE_2015.index, F_5P0_TABLE_2015[site_class]
    )
    F_10p0 = np.interp(
        PGA_ref, F_10P0_TABLE_2015.index, F_10P0_TABLE_2015[site_class]
    )
    F_pga = np.interp(
        PGA_ref, F_PGA_TABLE_2015.index, F_PGA_TABLE_2015[site_class]
    )
    F_pgv = np.interp(
        PGA_ref, F_PGV_TABLE_2015.index, F_PGV_TABLE_2015[site_class]
    )

    # TODO: temporarily estimate bldg_height if not available
    # estimate as 3.5m stories
    if np.isnan(bldg_height):
        h_n = 3.5 * units.m * number_of_stories
    else:
        h_n = bldg_height * units.m

    # SMF and SBF values are assuming ductile frames
    # a lesser, intermediate "nominal" ductility form is 
    # also available 

    # all frames not explicitly called out to be moment
    # frames are assumed to be a nominal ductility one
    # i.e. precast = nominal ductility RC frame

    # allowing RM to be upgraded to highest ductility if they are built to this code-year
    built_after_2015 = code_year >= 2015

    # ductility
    # 2010 added cold-formed steel category (and BRBs)
    ductility_lookup = {
        "WLF-P9": 'conventional', # assuming CAN/CSA-O86.1-M compliant, conventional construction
        "WLF": 'conventional', # assuming CAN/CSA-O86.1-M compliant
        "WPB": 'moderate', # assuming CAN/CSA-O86.1-M compliant, moment-frame like
        "SMF": 'ductile', # ductile mrf
        "SBF": 'ductile', # ductile braced frame
        "SLF": 'ductile', # ductile mrf
        "SCW": 'moderate', # assuming wall controls, designed to moderate ductility
        "SIW": 'moderate', # assuming wall controls, designed to moderate ductility
        "CMF": 'ductile', # ductile mrf
        "CSW": 'ductile', # ductile shear wall
        "CIW": 'moderate', # assuming wall controls, designed to moderate ductility
        "PCW": 'moderate', # moderate ductility
        "PCF1": 'moderate', # assuming wall controls, designed to moderate ductility
        "PCF2": 'moderate', # nominal ductility frame
        "RML": 'ductile' if built_after_2015 else 'moderate', # "nominal ductility" RM depending on construction year
        "RMC": 'ductile' if built_after_2015 else 'moderate', # "nominal ductility" RM depending on construction year
        "URM": 'conventional',
        "CFS1": 'ductile', # non-ductile steel frame assumed, other category
        "CFS2": 'ductile',
    }

    # ductile coupled walls is assumed to be classified as "CSW", which is ductile shear wall as it is the more conservative one
    # it is assumed that RM construction 2015 and after are "ductile"

    def flowchart_2015(lfrs):
        ductility_level = ductility_lookup[lfrs]
        R_d = R_D_TABLE(lfrs, ductility_level=ductility_level)
        R_o = R_O_TABLE(lfrs, ductility_level=ductility_level)

        # period estimation, using the more detailed moment-frame

        # 2015 has a specific estimation to allow for the lengthening
        # of periods for single-story buildings with steel deck or wood roof diaphragms
        # presumably for warehouse/gathering hall type buildings. The lengthening 
        # is based on the shortest bay length
        # TODO: currently omitted
        T_period = determine_period_post_1995(lfrs=lfrs, h_n=h_n)

        # 4.1.8.4 Sentence 6, design spectral acceleration
        T_anchor = np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
        S_T_functions = np.array([
            np.maximum(F_0p2*Sa_0p2, F_0p5*Sa_0p5),
            F_0p5*Sa_0p5,
            F_1p0*Sa_1p0,
            F_2p0*Sa_2p0,
            F_5p0*Sa_5p0,
            F_10p0*Sa_10p0,
        ])

        S_Tperiod = np.interp(T_period, T_anchor, S_T_functions)
        S_0p2 = np.interp(0.2, T_anchor, S_T_functions)
        S_0p5 = np.interp(0.5, T_anchor, S_T_functions)
        S_2p0 = np.interp(2.0, T_anchor, S_T_functions)
        S_4p0 = np.interp(4.0, T_anchor, S_T_functions)

        # M_v Table 4.1.8.11
        def M_v_decision_tree(Sa_ratio, lfrs_name):
            '''
            Given Sa(0.2)/Sa(5.0) and the LFRS
            
            Return the two bound values for T_a < 1.0 and T_a > 2.0
            '''
            S_ratio_Mv_anchors = np.array([5.0, 20.0, 40.0, 65.0])
            Ta_Mv_anchors = np.array([0.5, 1.0, 2.0, 5.0])

            if lfrs_name in ['SMF', 'CMF']:
                Mv_table = np.array([
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1.03, 1.03],
                ])
            elif lfrs_name in ['SBF']:
                Mv_table = np.array([
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1.04, 1.07, 1.07],
                ])
            # walls and wall-frame systems
            elif lfrs_name in ['SCW', 'SIW', 'CSW', 'CIW', 'PCW', 'PCF1', 'RML', 'RMC', 'URM', 'CFS1', 'CFS2']:
                Mv_table = np.array([
                    [1, 1, 1, 1.25],
                    [1, 1, 1.18, 2.30],
                    [1, 1.19, 1.75, 3.70],
                    [1, 1.55, 2.25, 4.65],
                ])
            else:
                Mv_table = np.array([
                    [1, 1, 1, 1],
                    [1, 1, 1.18, 1.18],
                    [1, 1.19, 1.75, 1.75],
                    [1, 1.55, 2.25, 2.25],
                ])

            interp = RegularGridInterpolator(
                (S_ratio_Mv_anchors, Ta_Mv_anchors),
                Mv_table,
                bounds_error=False,
                fill_value=None,
            )

            # clip at bounds (no extrapolation)
            Sa_ratio = np.clip(Sa_ratio, S_ratio_Mv_anchors[0], S_ratio_Mv_anchors[-1])
            # walls clip at T=4.0s, but use the 5.0 interpolation bound
            if lfrs_name in ['SCW', 'SIW', 'CSW', 'CIW', 'PCW', 'PCF1', 'RML', 'RMC', 'URM', 'CFS1', 'CFS2']:
                T_a_Mv = np.clip(T_period, Ta_Mv_anchors[0], 4.0)
            else:
                T_a_Mv = np.clip(T_period, Ta_Mv_anchors[0], Ta_Mv_anchors[-1])

            return interp((Sa_ratio, T_a_Mv))
        
        M_v = M_v_decision_tree(Sa_0p2/Sa_5p0, lfrs)
        
        # importance factor
        I_factor = importance_factor

        V_b = S_Tperiod * M_v * I_factor / (R_d * R_o)
        # wall and wall-frame minimums:
        if lfrs in ['SCW', 'SIW', 'CSW', 'CIW', 'PCW', 'PCF1', 'RML', 'RMC', 'URM', 'CFS1', 'CFS2']:
            V_min = S_4p0 * M_v * I_factor / (R_d * R_o)
        # moment frames, braced frames and other systems
        else:
            V_min = S_2p0 * M_v * I_factor / (R_d * R_o)

        # has an exception for F sites, but we have None
        if R_d >= 1.5:
            V_max = 2/3* S_0p2 * I_factor / (R_d * R_o)
            V_b = np.minimum(V_b, V_max)

            V_max = S_0p5 * I_factor / (R_d * R_o)
            V_b = np.minimum(V_b, V_max)

        return np.maximum(V_b, V_min)

    
    vs_ns = flowchart_2015(lfrs_ns)
    vs_ew = flowchart_2015(lfrs_ew)

    # unsupported features:
    # cantilever-style walls
    # ornamentations
    # towers, tanks, chimneys, smokestacks, penthouses
    # floors and roofs acting as diaphragms

    return vs_ns, vs_ew

def vs_nbcc_2020(row, seismic_hazard_params):
    '''
    Calculate the lateral force coefficient based on NBC 2020, as outlined
    in Section 4.1.8.

    The coefficient is NOT yet multiplied with the building weight. 

    Distribution is later available in the same section.

    Parameters
    ----------
    row: pd.Series
        row of the inventory df

    row["Seismic Force Resisting System in the North-South Direction"]: str
        modern-classification of the n-s lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Seismic Force Resisting System in the East-West Direction"]: str
        modern-classification of the e-w lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Ground Floor Plan Area (sq.m.)"]: numeric
        Ground floor plan area in square metres, to identify dimension length
    
    row["Building Height (Total Height Above Ground (m) to Roof Slab)"]: numeric
        Building height in metres. If not provided, will be estimated with 3.5m stories. 

    row["Floors Above Grade"]: numeric
        number of stories above grade

    row['"Original" Building Importance Factor Ie']
        Importance factor assigned to the building in its original design

    row["Site Class"]: str
        modern-assessed site class of the building

    seismic_hazard_params: Dictionary
        2015 NBCC added more seismic parameters
        - Sa_0p2, Sa_0p5, Sa_1p0, Sa_2p0, Sa_5p0, Sa_10p0: 5% damped Sa at periods
        - Sa_pga: peak ground acceleration
        - Sa_pgv: peak ground velocity (misnomer but to keep consistency)


    Returns
    -------
    vs_ns: float
        Lateral force coefficient in the n-s
    vs_ew: float
        Lateral force coefficient in the e-w
    '''
    number_of_stories = row["Floors Above Grade"]
    importance_factor = row['"Original" Building Importance Factor Ie']
    site_class = row["Site Class"]
    bldg_height = row["Building Height (Total Height Above Ground (m) to Roof Slab)"]*units.m
    lfrs_ns = row["Seismic Force Resisting System in the North-South Direction"]
    lfrs_ew = row["Seismic Force Resisting System in the East-West Direction"]
    code_year = row["effective_nbcc_year"]

    Sa_0p2 = seismic_hazard_params['Sa_0p2']
    Sa_0p5 = seismic_hazard_params['Sa_0p5']
    Sa_1p0 = seismic_hazard_params['Sa_1p0']
    Sa_2p0 = seismic_hazard_params['Sa_2p0']
    Sa_5p0 = seismic_hazard_params['Sa_5p0']
    Sa_10p0 =seismic_hazard_params['Sa_10p0']
    Sa_pga = seismic_hazard_params['Sa_pga']
    Sa_pgv = seismic_hazard_params['Sa_pgv']

    # TODO: temporarily estimate bldg_height if not available
    # estimate as 3.5m stories
    if np.isnan(bldg_height):
        h_n = 3.5 * units.m * number_of_stories
    else:
        h_n = bldg_height * units.m

    # SMF and SBF values are assuming ductile frames
    # a lesser, intermediate "nominal" ductility form is 
    # also available 

    # all frames not explicitly called out to be moment
    # frames are assumed to be a nominal ductility one
    # i.e. precast = nominal ductility RC frame

    # allowing RM to be upgraded to highest ductility if they are built to this code-year
    built_after_2015 = code_year >= 2015

    # ductility
    # 2010 added cold-formed steel category (and BRBs)
    ductility_lookup = {
        "WLF-P9": 'conventional', # assuming CAN/CSA-O86.1-M compliant, conventional construction
        "WLF": 'conventional', # assuming CAN/CSA-O86.1-M compliant
        "WPB": 'moderate', # assuming CAN/CSA-O86.1-M compliant, moment-frame like
        "SMF": 'ductile', # ductile mrf
        "SBF": 'ductile', # ductile braced frame
        "SLF": 'ductile', # ductile mrf
        "SCW": 'moderate', # assuming wall controls, designed to moderate ductility
        "SIW": 'moderate', # assuming wall controls, designed to moderate ductility
        "CMF": 'ductile', # ductile mrf
        "CSW": 'ductile', # ductile shear wall
        "CIW": 'moderate', # assuming wall controls, designed to moderate ductility
        "PCW": 'moderate', # moderate ductility
        "PCF1": 'moderate', # assuming wall controls, designed to moderate ductility
        "PCF2": 'moderate', # nominal ductility frame
        "RML": 'ductile' if built_after_2015 else 'moderate', # "nominal ductility" RM depending on construction year
        "RMC": 'ductile' if built_after_2015 else 'moderate', # "nominal ductility" RM depending on construction year
        "URM": 'conventional',
        "CFS1": 'ductile', # non-ductile steel frame assumed, other category
        "CFS2": 'ductile',
    }

    # ductile coupled walls is assumed to be classified as "CSW", which is ductile shear wall as it is the more conservative one
    # it is assumed that RM construction 2015 and after are "ductile"

    def flowchart_2020(lfrs):
        ductility_level = ductility_lookup[lfrs]
        R_d = R_D_TABLE(lfrs, ductility_level=ductility_level)
        R_o = R_O_TABLE(lfrs, ductility_level=ductility_level)

        # period estimation, using the more detailed moment-frame
        T_period = determine_period_post_1995(lfrs=lfrs, h_n=h_n)

        # 4.1.8.4 Sentence 6, design spectral acceleration
        # 2020 directly calculated site values rather than using 
        # F factors
        T_anchor = np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
        S_T_functions = np.array([
            np.maximum(Sa_0p2, Sa_0p5),
            Sa_0p5,
            Sa_1p0,
            Sa_2p0,
            Sa_5p0,
            Sa_10p0,
        ])

        S_Tperiod = np.interp(T_period, T_anchor, S_T_functions)
        S_0p2 = np.interp(0.2, T_anchor, S_T_functions)
        S_0p5 = np.interp(0.5, T_anchor, S_T_functions)
        S_2p0 = np.interp(2.0, T_anchor, S_T_functions)
        S_4p0 = np.interp(4.0, T_anchor, S_T_functions)

        # M_v Table 4.1.8.11
        def M_v_decision_tree(Sa_ratio, lfrs_name):
            '''
            Given Sa(0.2)/Sa(5.0) and the LFRS
            
            Return the two bound values for T_a < 1.0 and T_a > 2.0
            '''
            S_ratio_Mv_anchors = np.array([5.0, 20.0, 40.0, 65.0])
            Ta_Mv_anchors = np.array([0.5, 1.0, 2.0, 5.0])

            if lfrs_name in ['SMF', 'CMF']:
                Mv_table = np.array([
                    [1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                ])
            elif lfrs_name in ['SBF']:
                Mv_table = np.array([
                    [1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                    [1., 1., 1.19, 1.19],
                ])
            # walls and wall-frame systems
            elif lfrs_name in ['SCW', 'SIW', 'CSW', 'CIW', 'PCW', 'PCF1', 'RML', 'RMC', 'URM', 'CFS1', 'CFS2']:
                Mv_table = np.array([
                    [1., 1., 1., 1.30],
                    [1., 1., 1.18, 2.50],
                    [1., 1.25, 1.85, 4.10],
                    [1., 1.25, 2.30, 6.40],
                ])
            else:
                Mv_table = np.array([
                    [1., 1., 1., 1.],
                    [1., 1., 1.18, 1.18],
                    [1., 1.25, 1.85, 1.85],
                    [1., 1.37, 2.30, 2.30],
                ])

            interp = RegularGridInterpolator(
                (S_ratio_Mv_anchors, Ta_Mv_anchors),
                Mv_table,
                bounds_error=False,
                fill_value=None,
            )

            # clip at bounds (no extrapolation)
            Sa_ratio = np.clip(Sa_ratio, S_ratio_Mv_anchors[0], S_ratio_Mv_anchors[-1])
            # walls clip at T=4.0s, but use the 5.0 interpolation bound
            if lfrs_name in ['SCW', 'SIW', 'CSW', 'CIW', 'PCW', 'PCF1', 'RML', 'RMC', 'URM', 'CFS1', 'CFS2']:
                T_a_Mv = np.clip(T_period, Ta_Mv_anchors[0], 4.0)
            else:
                T_a_Mv = np.clip(T_period, Ta_Mv_anchors[0], Ta_Mv_anchors[-1])

            return interp((Sa_ratio, T_a_Mv))
        
        M_v = M_v_decision_tree(Sa_0p2/Sa_5p0, lfrs)
        
        # importance factor
        I_factor = importance_factor

        V_b = S_Tperiod * M_v * I_factor / (R_d * R_o)
        # wall and wall-frame minimums:
        if lfrs in ['SCW', 'SIW', 'CSW', 'CIW', 'PCW', 'PCF1', 'RML', 'RMC', 'URM', 'CFS1', 'CFS2']:
            V_min = S_4p0 * M_v * I_factor / (R_d * R_o)
        # moment frames, braced frames and other systems
        else:
            V_min = S_2p0 * M_v * I_factor / (R_d * R_o)

        # has an exception for F sites, but we have None
        if R_d >= 1.5:
            V_max = 2/3* S_0p2 * I_factor / (R_d * R_o)
            V_b = np.minimum(V_b, V_max)

            V_max = S_0p5 * I_factor / (R_d * R_o)
            V_b = np.minimum(V_b, V_max)

        return np.maximum(V_b, V_min)

    
    vs_ns = flowchart_2020(lfrs_ns)
    vs_ew = flowchart_2020(lfrs_ew)

    # unsupported features:
    # cantilever-style walls
    # ornamentations
    # towers, tanks, chimneys, smokestacks, penthouses
    # floors and roofs acting as diaphragms

    return vs_ns, vs_ew


def vs_nbcc_2025(row, seismic_hazard_params, historical_mode=False):
    '''
    Calculate the lateral force coefficient based on NBC 2020, as outlined
    in Section 4.1.8.

    The coefficient is NOT yet multiplied with the building weight. 

    Distribution is later available in the same section.

    Parameters
    ----------
    row: pd.Series
        row of the inventory df

    row["Seismic Force Resisting System in the North-South Direction"]: str
        modern-classification of the n-s lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Seismic Force Resisting System in the East-West Direction"]: str
        modern-classification of the e-w lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Ground Floor Plan Area (sq.m.)"]: numeric
        Ground floor plan area in square metres, to identify dimension length
    
    row["Building Height (Total Height Above Ground (m) to Roof Slab)"]: numeric
        Building height in metres. If not provided, will be estimated with 3.5m stories. 

    row["Floors Above Grade"]: numeric
        number of stories above grade

    row['"Original" Building Importance Factor Ie']
        Importance factor assigned to the building in its original design

    row["Site Class"]: str
        modern-assessed site class of the building

    seismic_hazard_params: Dictionary
        2015 NBCC added more seismic parameters
        - Sa_0p2, Sa_0p5, Sa_1p0, Sa_2p0, Sa_5p0, Sa_10p0: 5% damped Sa at periods
        - Sa_pga: peak ground acceleration
        - Sa_pgv: peak ground velocity (misnomer but to keep consistency)

    historical_mode: Boolean
        Flag used to enable retroactive calculation of base shear demand based on 
        NBCC 2025 for a building constructed to previous versions. Outlined in 
        Level 3 SEG Section 3.11
    Returns
    -------
    vs_ns: float
        Lateral force coefficient in the n-s
    vs_ew: float
        Lateral force coefficient in the e-w
    '''
    number_of_stories = row["Floors Above Grade"]
    importance_factor = row['"Original" Building Importance Factor Ie']
    site_class = row["Site Class"]
    bldg_height = row["Building Height (Total Height Above Ground (m) to Roof Slab)"]*units.m
    lfrs_ns = row["Seismic Force Resisting System in the North-South Direction"]
    lfrs_ew = row["Seismic Force Resisting System in the East-West Direction"]
    code_year = row["effective_nbcc_year"]

    Sa_0p2 = seismic_hazard_params['Sa_0p2']
    Sa_0p5 = seismic_hazard_params['Sa_0p5']
    Sa_1p0 = seismic_hazard_params['Sa_1p0']
    Sa_2p0 = seismic_hazard_params['Sa_2p0']
    Sa_5p0 = seismic_hazard_params['Sa_5p0']
    Sa_10p0 =seismic_hazard_params['Sa_10p0']
    Sa_pga = seismic_hazard_params['Sa_pga']
    Sa_pgv = seismic_hazard_params['Sa_pgv']

    # TODO: temporarily estimate bldg_height if not available
    # estimate as 3.5m stories
    if np.isnan(bldg_height):
        h_n = 3.5 * units.m * number_of_stories
    else:
        h_n = bldg_height * units.m

    # SMF and SBF values are assuming ductile frames
    # a lesser, intermediate "nominal" ductility form is 
    # also available 

    # all frames not explicitly called out to be moment
    # frames are assumed to be a nominal ductility one
    # i.e. precast = nominal ductility RC frame

    # allowing RM to be upgraded to highest ductility if they are built to this code-year
    built_after_2015 = code_year >= 2015

    # ductility
    # 2010 added cold-formed steel category (and BRBs)
    ductility_lookup = {
        "WLF-P9": 'conventional', # assuming CAN/CSA-O86.1-M compliant, conventional construction
        "WLF": 'conventional', # assuming CAN/CSA-O86.1-M compliant
        "WPB": 'moderate', # assuming CAN/CSA-O86.1-M compliant, moment-frame like
        "SMF": 'ductile', # ductile mrf
        "SBF": 'ductile', # ductile braced frame
        "SLF": 'ductile', # ductile mrf
        "SCW": 'moderate', # assuming wall controls, designed to moderate ductility
        "SIW": 'moderate', # assuming wall controls, designed to moderate ductility
        "CMF": 'ductile', # ductile mrf
        "CSW": 'ductile', # ductile shear wall
        "CIW": 'moderate', # assuming wall controls, designed to moderate ductility
        "PCW": 'moderate', # moderate ductility
        "PCF1": 'moderate', # assuming wall controls, designed to moderate ductility
        "PCF2": 'moderate', # nominal ductility frame
        "RML": 'ductile' if built_after_2015 else 'moderate', # "nominal ductility" RM depending on construction year
        "RMC": 'ductile' if built_after_2015 else 'moderate', # "nominal ductility" RM depending on construction year
        "URM": 'conventional',
        "CFS1": 'ductile', # non-ductile steel frame assumed, other category
        "CFS2": 'ductile',
    }

    # ductile coupled walls is assumed to be classified as "CSW", which is ductile shear wall as it is the more conservative one
    # it is assumed that RM construction 2015 and after are "ductile"

    def flowchart_2025(lfrs):
        ductility_level = ductility_lookup[lfrs]
        R_d = R_D_TABLE(lfrs, ductility_level=ductility_level)
        R_o = R_O_TABLE(lfrs, ductility_level=ductility_level)

        if historical_mode:
            if code_year < 1965:
                R_d = 1.0
                R_o = 1.0
            elif code_year == 1965:
                C_factor = C_TABLE_1965(lfrs)
                R_d = np.minimum(1/C_factor, 1.35)/1.35
                R_o = 1.0
            elif code_year == 1970:
                K_factor = K_TABLE_1970(lfrs)
                R_d = np.minimum(6.8/K_factor, 1.35*R_d)/1.35
                R_o = 1.0
            elif (code_year >= 1975) and (code_year <= 1985):
                K_factor = K_TABLE_1975(lfrs, number_of_stories)
                R_d_2025 = R_D_TABLE(lfrs, ductility_level='conventional')
                R_d = np.minimum(6.8/K_factor, 1.5*R_d_2025)/1.5
                R_o = 1.0
            elif code_year in [1990, 1995]:
                flexure_controlled = number_of_stories > 3
                built_after_1995 = code_year >= 1995
                R_factor = R_TABLE_1990(lfrs, flexure_controlled=flexure_controlled, built_after_1995=built_after_1995)
                R_d = np.minimum(R_factor, R_d)
                R_o = 1.0
            else:
                pass

        # period estimation, using the more detailed moment-frame
        T_period = determine_period_post_1995(lfrs=lfrs, h_n=h_n)

        # 4.1.8.4 Sentence 6, design spectral acceleration
        # 2020 directly calculated site values rather than using 
        # F factors
        T_anchor = np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
        S_T_functions = np.array([
            np.maximum(Sa_0p2, Sa_0p5),
            Sa_0p5,
            Sa_1p0,
            Sa_2p0,
            Sa_5p0,
            Sa_10p0,
        ])

        S_Tperiod = np.interp(T_period, T_anchor, S_T_functions)
        S_0p2 = np.interp(0.2, T_anchor, S_T_functions)
        S_0p5 = np.interp(0.5, T_anchor, S_T_functions)
        S_2p0 = np.interp(2.0, T_anchor, S_T_functions)
        S_4p0 = np.interp(4.0, T_anchor, S_T_functions)

        # M_v Table 4.1.8.11
        def M_v_decision_tree(Sa_ratio, lfrs_name):
            '''
            Given Sa(0.2)/Sa(5.0) and the LFRS
            
            Return the two bound values for T_a < 1.0 and T_a > 2.0
            '''
            S_ratio_Mv_anchors = np.array([5.0, 20.0, 40.0, 65.0])
            Ta_Mv_anchors = np.array([0.5, 1.0, 2.0, 5.0])

            if lfrs_name in ['SMF', 'CMF']:
                Mv_table = np.array([
                    [1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                ])
            elif lfrs_name in ['SBF']:
                Mv_table = np.array([
                    [1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                    [1., 1., 1.19, 1.19],
                ])
            # walls and wall-frame systems
            elif lfrs_name in ['SCW', 'SIW', 'CSW', 'CIW', 'PCW', 'PCF1', 'RML', 'RMC', 'URM', 'CFS1', 'CFS2']:
                Mv_table = np.array([
                    [1., 1., 1., 1.30],
                    [1., 1., 1.18, 2.50],
                    [1., 1.25, 1.85, 4.10],
                    [1., 1.25, 2.30, 6.40],
                ])
            else:
                Mv_table = np.array([
                    [1., 1., 1., 1.],
                    [1., 1., 1.18, 1.18],
                    [1., 1.25, 1.85, 1.85],
                    [1., 1.37, 2.30, 2.30],
                ])

            interp = RegularGridInterpolator(
                (S_ratio_Mv_anchors, Ta_Mv_anchors),
                Mv_table,
                bounds_error=False,
                fill_value=None,
            )

            # clip at bounds (no extrapolation)
            Sa_ratio = np.clip(Sa_ratio, S_ratio_Mv_anchors[0], S_ratio_Mv_anchors[-1])
            # walls clip at T=4.0s, but use the 5.0 interpolation bound
            if lfrs_name in ['SCW', 'SIW', 'CSW', 'CIW', 'PCW', 'PCF1', 'RML', 'RMC', 'URM', 'CFS1', 'CFS2']:
                T_a_Mv = np.clip(T_period, Ta_Mv_anchors[0], 4.0)
            else:
                T_a_Mv = np.clip(T_period, Ta_Mv_anchors[0], Ta_Mv_anchors[-1])

            return interp((Sa_ratio, T_a_Mv))
        
        M_v = M_v_decision_tree(Sa_0p2/Sa_5p0, lfrs)
        
        # importance factor
        I_factor = importance_factor

        V_b = S_Tperiod * M_v * I_factor / (R_d * R_o)
        # wall and wall-frame minimums:
        if lfrs in ['SCW', 'SIW', 'CSW', 'CIW', 'PCW', 'PCF1', 'RML', 'RMC', 'URM', 'CFS1', 'CFS2']:
            V_min = S_4p0 * M_v * I_factor / (R_d * R_o)
        # moment frames, braced frames and other systems
        else:
            V_min = S_2p0 * M_v * I_factor / (R_d * R_o)

        # has an exception for F sites, but we have None
        if R_d >= 1.5:
            V_max = 2/3* S_0p2 * I_factor / (R_d * R_o)
            V_b = np.minimum(V_b, V_max)

            V_max = S_0p5 * I_factor / (R_d * R_o)
            V_b = np.minimum(V_b, V_max)

        return np.maximum(V_b, V_min)

    
    vs_ns = flowchart_2025(lfrs_ns)
    vs_ew = flowchart_2025(lfrs_ew)

    # unsupported features:
    # cantilever-style walls
    # ornamentations
    # towers, tanks, chimneys, smokestacks, penthouses
    # floors and roofs acting as diaphragms

    return vs_ns, vs_ew

NBCC_VS_CALCULATORS = {
    1941: vs_nbcc_1941,
    1953: vs_nbcc_1953,
    1960: vs_nbcc_1953, # seismic provisions did not change
    1965: vs_nbcc_1965,
    1970: vs_nbcc_1970,
    1975: vs_nbcc_1975,
    1977: vs_nbcc_1975, # seismic provisions did not change
    1980: vs_nbcc_1975, # seismic provisions did not change except for S_factor
    1985: vs_nbcc_1985, 
    1990: vs_nbcc_1990, 
    1995: vs_nbcc_1995, # similar to 1990
    2005: vs_nbcc_2005, 
    2010: vs_nbcc_2010,
    2015: vs_nbcc_2015,
    2020: vs_nbcc_2020,
    2025: vs_nbcc_2025, # only hazard changed, added historical mode
}

def determine_period_post_1995(lfrs, h_n):
    '''
    Determine the fundamental period using the 1995 NBCC
    estimation equations (and after 1995).

    2015 onwards has a specific estimation to allow for the lengthening
    of periods for single-story buildings with steel deck or wood roof diaphragms
    presumably for warehouse/gathering hall type buildings. The lengthening 
    is based on the shortest bay length. This is not considered herein.

    Second period is based on S. Lagomarsino (1993), which is based on
    69 steel buildings, 52 RC buildings, and 40 mixed buildings.

    Masonry buildings use the same formula as RC for second period estimation.

    Timber building is estimated as 1/3 as a rough empirical estimation in
    lieu of better empirical formulae.

    Mixed buildings refer to mixed steel and concrete.
    
    Parameters
    -----------
    lfrs: lateral force resisting system, as conforming to the SEG typology list
    h_n: Height from building to roof of building in meters

    Returns
    -----------
    float:
        Fundamental period in seconds
    '''

    if lfrs == 'SMF':
        T_1 =  0.085*(h_n**0.75)
    elif lfrs == 'CMF':
        T_1 =   0.075*(h_n**0.75)
    elif lfrs == 'SBF':
        T_1 =   0.025*h_n
    else:
        T_1 =   0.05*(h_n**0.75)

    steel_buildings = ["SMF", "SBF", "SLF"]
    concrete_buildings = ["CMF", "CSW", "PCW", "CIW", "PCF1", "PCF2", "RML", "RMC", "URM"]
    mixed_buildings = ["SIW", "SCW", "CFS1", "CFS2"]
    wood_buildings = ["WLF", "WLF-P9", "WPB"]

    if lfrs in steel_buildings:
        T_2 = 0.338 * T_1
    elif lfrs in concrete_buildings:
        T_2 = 0.266 * T_1
    elif lfrs in mixed_buildings:
        T_2 = 0.274 * T_1
    elif lfrs in wood_buildings:
        T_2 = T_1 / 3

    return [T_1, T_2]

def distribute_story_shear(row, 
                           methodology_year=2025, 
                           shear_field='SEG_adjusted_NBCC_factored_Ve',
                           weight_array=None):
    '''
    Function to distribute the base shear to story forces along the 
    building height for the purposes of determining story strength forces.

    Parameters
    ------------------
    row: pd.Series
        Current analysis building
    methodology_year: int
        Year of methodology of the distribution function.
    shear_field: str
        Field of the pd.Series representing the base shear to be distributed. 
        Default is the NBCC factored base shear, representing the SEG's
        best estimate of the design code of the time.
    weight_array: np.array
        Array of the weight distribution.
        Default is None, which will generate a generic distribution

    row["Seismic Force Resisting System in the North-South Direction"]: str
        modern-classification of the n-s lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies

    row["Seismic Force Resisting System in the East-West Direction"]: str
        modern-classification of the e-w lateral force resisting system in the 
        NRC Seismic Evaluation Guidelines typologies
    
    row["Building Height (Total Height Above Ground (m) to Roof Slab)"]: numeric
        Building height in metres. If not provided, will be estimated with 3.5m stories. 

    row["Floors Above Grade"]: numeric
        number of stories above grade

    row[shear_field]: tuple
        Pair of n-s e-w base shear to be distributed

    Returns
    tuple:
        (np.array, np.array) corresponding to the NS and EW story strength distribution.
        This is the cumulative sum of the story forces F_x above that level.

    '''
    vs_ns, vs_ew = row[shear_field]
    number_of_stories = int(row["Floors Above Grade"])
    bldg_height = row["Building Height (Total Height Above Ground (m) to Roof Slab)"]*units.m
    lfrs_ns = row["Seismic Force Resisting System in the North-South Direction"]
    lfrs_ew = row["Seismic Force Resisting System in the East-West Direction"]

    # placeholder, a seismic weight array with 1.0 for floors and 0.75 for roof
    if weight_array is None:
        W_x = np.ones(number_of_stories)
        W_x[-1] = 0.75

    # estimate building height array
    if np.isnan(bldg_height):
        h_n = 3.5 * units.m * number_of_stories
        h_x = 3.5 * units.m * np.arange(1.0, number_of_stories+1)
    else:
        h_n = bldg_height * units.m
        h_ix = h_n / number_of_stories
        h_x = h_ix * np.arange(1.0, number_of_stories+1)


    def calculate_Vj(lfrs, V_E):

        # top level force. currently only 2025 is supported
        T_n = determine_period_post_1995(lfrs=lfrs, h_n=h_n)
        T_a = T_n[0]
        if methodology_year == 2025:
            F_t = 0.07 * T_a * V_E
            F_t = np.minimum(F_t, 0.25*V_E)
            if T_a <= 0.7:
                F_t = 0.0

        F_x = (V_E - F_t) * W_x * h_x / (np.dot(W_x, h_x))
        F_x[-1] += F_t

        return np.cumsum(F_x[::-1])[::-1]

    V_j_ns = calculate_Vj(lfrs_ns, vs_ns)
    V_j_ew = calculate_Vj(lfrs_ew, vs_ew)

    return V_j_ns, V_j_ew

# TODO: condense repeated functions
# R factor


def C_TABLE_1965(lfrs):
    # TODO: request review on this
    ductile_mrf_rcsw = ["SMF", "SCW", "SIW", "CMF", "CIW", "CFS1"]
    if lfrs in ductile_mrf_rcsw:
        return 0.75
    else:
        return 1.25 

def K_TABLE_1970(lfrs):

    # TODO: request review on this
    # ductile systems weren't provided until 1973
    ductile_moment_frames = ["SMF", "CMF"]

    # consisting of a complete ductile moment resisting space frame and shear walls
    # 1) resist total lateral force in accordance with their rigidity
    # 2) shear walls resist total lateral force independent of ductile MF
    # 3) MF resist at least 25% of required lateral force
    dual_systems = ["SCW", "PCF1", "CFS1"]

    if lfrs in ductile_moment_frames:
        K_factor = 0.67
    elif lfrs in dual_systems:
        K_factor = 0.80
    else:
        K_factor = 1.00
    return K_factor

def K_TABLE_1975(lfrs, number_of_stories):
    # TODO: request review on this
    ductile_moment_frames = ["SMF", "CMF"]

    # consisting of a complete ductile moment resisting space frame and shear walls
    # 1) resist total lateral force in accordance with their rigidity
    # 2) shear walls resist total lateral force independent of ductile MF
    # 3) MF resist at least 25% of required lateral force
    # assume that if 3 stories or more, flexure wall
    dual_systems = ["SCW", "PCF1", "CFS1"]

    # ductile flexural wall and buildings with 
    # ductile framing systems not otherwise classified in this 
    # Table as Cases 1,2,3 or 5. 
    wall_systems = ["CSW", "PCW"] 
    # it is assumed a designer would consider PC walls to be ductile at the time

    # Buildings with a dual structural system consisting of a 
    # complete ductile moment-resisting space frame with 
    # masonry infilling 
    infill_systems = ["SIW", "CIW"]

    # Buildings (other than Cases I , 2, 3, 4 and 5) of (a) continu- 
    # ously reinforced concrete. (b) structural steel. and (c) rein- 
    # forced masonry shear walls. 
    other_and_rm_systems = ["SBF", "SLF", "RML", "RMC", "PCF2"]

    # unreinforced masonry
    urm_systems = ["URM"]

     # system-specific force reductions
    if lfrs in ductile_moment_frames:
        K_factor = 0.70
    # shear-controlled walls dual systems
    elif (lfrs in dual_systems) and (number_of_stories < 3):
        K_factor = 0.80
    # flexure-controlled walls dual systems
    elif (lfrs in dual_systems) and (number_of_stories >= 3):
        K_factor = 0.70
    # ductile walls and frames
    elif lfrs in wall_systems:
        K_factor = 1.0
    # infill systems
    elif lfrs in infill_systems:
        K_factor = 1.3
    # other continuous rc steel rm systems
    elif lfrs in other_and_rm_systems:
        K_factor = 1.3
    # urms
    elif lfrs in urm_systems:
        K_factor = 2.0
    # all others
    else:
        K_factor = 2.0

    return K_factor

def R_TABLE_1990(lfrs, flexure_controlled, built_after_1995=False):
    '''
    SMF and SBF values are assuming ductile frames
    a lesser, intermediate "nominal" ductility form is 
    also available 


    all frames not explicitly called out to be moment
    frames are assumed to be a nominal ductility one
    i.e. precast = nominal ductility RC frame or wall
    
    1995 added steel plate shear walls, ductile coupled walls, and RM with nominal ductility
    however, there is no NRC typology for steel shear wall
    ductile coupled walls is assumed to be classified as "CSW"
    it is assumed that RM construction 1995 and after are "nominal ductility"

    WLF assume "conventional construction" typical for homes, with incomplete 
    lateral resisting system
    '''

    R_lookup_table = {
        "WLF-P9": 1.5, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections
        "WLF": 1.5, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections
        "WPB": 1.5, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections
        "SMF": 4.0, # ductile mrf
        "SBF": 3.0, # ductile braced frame
        "SLF": 1.5, # non-ductile steel frame assumed, other category
        "SCW": 2.0, # assuming wall controls, nominal ductility
        "SIW": 2.0, # assuming wall controls, nominal ductility
        "CMF": 4.0, # ductile mrf
        "CSW": 3.5 if flexure_controlled else 2.0, # either ductile flexural wall or nominal ductility (shear-controlled)
        "CIW": 2.0, # assuming wall controls, nominal ductility
        "PCW": 2.0, # nominal ductility
        "PCF1": 2.0, # nominal ductility frame or wall (both same R factor)
        "PCF2": 2.0, # nominal ductility frame
        "RML": 2.0 if built_after_1995 else 1.5, # "nominal ductility" RM depending on construction year
        "RMC": 2.0 if built_after_1995 else 1.5, # "nominal ductility" RM depending on construction year 
        "URM": 1.0,
        "CFS1": 1.5, # non-ductile steel frame assumed, other category
        "CFS2": 1.5
        }
    return R_lookup_table[lfrs]


def R_D_TABLE(lfrs, ductility_level='ductile'):
    # ductility
    # 2010 added cold-formed steel category (and BRBs)

    if ductility_level == 'ductile':
        Rd_lookup_table = {
            "WLF-P9": 2.0, # assuming CAN/CSA-O86.1-M moderately ductile MRF
            "WLF": 2.0, # assuming CAN/CSA-O86.1-M moderately ductile MRF
            "WPB": 2.0, # assuming CAN/CSA-O86.1-M MRF
            "SMF": 5.0, # ductile mrf
            "SBF": 3.0, # ductile braced frame
            "SLF": 5.0, # ductile mrf
            "SCW": 3.5, # ductile shear wall controls
            "SIW": 3.0, # ductile rm wall controls
            "CMF": 4.0, # ductile mrf
            "CSW": 3.5, # ductile shear wall
            "CIW": 3.0, # ductile rm wall controls
            "PCW": 3.5, # ductile shear wall
            "PCF1": 3.5, # ductile shear wall controls
            "PCF2": 4.0, # ductile mrf
            "RML": 3.0, # ductile rm
            "RMC": 3.0,
            "URM": 1.0,
            "CFS1": 2.5, # wood-only shear walls with cold-formed steel (no gypsum)
            "CFS2": 1.9, # "limited ductility" diagonal strap concentrically braced wall (better than conventional)
        }
    elif ductility_level == 'moderate':
        Rd_lookup_table = {
            "WLF-P9": 2.0, # assuming CAN/CSA-O86.1-M moderately ductile MRF
            "WLF": 2.0, # assuming CAN/CSA-O86.1-M moderately ductile MRF
            "WPB": 2.0, # assuming CAN/CSA-O86.1-M MRF
            "SMF": 3.5, # moderately ductile mrf
            "SBF": 3.0, # moderately ductile braced frame
            "SLF": 3.5, # moderately ductile mrf
            "SCW": 2.0, # moderately ductile shear wall controls
            "SIW": 2.0, # moderately ductile rm wall controls
            "CMF": 2.5, # moderately ductile mrf
            "CSW": 2.0, # moderately ductile shear wall
            "CIW": 2.0, # moderately ductile rm wall controls
            "PCW": 2.0, # moderately ductile tilt-up wall
            "PCF1": 2.0, # moderately ductile tilt-up wall controls
            "PCF2": 2.5, # moderately ductile mrf
            "RML": 2.0, # moderately ductile rm
            "RMC": 2.0, # moderately
            "URM": 1.0,
            "CFS1": 2.5, # wood-only shear walls with cold-formed steel (no gypsum)
            "CFS2": 1.9, # "limited ductility" diagonal strap concentrically braced wall (better than conventional)
        }
        
    elif ductility_level == 'limited':
        Rd_lookup_table = {
            "WLF-P9": 1.0, # assuming CAN/CSA-O86.1-M limited ductile MRF
            "WLF": 1.0, # assuming CAN/CSA-O86.1-M limited ductile MRF
            "WPB": 1.0, # assuming CAN/CSA-O86.1-M MRF
            "SMF": 2.0, # limited ductile mrf
            "SBF": 2.0, # limited ductile braced frame
            "SLF": 2.0, # limited ductile mrf
            "SCW": 1.5, # conventional ductile shear wall controls
            "SIW": 1.5, # conventional ductile rm wall controls
            "CMF": 1.5, # conventional ductile mrf
            "CSW": 1.5, # conventional ductile shear wall
            "CIW": 1.5, # conventional ductile rm wall controls
            "PCW": 1.5, # conventional ductile tilt-up wall
            "PCF1": 1.5, # conventional ductile tilt-up wall controls
            "PCF2": 1.5, # conventional ductile mrf
            "RML": 1.5, # conventional ductile rm
            "RMC": 1.5, # conventional
            "URM": 1.0,
            "CFS1": 1.5, # wood-gypsum only shear walls with cold-formed steel
            "CFS2": 1.9, # "limited ductility" diagonal strap concentrically braced wall (better than conventional)
        }
        
    elif ductility_level == 'conventional':
        Rd_lookup_table = {
            "WLF-P9": 1.0, # assuming CAN/CSA-O86.1-M other MRF
            "WLF": 1.0, # assuming CAN/CSA-O86.1-M other MRF
            "WPB": 1.0, # assuming CAN/CSA-O86.1-M otherMRF
            "SMF": 1.5, # conventional ductile mrf
            "SBF": 1.5, # conventional ductile braced frame
            "SLF": 1.5, # conventional ductile mrf
            "SCW": 1.5, # conventional ductile shear wall controls
            "SIW": 1.5, # conventional ductile rm wall controls
            "CMF": 1.5, # conventional ductile mrf
            "CSW": 1.5, # conventional ductile shear wall
            "CIW": 1.5, # conventional ductile rm wall controls
            "PCW": 1.5, # conventional ductile tilt-up wall
            "PCF1": 1.5, # conventional ductile tilt-up wall controls
            "PCF2": 1.5, # conventional ductile mrf
            "RML": 1.5, # conventional ductile rm
            "RMC": 1.5, # conventional
            "URM": 1.0,
            "CFS1": 1.5, # wood-gypsum only shear walls with cold-formed steel
            "CFS2": 1.2, # conventional diagonal strap cbf
        }

    return Rd_lookup_table[lfrs]

def R_O_TABLE(lfrs, ductility_level='ductile'):
    # overstrength
    if ductility_level == 'ductile':
        Ro_lookup_table = {
            "WLF-P9": 1.5, # assuming CAN/CSA-O86.1-M moderately ductile MRF
            "WLF": 1.5, # assuming CAN/CSA-O86.1-M moderately ductile MRF
            "WPB": 1.5, # assuming CAN/CSA-O86.1-M MRF
            "SMF": 1.5, # ductile mrf
            "SBF": 1.3, # ductile braced frame
            "SLF": 1.5, # ductile mrf
            "SCW": 1.6, # ductile shear wall controls
            "SIW": 1.5, # ductile rm wall controls
            "CMF": 1.7, # ductile mrf
            "CSW": 1.6, # ductile shear wall
            "CIW": 1.5, # ductile rm wall controls
            "PCW": 1.6, # ductile shear wall
            "PCF1": 1.6, # ductile shear wall controls
            "PCF2": 1.7, # ductile mrf
            "RML": 1.5, # ductile rm
            "RMC": 1.5,
            "URM": 1.0,
            "CFS1": 1.7, # wood-only shear walls with cold-formed steel (no gypsum)
            "CFS2": 1.3, # "limited ductility" diagonal strap concentrically braced wall (better than conventional)
        }
    
    if ductility_level == 'moderate':
        Ro_lookup_table = {
            "WLF-P9": 1.5, # assuming CAN/CSA-O86.1-M moderately ductile MRF
            "WLF": 1.5, # assuming CAN/CSA-O86.1-M moderately ductile MRF
            "WPB": 1.5, # assuming CAN/CSA-O86.1-M MRF
            "SMF": 1.5, # moderately ductile mrf
            "SBF": 1.3, # moderately ductile braced frame
            "SLF": 1.5, # moderately ductile mrf
            "SCW": 1.6, # moderately ductile shear wall controls
            "SIW": 1.5, # moderately ductile rm wall controls
            "CMF": 1.4, # moderately ductile mrf
            "CSW": 1.6, # moderately ductile shear wall
            "CIW": 1.5, # moderately ductile rm wall controls
            "PCW": 1.3, # moderately ductile tilt-up wall
            "PCF1": 1.3, # moderately ductile tilt-up wall controls
            "PCF2": 1.4, # moderately ductile mrf
            "RML": 1.5, # moderately ductile rm
            "RMC": 1.5,
            "URM": 1.0,
            "CFS1": 1.7, # wood-only shear walls with cold-formed steel (no gypsum)
            "CFS2": 1.3, # "limited ductility" diagonal strap concentrically braced wall (better than conventional)
        }
    
    elif ductility_level == 'limited':
        Ro_lookup_table = {
            "WLF-P9": 1.0, # assuming CAN/CSA-O86.1-M other MRF
            "WLF": 1.0, # assuming CAN/CSA-O86.1-M other MRF
            "WPB": 1.0, # assuming CAN/CSA-O86.1-M other MRF
            "SMF": 1.3, # limited ductile mrf
            "SBF": 1.3, # limited ductile braced frame
            "SLF": 1.3, # limited ductile mrf
            "SCW": 1.3, # conventional ductile shear wall controls
            "SIW": 1.5, # conventional ductile rm wall controls
            "CMF": 1.3, # conventional ductile mrf
            "CSW": 1.3, # conventional ductile shear wall
            "CIW": 1.5, # conventional ductile rm wall controls
            "PCW": 1.3, # limited ductile tilt-up wall
            "PCF1": 1.3, # limited ductile tilt-up wall controls
            "PCF2": 1.3, # conventional ductile mrf
            "RML": 1.5, # conventional ductile rm
            "RMC": 1.5, # conventional
            "URM": 1.0,
            "CFS1": 1.7, # wood-gypsum shear walls with cold-formed steel
            "CFS2": 1.3, # "limited ductility" diagonal strap concentrically braced wall (better than conventional)
        }
    
    elif ductility_level == 'conventional':
        Ro_lookup_table = {
            "WLF-P9": 1.0, # assuming CAN/CSA-O86.1-M other MRF
            "WLF": 1.0, # assuming CAN/CSA-O86.1-M other MRF
            "WPB": 1.0, # assuming CAN/CSA-O86.1-M other MRF
            "SMF": 1.3, # conventional ductile mrf
            "SBF": 1.3, # conventional ductile braced frame
            "SLF": 1.3, # conventional ductile mrf
            "SCW": 1.3, # conventional ductile shear wall controls
            "SIW": 1.5, # conventional ductile rm wall controls
            "CMF": 1.3, # conventional ductile mrf
            "CSW": 1.3, # conventional ductile shear wall
            "CIW": 1.5, # conventional ductile rm wall controls
            "PCW": 1.3, # conventional ductile tilt-up wall
            "PCF1": 1.3, # conventional ductile tilt-up wall controls
            "PCF2": 1.3, # conventional ductile mrf
            "RML": 1.5, # conventional ductile rm
            "RMC": 1.5, # conventional
            "URM": 1.0,
            "CFS1": 1.7, # wood-gypsum shear walls with cold-formed steel
            "CFS2": 1.3, # conventional
        }
    return Ro_lookup_table[lfrs]

# index is Sa_0p2
FA_TABLE_2005 = pd.DataFrame({
    'A': [0.7, 0.7, 0.8, 0.8, 0.8],
    'B': [0.8, 0.8, 0.9, 1.0, 1.0],
    'C': [1.0, 1.0, 1.0, 1.0, 1.0],
    'D': [1.3, 1.2, 1.1, 1.1, 1.0],
    'E': [2.1, 1.4, 1.1, 0.9, 0.9]
},
index=[0.25, 0.50, 0.75, 1.0, 1.25], dtype=float)


# index is Sa_1p0
FV_TABLE_2005 = pd.DataFrame({
    'A': [0.5, 0.5, 0.5, 0.6, 0.6],
    'B': [0.6, 0.7, 0.7, 0.8, 0.8],
    'C': [1.0, 1.0, 1.0, 1.0, 1.0],
    'D': [1.4, 1.3, 1.2, 1.1, 1.1],
    'E': [2.1, 2.0, 1.9, 1.7, 1.7]
},
index=[0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)

# index is PGA_ref
F_0P2_TABLE_2015 = pd.DataFrame({
    'A': [0.69, 0.69, 0.69, 0.69, 0.69],
    'B': [0.77, 0.77, 0.77, 0.77, 0.77],
    'C': [1.0, 1.0, 1.0, 1.0, 1.0],
    'D': [1.24, 1.09, 1.00, 0.94, 0.90],
    'E': [1.64, 1.24, 1.05, 0.93, 0.85]
},
index=[0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)

# index is PGA_ref
F_0P5_TABLE_2015 = pd.DataFrame({
    'A': [0.57, 0.57, 0.57, 0.57, 0.57],
    'B': [0.65, 0.65, 0.65, 0.65, 0.65],
    'C': [1.0, 1.0, 1.0, 1.0, 1.0],
    'D': [1.47, 1.30, 1.20, 1.14, 1.10],
    'E': [2.47, 1.80, 1.48, 1.30, 1.17]
},
index=[0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)

# index is PGA_ref
F_1P0_TABLE_2015 = pd.DataFrame({
    'A': [0.57, 0.57, 0.57, 0.57, 0.57],
    'B': [0.63, 0.63, 0.63, 0.63, 0.63],
    'C': [1.0, 1.0, 1.0, 1.0, 1.0],
    'D': [1.55, 1.39, 1.31, 1.25, 1.21],
    'E': [2.81, 2.08, 1.74, 1.53, 1.39]
},
index=[0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)

# index is PGA_ref
F_2P0_TABLE_2015 = pd.DataFrame({
    'A': [0.58, 0.58, 0.58, 0.58, 0.58],
    'B': [0.63, 0.63, 0.63, 0.63, 0.63],
    'C': [1.0, 1.0, 1.0, 1.0, 1.0],
    'D': [1.57, 1.44, 1.36, 1.31, 1.27],
    'E': [2.90, 2.24, 1.92, 1.72, 1.58]
},
index=[0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)

# index is PGA_ref
F_5P0_TABLE_2015 = pd.DataFrame({
    'A': [0.61, 0.61, 0.61, 0.61, 0.61],
    'B': [0.64, 0.64, 0.64, 0.64, 0.64],
    'C': [1.0, 1.0, 1.0, 1.0, 1.0],
    'D': [1.58, 1.48, 1.41, 1.37, 1.34],
    'E': [2.93, 2.40, 2.14, 1.96, 1.84]
},
index=[0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)

# index is PGA_ref
F_10P0_TABLE_2015 = pd.DataFrame({
    'A': [0.67, 0.67, 0.67, 0.67, 0.67],
    'B': [0.69, 0.69, 0.69, 0.69, 0.69],
    'C': [1.0, 1.0, 1.0, 1.0, 1.0],
    'D': [1.49, 1.41, 1.37, 1.34, 1.31],
    'E': [2.52, 2.18, 2.00, 1.88, 1.79]
},
index=[0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)

# index is PGA_ref
F_PGA_TABLE_2015 = pd.DataFrame({
    'A': [0.90, 0.90, 0.90, 0.90, 0.90],
    'B': [0.87, 0.87, 0.87, 0.87, 0.87],
    'C': [1.0, 1.0, 1.0, 1.0, 1.0],
    'D': [1.29, 1.10, 0.99, 0.93, 0.88],
    'E': [1.81, 1.23, 0.98, 0.83, 0.74]
},
index=[0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)

# index is PGA_ref
F_PGV_TABLE_2015 = pd.DataFrame({
    'A': [0.62, 0.62, 0.62, 0.62, 0.62],
    'B': [0.67, 0.67, 0.67, 0.67, 0.67],
    'C': [1.0, 1.0, 1.0, 1.0, 1.0],
    'D': [1.47, 1.30, 1.20, 1.14, 1.10],
    'E': [2.47, 1.80, 1.48, 1.30, 1.17]
},
index=[0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)

SEG_TO_FEMA = {
    "WLF-P9": "W1",
    "WLF": "W2",
    "WPB": "W2",
    "SMF": "S1",
    "SBF": "S2",
    "SLF": "S3",
    "SCW": "S4",
    "SIW": "S5",
    "CMF": "C1",
    "CSW": "C2",
    "CIW": "C3",
    "PCW": "PC1",
    "PCF1": "PC2",
    "PCF2": "PC2",
    "RML": "RM1",
    "RMC": "RM2",
    "URM": "URM",
    "CFS1": "S3",
    "CFS2": "S3",
}


BENCHMARK_YEAR = {
    "WLF-P9": 2010,
    "WLF": 2005,
    "WPB": 2005,
    "SMF": 2005,
    "SBF": 2005,
    "SLF": 2005,
    "SCW": 2005,
    "SIW": 2005,
    "CMF": 2005,
    "CSW": 2005,
    "CIW": 2005,
    "PCW": 2015,
    "PCF1":2005,
    "PCF2":2005,
    "RML": 2005,
    "RMC": 2005,
    "URM": 2005,
    "CFS1":2010,
    "CFS2":2010,
}