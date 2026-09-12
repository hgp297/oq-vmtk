import numpy as np
import pandas as pd
from openquake.vmtk.units import units
from scipy.interpolate import RegularGridInterpolator

NBCC_YEARS = np.array([1941, 1953, 1960, 1965, 1970, 1975, 1977, 1980, 1985, 1990, 1995, 2005, 2010, 2015, 2020])

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
    
    row['original_nbcc_unfactored_V']: tuple
        Unfactored base shear calculated from properties.determine_code_strength
        (vs_ns, vs_ew)

    Returns
    -------
    tuple
        Factored base shear in each direction (ns, ew)
    '''

    code_year = row["effective_nbcc_year"]
    vs_ns, vs_ew = row["original_nbcc_unfactored_V"]
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
            2020: 1.0
        }[code_year]

    load_factor_ns = lookup_load_factor(lfrs_ns)
    load_factor_ew = lookup_load_factor(lfrs_ew)


    return load_factor_ns * vs_ns, load_factor_ew * vs_ew

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

    def flowchart_1941(lfrs, site_class):
        if lfrs in bearing_wall_systems:
            return 0.05
        elif site_class not in stronger_soil_bearing_sites:
            return 0.04
        else:
            return 0.02
        
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
    # TODO: request review on this
    ductile_mrf_rcsw = ["SMF", "SCW", "SIW", "CMF", "CIW", "CFS1"]

    # 1965 Foundation factor depends on "compressible" soil
    compressible_soil_sites = ['E', 'F']

    def flowchart_1965(lfrs, site_class):
        if lfrs in ductile_mrf_rcsw:
            C_factor = 0.75
        else:
            C_factor = 1.25

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
        if lfrs in ductile_moment_frames:
            K_factor = 0.67
            T_period = 0.1 * number_of_stories
        elif lfrs in dual_systems:
            K_factor = 0.80
            T_period = 0.05*h_n_ft/(D_ft**0.5)
        else:
            K_factor = 1.00
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
        return 4*vs_ns, 4*vs_ew
    elif seismic_zone == 2:
        return 2*vs_ns, 2*vs_ew
    elif seismic_zone == 1:
        return vs_ns, vs_ew
    else:
        return 0.0, 0.0

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
    in Section 4.1.7.

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
    # else, roughly estimate D_s as about 50% of the length
    # length of just the LFRS
    whole_length_systems = ["SMF", "CMF", "SCW", "PCF1", "CFS1", "CFS2",
                            "RML", "RMC", "SLF", "PCF2", "URM", "SIW", "CIW"]

    if lfrs_ns in whole_length_systems:
        D_s_ns = (plan_area**0.5)
    else:
        D_s_ns = (plan_area**0.5)*0.50

    if lfrs_ew in whole_length_systems:
        D_s_ew = (plan_area**0.5)
    else:
        D_s_ew = (plan_area**0.5)*0.50

    # TODO: temporarily estimate bldg_height if not available
    # estimate as 3.5 m stories
    if np.isnan(bldg_height):
        h_n = 3.5*units.m * number_of_stories
    else:
        h_n = bldg_height * units.m

    # "A ductile moment-resisting space frame is a space frame that is designed to resist
    # all the specified seismic forces and that, in addition, has adequate ductility or
    # energy-absorptive capacity."
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
    # else, roughly estimate D_s as about 50% of the length
    # length of just the LFRS
    whole_length_systems = ["SMF", "CMF", "SCW", "PCF1", "CFS1", "CFS2",
                            "RML", "RMC", "SLF", "PCF2", "URM", "SIW", "CIW"]

    if lfrs_ns in whole_length_systems:
        D_s_ns = (plan_area**0.5)
    else:
        D_s_ns = (plan_area**0.5)*0.50

    if lfrs_ew in whole_length_systems:
        D_s_ew = (plan_area**0.5)
    else:
        D_s_ew = (plan_area**0.5)*0.50

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

    # 1990 has an additional category for very soft fine-grained soils >15m
    # assumed that it wouldn't be here unless a flag is raised

    # SMF and SBF values are assuming ductile frames
    # a lesser, intermediate "nominal" ductility form is 
    # also available 

    # all frames not explicitly called out to be moment
    # frames are assumed to be a nominal ductility one
    # i.e. precast = nominal ductility RC frame or wall
    flexure_controlled = number_of_stories > 3
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
        "RML": 1.5, 
        "RMC": 1.5,
        "URM": 1.0,
        "CFS1": 1.5, # non-ductile steel frame assumed, other category
        "CFS2": 1.5
    }

    # D_s could change per-direction
    def flowchart_1990(lfrs, D_s):

        R_factor = R_lookup_table[lfrs]

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
    # else, roughly estimate D_s as about 50% of the length
    # length of just the LFRS
    whole_length_systems = ["SMF", "CMF", "SCW", "PCF1", "CFS1", "CFS2",
                            "RML", "RMC", "SLF", "PCF2", "URM", "SIW", "CIW"]

    if lfrs_ns in whole_length_systems:
        D_s_ns = (plan_area**0.5)
    else:
        D_s_ns = (plan_area**0.5)*0.50

    if lfrs_ew in whole_length_systems:
        D_s_ew = (plan_area**0.5)
    else:
        D_s_ew = (plan_area**0.5)*0.50

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

    # 1990 has an additional category for very soft fine-grained soils >15m
    # assumed that it wouldn't be here unless a flag is raised

    # SMF and SBF values are assuming ductile frames
    # a lesser, intermediate "nominal" ductility form is 
    # also available 

    # all frames not explicitly called out to be moment
    # frames are assumed to be a nominal ductility one
    # i.e. precast = nominal ductility RC frame
    flexure_controlled = number_of_stories > 3
    built_after_1995 = code_year >= 1995
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
    # 1995 added steel plate shear walls, ductile coupled walls, and RM with nominal ductility
    # however, there is no NRC typology for steel shear wall
    # ductile coupled walls is assumed to be classified as "CSW"
    # it is assumed that RM construction 1995 and after are "nominal ductility"

    # D_s could change per-direction
    def flowchart_1995(lfrs, D_s):

        R_factor = R_lookup_table[lfrs]

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
    Rd_lookup_table = {
        "WLF-P9": 1.0, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections or shear walls
        "WLF": 1.0, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections or shear walls
        "WPB": 1.0, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections or shear walls
        "SMF": 5.0, # ductile mrf
        "SBF": 3.0, # ductile braced frame
        "SLF": 1.0, # non-ductile steel frame assumed, other category
        "SCW": 2.0, # assuming wall controls, designed to moderate ductility
        "SIW": 2.0, # assuming wall controls, designed to moderate ductility
        "CMF": 4.0, # ductile mrf
        "CSW": 3.5, # ductile shear wall
        "CIW": 2.0, # assuming wall controls, designed to moderate ductility
        "PCW": 2.5, # nominal ductility
        "PCF1": 2.0, # assuming wall controls, designed to moderate ductility
        "PCF2": 2.5, # nominal ductility frame
        "RML": 2.0 if built_after_1995 else 1.5, # "nominal ductility" RM depending on construction year
        "RMC": 2.0 if built_after_1995 else 1.5, # "nominal ductility" RM depending on construction year 
        "URM": 1.0,
        "CFS1": 1.0, # non-ductile steel frame assumed, other category
        "CFS2": 1.0
    }
    
    # overstrength
    Ro_lookup_table = {
        "WLF-P9": 1.0, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections
        "WLF": 1.0, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections
        "WPB": 1.0, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections
        "SMF": 1.5, # ductile mrf
        "SBF": 1.3, # ductile braced frame
        "SLF": 1.0, # non-ductile steel frame assumed, other category
        "SCW": 1.4, # assuming wall controls, moderate ductility
        "SIW": 1.4, # assuming wall controls, moderate ductility
        "CMF": 1.7, # ductile mrf
        "CSW": 1.6, # ductile shear wall
        "CIW": 1.4, # assuming wall controls, moderate ductility
        "PCW": 1.4, # nominal ductility
        "PCF1": 1.4, # assuming wall controls, moderate ductility
        "PCF2": 1.4, # nominal ductility frame
        "RML": 1.5, # 
        "RMC": 1.5, # 
        "URM": 1.0,
        "CFS1": 1.0, # non-ductile steel frame assumed, other category
        "CFS2": 1.0
    }
    # ductile coupled walls is assumed to be classified as "CSW", which is ductile shear wall as it is the more conservative one
    # it is assumed that RM construction 1995 and after are "nominal ductility"

    def flowchart_2005(lfrs):

        R_d = Rd_lookup_table[lfrs]
        R_o = Ro_lookup_table[lfrs]

        # period estimation, using the more detailed moment-frame
        if lfrs == 'SMF':
            T_period = 0.085*(h_n**0.75)
        elif lfrs == 'CMF':
            T_period = 0.075*(h_n**0.75)
        elif lfrs == 'SBF':
            T_period = 0.025*h_n
        else:
            T_period = 0.05*(h_n**0.75)

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
    Rd_lookup_table = {
        "WLF-P9": 1.0, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections or shear walls
        "WLF": 1.0, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections or shear walls
        "WPB": 1.0, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections or shear walls
        "SMF": 5.0, # ductile mrf
        "SBF": 3.0, # ductile braced frame
        "SLF": 1.0, # non-ductile steel frame assumed, other category
        "SCW": 2.0, # assuming wall controls, designed to moderate ductility
        "SIW": 2.0, # assuming wall controls, designed to moderate ductility
        "CMF": 4.0, # ductile mrf
        "CSW": 3.5, # ductile shear wall
        "CIW": 2.0, # assuming wall controls, designed to moderate ductility
        "PCW": 2.5, # nominal ductility
        "PCF1": 2.0, # assuming wall controls, designed to moderate ductility
        "PCF2": 2.5, # nominal ductility frame
        "RML": 2.0 if built_after_1995 else 1.5, # "nominal ductility" RM depending on construction year
        "RMC": 2.0 if built_after_1995 else 1.5, # "nominal ductility" RM depending on construction year 
        "URM": 1.0,
        "CFS1": 2.5, # wood-only shear walls with cold-formed steel (no gypsum)
        "CFS2": 1.9, # "limited ductility" diagonal strap concentrically braced wall (better than conventional)
    }
    
    # overstrength
    Ro_lookup_table = {
        "WLF-P9": 1.0, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections
        "WLF": 1.0, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections
        "WPB": 1.0, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections
        "SMF": 1.5, # ductile mrf
        "SBF": 1.3, # ductile braced frame
        "SLF": 1.0, # non-ductile steel frame assumed, other category
        "SCW": 1.4, # assuming wall controls, moderate ductility
        "SIW": 1.4, # assuming wall controls, moderate ductility
        "CMF": 1.7, # ductile mrf
        "CSW": 1.6, # ductile shear wall
        "CIW": 1.4, # assuming wall controls, moderate ductility
        "PCW": 1.4, # nominal ductility
        "PCF1": 1.4, # assuming wall controls, moderate ductility
        "PCF2": 1.4, # nominal ductility frame
        "RML": 1.5, # 
        "RMC": 1.5, # 
        "URM": 1.0,
        "CFS1": 1.7, # wood-only shear walls with cold-formed steel (no gypsum)
        "CFS2": 1.3, # "limited ductility" diagonal strap concentrically braced wall (better than conventional)
    }

    # ductile coupled walls is assumed to be classified as "CSW", which is ductile shear wall as it is the more conservative one
    # it is assumed that RM construction 1995 and after are "nominal ductility"

    def flowchart_2010(lfrs):

        R_d = Rd_lookup_table[lfrs]
        R_o = Ro_lookup_table[lfrs]

        # period estimation, using the more detailed moment-frame
        if lfrs == 'SMF':
            T_period = 0.085*(h_n**0.75)
        elif lfrs == 'CMF':
            T_period = 0.075*(h_n**0.75)
        elif lfrs == 'SBF':
            T_period = 0.025*h_n
        else:
            T_period = 0.05*(h_n**0.75)

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
    Rd_lookup_table = {
        "WLF-P9": 1.0, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections or shear walls
        "WLF": 1.0, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections or shear walls
        "WPB": 1.0, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections or shear walls
        "SMF": 5.0, # ductile mrf
        "SBF": 3.0, # ductile braced frame
        "SLF": 1.0, # non-ductile steel frame assumed, other category
        "SCW": 2.0, # assuming wall controls, designed to moderate ductility
        "SIW": 2.0, # assuming wall controls, designed to moderate ductility
        "CMF": 4.0, # ductile mrf
        "CSW": 3.5, # ductile shear wall
        "CIW": 2.0, # assuming wall controls, designed to moderate ductility
        "PCW": 2.5, # nominal ductility
        "PCF1": 2.0, # assuming wall controls, designed to moderate ductility
        "PCF2": 2.5, # nominal ductility frame
        "RML": 3.0 if built_after_2015 else 1.5, # "ductile" RM depending on construction year
        "RMC": 3.0 if built_after_2015 else 1.5, # "ductile" RM depending on construction year 
        "URM": 1.0,
        "CFS1": 2.5, # wood-only shear walls with cold-formed steel (no gypsum)
        "CFS2": 1.9, # "limited ductility" diagonal strap concentrically braced wall (better than conventional)
    }
    
    # overstrength
    Ro_lookup_table = {
        "WLF-P9": 1.0, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections
        "WLF": 1.0, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections
        "WPB": 1.0, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections
        "SMF": 1.5, # ductile mrf
        "SBF": 1.3, # ductile braced frame
        "SLF": 1.0, # non-ductile steel frame assumed, other category
        "SCW": 1.4, # assuming wall controls, moderate ductility
        "SIW": 1.4, # assuming wall controls, moderate ductility
        "CMF": 1.7, # ductile mrf
        "CSW": 1.6, # ductile shear wall
        "CIW": 1.4, # assuming wall controls, moderate ductility
        "PCW": 1.4, # nominal ductility
        "PCF1": 1.4, # assuming wall controls, moderate ductility
        "PCF2": 1.4, # nominal ductility frame
        "RML": 1.5, # 
        "RMC": 1.5, # 
        "URM": 1.0,
        "CFS1": 1.7, # wood-only shear walls with cold-formed steel (no gypsum)
        "CFS2": 1.3, # "limited ductility" diagonal strap concentrically braced wall (better than conventional)
    }

    # ductile coupled walls is assumed to be classified as "CSW", which is ductile shear wall as it is the more conservative one
    # it is assumed that RM construction 2015 and after are "ductile"

    def flowchart_2015(lfrs):

        R_d = Rd_lookup_table[lfrs]
        R_o = Ro_lookup_table[lfrs]

        # period estimation, using the more detailed moment-frame

        # 2015 has a specific estimation to allow for the lengthening
        # of periods for single-story buildings with steel deck or wood roof diaphragms
        # presumably for warehouse/gathering hall type buildings. The lengthening 
        # is based on the shortest bay length
        # TODO: currently omitted

        if lfrs == 'SMF':
            T_period = 0.085*(h_n**0.75)
        elif lfrs == 'CMF':
            T_period = 0.075*(h_n**0.75)
        elif lfrs == 'SBF':
            T_period = 0.025*h_n
        else:
            T_period = 0.05*(h_n**0.75)

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
    Rd_lookup_table = {
        "WLF-P9": 1.0, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections or shear walls
        "WLF": 1.0, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections or shear walls
        "WPB": 1.0, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections or shear walls
        "SMF": 5.0, # ductile mrf
        "SBF": 3.0, # ductile braced frame
        "SLF": 1.0, # non-ductile steel frame assumed, other category
        "SCW": 2.0, # assuming wall controls, designed to moderate ductility
        "SIW": 2.0, # assuming wall controls, designed to moderate ductility
        "CMF": 4.0, # ductile mrf
        "CSW": 3.5, # ductile shear wall
        "CIW": 2.0, # assuming wall controls, designed to moderate ductility
        "PCW": 2.5, # nominal ductility
        "PCF1": 2.0, # assuming wall controls, designed to moderate ductility
        "PCF2": 2.5, # nominal ductility frame
        "RML": 3.0 if built_after_2015 else 1.5, # "ductile" RM depending on construction year
        "RMC": 3.0 if built_after_2015 else 1.5, # "ductile" RM depending on construction year 
        "URM": 1.0,
        "CFS1": 2.5, # wood-only shear walls with cold-formed steel (no gypsum)
        "CFS2": 1.9, # "limited ductility" diagonal strap concentrically braced wall (better than conventional)
    }
    
    # overstrength
    Ro_lookup_table = {
        "WLF-P9": 1.0, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections
        "WLF": 1.0, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections
        "WPB": 1.0, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections
        "SMF": 1.5, # ductile mrf
        "SBF": 1.3, # ductile braced frame
        "SLF": 1.0, # non-ductile steel frame assumed, other category
        "SCW": 1.4, # assuming wall controls, moderate ductility
        "SIW": 1.4, # assuming wall controls, moderate ductility
        "CMF": 1.7, # ductile mrf
        "CSW": 1.6, # ductile shear wall
        "CIW": 1.4, # assuming wall controls, moderate ductility
        "PCW": 1.4, # nominal ductility
        "PCF1": 1.4, # assuming wall controls, moderate ductility
        "PCF2": 1.4, # nominal ductility frame
        "RML": 1.5, # 
        "RMC": 1.5, # 
        "URM": 1.0,
        "CFS1": 1.7, # wood-only shear walls with cold-formed steel (no gypsum)
        "CFS2": 1.3, # "limited ductility" diagonal strap concentrically braced wall (better than conventional)
    }

    # ductile coupled walls is assumed to be classified as "CSW", which is ductile shear wall as it is the more conservative one
    # it is assumed that RM construction 2015 and after are "ductile"

    def flowchart_2020(lfrs):

        R_d = Rd_lookup_table[lfrs]
        R_o = Ro_lookup_table[lfrs]

        # period estimation, using the more detailed moment-frame

        # 2015 has a specific estimation to allow for the lengthening
        # of periods for single-story buildings with steel deck or wood roof diaphragms
        # presumably for warehouse/gathering hall type buildings. The lengthening 
        # is based on the shortest bay length
        # TODO: currently omitted

        if lfrs == 'SMF':
            T_period = 0.085*(h_n**0.75)
        elif lfrs == 'CMF':
            T_period = 0.075*(h_n**0.75)
        elif lfrs == 'SBF':
            T_period = 0.025*h_n
        else:
            T_period = 0.05*(h_n**0.75)

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
}

# TODO: condense repeated functions
# R factor

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


