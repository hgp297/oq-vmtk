import numpy as np
import pandas as pd
from openquake.vmtk.units import units

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

def determine_code_strength(row):
    '''
    Dispatcher function to redirect to the correct code calculation
    function based on the "effective_nbcc_year" function.

    Dispatcher is located at global variable NBCC_VS_CALCULATORS
    
    '''
    try:
        return NBCC_VS_CALCULATORS[row["effective_nbcc_year"]](row)
    except KeyError:
        raise ValueError(
            f"Unsupported code year: {row['effective_nbcc_year']}"
        )

def vs_nbcc_1941(row):
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

    return vs_ns, vs_ew

def vs_nbcc_1953(row, seismic_zone=3):
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
        return 4*vs_ns, 4*vs_ew
    elif seismic_zone == 2:
        return 2*vs_ns, 2*vs_ew
    elif seismic_zone == 1:
        return vs_ns, vs_ew
    else:
        return 0.0, 0.0

def vs_nbcc_1965(row, seismic_zone=3):
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
        return 4*vs_ns, 4*vs_ew
    elif seismic_zone == 2:
        return 2*vs_ns, 2*vs_ew
    elif seismic_zone == 1:
        return vs_ns, vs_ew
    else:
        return 0.0, 0.0

    # TODO: 1965 NBCC required that if the building had significant
    # torsional irregularity, design computed torsional moment would
    # be doubled

def vs_nbcc_1970(row, seismic_zone=3):
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
        Building height in metres. If not provided, will be estimated with 13ft stories. 

    row["Floors Above Grade"]: numeric
        number of stories above grade

    row['"Original" Building Importance Factor Ie']
        Importance factor assigned to the building in its original design

    row["Site Class"]: str
        modern-assessed site class of the building

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

    # TODO: temporarily stand-in square building
    D_ft = (plan_area**0.5)/units.ft # divide to go from m to ft

    # TODO: temporarily estimate bldg_height if not available
    # estimate as 13 ft stories
    if np.isnan(bldg_height):
        h_n_ft = 13.0 * number_of_stories
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

def vs_nbcc_1975(row, seismic_zone=3):
    '''
    Calculate the lateral force coefficient based on NBC1975, as outlined
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
        Building height in metres. If not provided, will be estimated with 13ft stories. 

    row["Floors Above Grade"]: numeric
        number of stories above grade

    row['"Original" Building Importance Factor Ie']
        Importance factor assigned to the building in its original design

    row["Site Class"]: str
        modern-assessed site class of the building

    seismic_zone: int
        Seismic region as determined by the map in the Table of Climactic Data of NBC 1975
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

    # TODO: temporarily stand-in square building
    D_ft = (plan_area**0.5)/units.ft # divide to go from m to ft

    # TODO: temporarily estimate bldg_height if not available
    # estimate as 13 ft stories
    if np.isnan(bldg_height):
        h_n_ft = 13.0 * number_of_stories
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
    # ductile systems weren't provided until 1973
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

def vs_nbcc_1980(row, seismic_zone=3):
    '''
    Calculate the lateral force coefficient based on NBC1980, as outlined
    in Section 4.1.9. This edition is nearly identical to the 1975 edition, with 
    the only change being to the S factor and switch to SI units. 
    As such, documentation will not be revised.

    The coefficient is NOT yet multiplied with the building weight. Vs is taken
    from Sentence (4) of the section above.

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
        Building height in metres. If not provided, will be estimated with 13ft stories. 

    row["Floors Above Grade"]: numeric
        number of stories above grade

    row['"Original" Building Importance Factor Ie']
        Importance factor assigned to the building in its original design

    row["Site Class"]: str
        modern-assessed site class of the building

    seismic_zone: int
        Seismic region as determined by the map in the Table of Climactic Data of NBC 1975
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

    # TODO: temporarily stand-in square building
    D_length = (plan_area**0.5)

    # TODO: temporarily estimate bldg_height if not available
    # estimate as 13 ft stories
    if np.isnan(bldg_height):
        h_n_ft = 13.0 * units.ft * number_of_stories
    else:
        h_n_ft = bldg_height

    
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
    # ductile systems weren't provided until 1973
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
    def flowchart_1980(lfrs, D_ft):
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
            T_period = 0.09*h_n_ft/(D_ft**0.5)

        # seismic response factor
        # only change from 1975
        S_factor = np.minimum(0.5/(T_period**(1/2)), 1.0)

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

    
    vs_ns = flowchart_1980(lfrs_ns, D_length)
    vs_ew = flowchart_1980(lfrs_ew, D_length)

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

def vs_nbcc_1985(row, seismic_hazard_params=None):
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
        Building height in metres. If not provided, will be estimated with 13ft stories. 

    row["Floors Above Grade"]: numeric
        number of stories above grade

    row['"Original" Building Importance Factor Ie']
        Importance factor assigned to the building in its original design

    row["Site Class"]: str
        modern-assessed site class of the building

    seismic_hazard_params: None, Dictionary
        1985 NBCC introduced finer control for hazard, acknowledging differences in spectral
        shape and probabilistically aiming for a 495-year return period. Thus, the hazard dictionary
        requires the following inputs:
        - Za: seismic acceleration zone
        - Zv: seismic velocity zone
        - v: zonal velocity ratio
        (zonal acceleration ratio is implicitly used in S factor)

        If None is provided, the assumed values are for Vancouver, British Columbia

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

    # defaults for Vancouver
    if seismic_hazard_params is None:
        Za = 4
        Zv = 4
        v_ratio = 0.20

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
    # estimate as 13 ft stories
    if np.isnan(bldg_height):
        h_n_ft = 13.0 * units.ft * number_of_stories
    else:
        h_n_ft = bldg_height

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
    def flowchart_1985(lfrs, D_ft):

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
            T_period = 0.09*h_n_ft/(D_ft**0.5)

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
    
def vs_nbcc_1990(row, seismic_hazard_params=None):
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
        Building height in metres. If not provided, will be estimated with 13ft stories. 

    row["Floors Above Grade"]: numeric
        number of stories above grade

    row['"Original" Building Importance Factor Ie']
        Importance factor assigned to the building in its original design

    row["Site Class"]: str
        modern-assessed site class of the building

    seismic_hazard_params: None, Dictionary
        1990 NBCC introduced finer control for hazard, acknowledging differences in spectral
        shape and probabilistically aiming for a 495-year return period. Thus, the hazard dictionary
        requires the following inputs:
        - Za: seismic acceleration zone
        - Zv: seismic velocity zone
        - v: zonal velocity ratio
        (zonal acceleration ratio is implicitly used in S factor)

        If None is provided, the assumed values are for Vancouver, British Columbia

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

    # defaults for Vancouver
    if seismic_hazard_params is None:
        Za = 4
        Zv = 4
        v_ratio = 0.20

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
    # estimate as 13 ft stories
    if np.isnan(bldg_height):
        h_n_ft = 13.0 * units.ft * number_of_stories
    else:
        h_n_ft = bldg_height

    # "A ductile moment-resisting space frame is a space frame that is designed to resist
    # all the specified seismic forces and that, in addition, has adequate ductility or
    # energy-absorptive capacity."
    # TODO: request review on this
    # ductile systems weren't provided until 1973
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
    R_lookup_table_flexural = {
        "WLF-P9": 1.5, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections
        "WLF": 1.5, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections
        "WPB": 1.5, # assuming CAN/CSA-O86.1-M compliant, but not ductile connections
        "SMF": 4.0, # ductile mrf
        "SBF": 3.0, # ductile braced frame
        "SLF": 1.5, # non-ductile steel frame assumed, other category
        "SCW": 3.5 if flexure_controlled else 2.0, # assuming wall controls
        "SIW": 3.5 if flexure_controlled else 2.0, # assuming wall controls
        "CMF": 4.0, # ductile mrf
        "CSW": 3.5 if flexure_controlled else 2.0, # either ductile flexural wall or nominal ductility (shear-controlled)
        "CIW": 3.5 if flexure_controlled else 2.0, # assuming wall controls
        "PCW": 3.5, # nominal ductility
        "PCF1": 2.0, # nominal ductility frame or wall (both same R factor)
        "PCF2": 2.0, # nominal ductility frame
        "RML": 1.5, 
        "RMC": 1.5,
        "URM": 1.0,
        "CFS1": 1.5, # non-ductile steel frame assumed, other category
        "CFS2": 1.5
    }

    # D_ft could change per-direction
    def flowchart_1990(lfrs, D_ft):

        R_factor = R_lookup_table_flexural[lfrs]

        # period estimation
        if lfrs in ductile_moment_frames:
            T_period = 0.1 * number_of_stories
        else:
            T_period = 0.09*h_n_ft/(D_ft**0.5)

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
    
NBCC_VS_CALCULATORS = {
    1941: vs_nbcc_1941,
    1953: vs_nbcc_1953,
    1960: vs_nbcc_1953, # seismic provisions did not change
    1965: vs_nbcc_1965,
    1970: vs_nbcc_1970,
    1975: vs_nbcc_1975,
    1977: vs_nbcc_1975, # seismic provisions did not change
    1980: vs_nbcc_1980, 
    1985: vs_nbcc_1985, 
    1990: vs_nbcc_1990, 
}
