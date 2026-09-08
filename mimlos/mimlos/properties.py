import numpy as np
import pandas as pd

NBCC_YEARS = np.array([1941, 1953, 1960, 1965, 1970, 1975, 1977, 1980, 1985, 1990, 1995, 2005, 2010, 2015, 2020, 2025])

def determine_effective_nbcc_year(original_year_series, renovation_year_series):
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

    renovation_year_series: pd.Series
        Actual year of latest major renovation

    Returns
    -------
    pd.Series of the effective code year.
    '''
    nbcc_code_year_original = original_year_series.fillna(1941)
    renovation_year = renovation_year_series.fillna(1941)

    effective_construction_year = np.maximum(nbcc_code_year_original, renovation_year)

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

    The strength here is taken as the controlling condition between the two.

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
    lfrs_ns = row["Seismic Force Resisting System in the North-South Direction"]
    lfrs_ew = row["Seismic Force Resisting System in the East-West Direction"]

    # assume that all "frame-with-wall" system have a non load-bearing wall
    bearing_wall_systems = ['CSW', 'PCW', 'RML', 'RMC', 'URM']

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

    The strength here is taken as the controlling condition between the two.

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

NBCC_VS_CALCULATORS = {
    1941: vs_nbcc_1941,
    1953: vs_nbcc_1953,
    1960: vs_nbcc_1953, # seismic provisions did not change
    1965: vs_nbcc_1965,
}
