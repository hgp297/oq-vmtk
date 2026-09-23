from pathlib import Path
import pandas as pd
import numpy as np
from . import nbcc
from openquake.vmtk.units import units
from . import hazus

DESIGN_HAZARD_PATH = Path(__file__).resolve().parent / "data" / "hazard" / "design"


class Inventory:
    """
    Manage the inventory database for a portfolio.

    This class contains the main data, including inputs with survey
    information, translations to modelling parameters.

    Methods hosted include preprocessing functions and step-by-step
    functions to go through the assessment methodology.

    Field names are currently hard-coded to be of the formatting
    used by the City of Vancouver's Real Estate and Facilities Management
    (CoV REFM) building survey sheet.

    TODO: description of methodology
    
    Attributes
    ----------
    df : dataframe
        Original raw dataframe read in from a path.
        TODO: specify rules/data validation/assertion

    Methods
    -------

    load_from_csv(cls, path: str | Path) -> "Inventory"
        Load the csv from a path
    """

    # load 2020 hazard design values for the class
    VANCOUVER_CITY_HALL_DESIGN_HAZARD_2020 = pd.read_csv(
        DESIGN_HAZARD_PATH / '2020_BC_Vancouver_cityhall_Hoteldeville.csv',
        encoding="cp863")
    VANCOUVER_CITY_HALL_DESIGN_HAZARD_2020.rename(
        columns={VANCOUVER_CITY_HALL_DESIGN_HAZARD_2020.columns[0]: "site_designation"}, inplace=True)
    VANCOUVER_CITY_HALL_DESIGN_HAZARD_2020.set_index('site_designation', inplace=True)

    VANCOUVER_GRANVILLE41_DESIGN_HAZARD_2020 = pd.read_csv(
        DESIGN_HAZARD_PATH / '2020_BC_Vancouver_Granville&41Ave_rueGranvilleet41eav.csv',
        encoding="cp863")
    VANCOUVER_GRANVILLE41_DESIGN_HAZARD_2020.rename(
            columns={VANCOUVER_GRANVILLE41_DESIGN_HAZARD_2020.columns[0]: "site_designation"}, inplace=True)
    VANCOUVER_GRANVILLE41_DESIGN_HAZARD_2020.set_index('site_designation', inplace=True)

    
    # load 2025 hazard design values for the class
    VANCOUVER_CITY_HALL_DESIGN_HAZARD_2025 = pd.read_csv(
        DESIGN_HAZARD_PATH / '2025_BC_Vancouver_cityhall_Hoteldeville.csv',
        encoding="cp863")
    VANCOUVER_CITY_HALL_DESIGN_HAZARD_2025.rename(
        columns={VANCOUVER_CITY_HALL_DESIGN_HAZARD_2025.columns[0]: "site_designation"}, inplace=True)
    VANCOUVER_CITY_HALL_DESIGN_HAZARD_2025.set_index('site_designation', inplace=True)

    VANCOUVER_GRANVILLE41_DESIGN_HAZARD_2025 = pd.read_csv(
        DESIGN_HAZARD_PATH / '2025_BC_Vancouver_Granville&41Ave_rueGranvilleet41eav.csv',
        encoding="cp863")
    VANCOUVER_GRANVILLE41_DESIGN_HAZARD_2025.rename(
            columns={VANCOUVER_GRANVILLE41_DESIGN_HAZARD_2025.columns[0]: "site_designation"}, inplace=True)
    VANCOUVER_GRANVILLE41_DESIGN_HAZARD_2025.set_index('site_designation', inplace=True)

    def __init__(self, df: pd.DataFrame):
        self.df_raw = df
        self.inventory_df = self.filter_reviewed()

        

    @classmethod
    def load_from_csv(cls, path: str | Path) -> "Inventory":
        """
        Loads in the raw survey csv and stores it within the class

        Parameters
        ----------
        cls : Inventory class

        path : Path
            Pathlib style path to the csv, which should be exported from
            an refm-style sheet. Currently, it expects for the header row 
            to be the second row of the sheet (index 1).

        Returns
        -------
        Inventory.df : stores the raw survey as a DataFrame
        """
        path = Path(path)

        # hardcoded to have header as row 2 (index 1)
        # assert that the header row is not empty
        header = pd.read_csv(
            path,
            encoding="cp863",
            header=None,
            skiprows=1,
            nrows=1,
        ).iloc[0]

        assert header.notna().all(), "Header row contains empty fields"

        df = pd.read_csv(path, encoding='cp863', header=1)
        return cls(df)
    
    @classmethod
    def load_from_df(cls, df: pd.DataFrame) -> "Inventory":
        """
        Loads directly from a DataFrame

        Parameters
        ----------
        cls : Inventory class

        df : pd.DataFrame
            DataFrame compatible with the REFM format.

        Returns
        -------
        Inventory.df : stores the raw survey as a DataFrame
        """
        
        return cls(df)


### transformations (preprocessing)

    def filter_reviewed(self):
        '''
        Filter the data to only engineer-reviewed buildings to ensure
        that the df row are more completely surveyed.

        Parameters
        ----------
        self: Instance of Inventory. Should have df_raw attribute
        generated from csv.

        Returns
        -------
        pd.DataFrame with only non-empty "Approved?" rows.
        '''
        return self.df_raw[self.df_raw['Approved?'].notna()].copy()

    def latest_seismic_upgrade_year(self):
        '''
        Cleans the "Seismic Upgrade Year" field to only
        the latest year found.

        Cleans the "NBC Code Year (Original Building)" field to 
        the original NBCC year. Fill to construction year if pre-code.
        '''
        years = (
            self.inventory_df["Seismic Upgrade Year"]
            .astype("string")
            .str.extractall(r"(\d{4})")[0]
            .astype("Int64")
            .groupby(level=0)
            .max()
        )

        self.inventory_df["latest_seismic_upgrade_year"] = (
            years.reindex(self.inventory_df.index)
)

        self.inventory_df["original_nbcc_year"] = (
            pd.to_numeric(
                self.inventory_df["NBC Code Year (Original Building)"]
                .astype("string")
                .str.extract(r"(\d+\.?\d*)")[0],
                errors="coerce",
            )
            .fillna(
                pd.to_numeric(
                    self.inventory_df["Year Built"],
                    errors="coerce",
                )
            )
            .astype("Int64")
        )

### calculation functions

    def preprocess_inventory(self):
        '''
        Various cleaning of "reviewed" data


        '''
        self.inventory_df = self.inventory_df.astype({'Floors Above Grade': int, 'Floors Below Grade': int})


    def estimate_Vs(self):
        '''
        Estimate the lateral strength of the building by using the 
        static lateral base shear estimate of the code at the time 
        of the construction or latest upgrade of the building.


        '''

        # clean upgrade year column
        self.latest_seismic_upgrade_year()

        # determine latest year of seismic code
        self.inventory_df['effective_nbcc_year'] = nbcc.determine_effective_nbcc_year(
            self.inventory_df["original_nbcc_year"],
            self.inventory_df["latest_seismic_upgrade_year"]
        )

        # self.inventory_df['Floors Above Grade'] = self.inventory_df['Floors Above Grade'].astype(int)

        # determine fundamental periods
        self.inventory_df['T_n'] = self.inventory_df.apply(
            nbcc.determine_period, axis=1
        ) 

        # calculate the original base shear
        def calc_design_base_shear_capacity(row):
            # determine hazard based on year, site class, and location
            # store as dict to pass to code-calculation function
            seismic_hazard_dict = self.determine_seismic_hazard_params(
                year=int(row['effective_nbcc_year']),
                site_class=row["Site Class"],
                location='vancouver_city_hall'
            )

            # calculate code strength 
            return nbcc.determine_code_strength(row, seismic_hazard_params=seismic_hazard_dict)

        # apply to every row
        self.inventory_df['original_nbcc_unfactored_Vd'] = self.inventory_df.apply(
            calc_design_base_shear_capacity, axis=1
        )

        # factor load using best guess at working stress/ultimate stress/limit state design
        # at the time

        # we assume that SEG adjustments will account for overstrength
        self.inventory_df['original_nbcc_factored_Ve'] = self.inventory_df.apply(
            nbcc.factor_lateral_earthquake_load, axis=1
        )
        self.inventory_df['SEG_adjusted_NBCC_factored_Ve'] = self.inventory_df.apply(
            nbcc.adjust_base_shear_capacity_SEG, axis=1
        )

        # TODO: this is used in SEG to trigger higher tier
        def calc_SEG_base_shear_demand(row):
            # uses the 2025 NBCC seismic base shear demand for evaluation
            # VQE = kappa * alpha_q * V_N
            # determine hazard based on year, site class, and location
            # store as dict to pass to code-calculation function

            seismic_hazard_dict = self.determine_seismic_hazard_params(
                year=2025,
                site_class=row["Site Class"],
                location='vancouver_city_hall'
            )

            # calculate code strength V_N
            return nbcc.vs_nbcc_2025(row, seismic_hazard_params=seismic_hazard_dict, historical_mode=True)

        # considering ductility deficiencies of previous code versions
        # what is the required Vd according to 2025 NBCC to be okay
        self.inventory_df['NBCC_2025_unfactored_Vd'] = self.inventory_df.apply(
                calc_SEG_base_shear_demand, axis=1
            )

        # distribute the factored forces across stories using 2025 code methodology
        self.inventory_df['Vd_distributed_SEG_adjusted_NBCC_factored_Ve'] = self.inventory_df.apply(
            nbcc.distribute_story_shear, axis=1
        )

    def estimate_capacities(self):
        '''
        Estimate the strength and displacement capacities of the building
        using the code-level and typology of the building, along with
        the original design strength.
        '''
        self.inventory_df['Vdj_story_design_shear'] = self.inventory_df['Vd_distributed_SEG_adjusted_NBCC_factored_Ve'].copy()
        self.inventory_df['Vyj_story_yield_shear'] = [tuple(x * y for x, y in zip(t1, t2)) 
                                    for t1, t2 in zip(
                                        self.inventory_df['Vdj_story_design_shear'], 
                                        self.inventory_df['hazus_Omega_y_yield_overstrength'])]
        self.inventory_df['Vpj_story_peak_shear'] = [tuple(x * y for x, y in zip(t1, t2)) 
                                    for t1, t2 in zip(
                                        self.inventory_df['Vyj_story_yield_shear'], 
                                        self.inventory_df['hazus_Omega_p_peak_overstrength'])]

        # if no building height, h_j is 3.5 meters per floor
        # if building height, divide it by n_stories

        # self.inventory_df['h_j'] = np.where(
        #     self.inventory_df["Building Height (Total Height Above Ground (m) to Roof Slab)"].isna(), 
        #     3.5*units.m*np.ones(self.inventory_df["Floors Above Grade"]), 
        #     (self.inventory_df["Building Height (Total Height Above Ground (m) to Roof Slab)"]/
        #      self.inventory_df["Floors Above Grade"])*units.m*np.ones(self.inventory_df["Floors Above Grade"].astype(int))
        #      )   

    def estimate_loads(self):
        '''
        Estimate weight of the building based on SEG's weight of content
        based on Occupancy Type and SFRS. Dead load estimation for floor and roof systems
        are based on CISC Table on 7-69. 

        Weight here are meant to be estimations of actual weight of the content, not
        design loads or factored loads.
        '''

        self.inventory_df['weight_x_N'] = self.inventory_df.apply(
            nbcc.determine_content_weight, axis=1
        )

        self.inventory_df['vertical_load_x_Pa'] = (self.inventory_df['weight_x_N']/
                                                self.inventory_df["Ground Floor Plan Area (sq.m.)"]*units.m2)



    def determine_code_level(self):
        '''
        From the construction year and the archetype of the building,
        determine the level of code construction for the building.
        
        This is used to redispatch to ductility values as outlined by
        HAZUS.
        
        Definitions for pre-, low-, and moderate code match that 
        of the 2025 UBC Seismic Risk study performed by Arup. Definitions
        of benchmark high-code are set by the Seismic Evaluation Guidelines
        
        Parameters
        -------------
        row['effective_nbcc_year']: int
                year of code design
            
        row["Seismic Force Resisting System in the North-South Direction"]: str
            modern-classification of the n-s lateral force resisting system in the 
            NRC Seismic Evaluation Guidelines typologies

        row["Seismic Force Resisting System in the East-West Direction"]: str
            modern-classification of the e-w lateral force resisting system in the 
            NRC Seismic Evaluation Guidelines typologies

        Returns
        -------------
        '''
        code_year = self.inventory_df["effective_nbcc_year"]
        lfrs_ns = self.inventory_df["Seismic Force Resisting System in the North-South Direction"]
        lfrs_ew = self.inventory_df["Seismic Force Resisting System in the East-West Direction"]

        def code_lookup(lfrs_series):
            ylc, ymc, yhc = nbcc.get_code_bins(lfrs_series)
                
            code_level = np.select(
                [
                    code_year < ylc,
                    code_year < ymc,
                    code_year < yhc,
                    ],
                [
                    "pre-code",
                    "low-code",
                    "moderate-code",
                ],
                default="high-code",
            )
            return [str(x) for x in code_level]
        
        code_level_ns = code_lookup(lfrs_ns)
        code_level_ew = code_lookup(lfrs_ew)
        self.inventory_df['code_level'] = list(zip(code_level_ns, code_level_ew))

        

    def determine_hazus_parameters(self):
        '''
        Returns HAZUS parameters

        C_s, design strength as fraction of building weight

        overstrength_ratios
        gamma: yield, ratio of yield to design strength
        lambda: ultimate, ratio of ultimate to yield strength

        ductility factor mu
        ratio of ultimate displacement to lambda * yield displacement
        '''
        
        # determine code level for each lfrs per direction
        self.determine_code_level()

        lfrs_ns = self.inventory_df["Seismic Force Resisting System in the North-South Direction"]
        lfrs_ew = self.inventory_df["Seismic Force Resisting System in the East-West Direction"]

        # determine FEMA/Hazus typology
        fema_ns_stem = lfrs_ns.map(hazus.SEG_TO_FEMA_TYPOLOGY)
        fema_ew_stem = lfrs_ew.map(hazus.SEG_TO_FEMA_TYPOLOGY)

        height_conditions = [
            (self.inventory_df['Floors Above Grade'] < 4),
            (self.inventory_df['Floors Above Grade'] < 7) & (self.inventory_df['Floors Above Grade'] >= 4),
            (self.inventory_df['Floors Above Grade'] > 7),
        ]

        height_archetypes = ['L', 'M', 'H']
        no_height_delin = ["W1", "W2", "S3", "PC1"]

        height_suffix = np.select(height_conditions, height_archetypes, default="")
        self.inventory_df["FEMA_lfrs_ns"] = np.where(
            fema_ns_stem.isin(no_height_delin),
            fema_ns_stem,
            fema_ns_stem + height_suffix,
        )

        self.inventory_df["FEMA_lfrs_ew"] = np.where(
            fema_ew_stem.isin(no_height_delin),
            fema_ew_stem,
            fema_ew_stem + height_suffix,
        )

        # map hazus parameters
        self.inventory_df["hazus_Cs_design_strength"] = [
            (
                hazus.HAZUS_DESIGN_STRENGTH_TABLE.loc[code[0], lfrs_ns],
                hazus.HAZUS_DESIGN_STRENGTH_TABLE.loc[code[1], lfrs_ew],
            )
            for code, lfrs_ns, lfrs_ew
            in zip(self.inventory_df["code_level"], 
                   self.inventory_df["FEMA_lfrs_ns"], 
                   self.inventory_df["FEMA_lfrs_ew"])
        ]

        self.inventory_df["hazus_Omega_y_yield_overstrength"] = [
            (
                hazus.HAZUS_PUSHOVER_TABLE.loc["gamma", lfrs_ns],
                hazus.HAZUS_PUSHOVER_TABLE.loc["gamma", lfrs_ew],
            )
            for lfrs_ns, lfrs_ew
            in zip(self.inventory_df["FEMA_lfrs_ns"], 
                   self.inventory_df["FEMA_lfrs_ew"])
        ]


        self.inventory_df["hazus_Omega_p_peak_overstrength"] = [
            (
                hazus.HAZUS_PUSHOVER_TABLE.loc["lambda", lfrs_ns],
                hazus.HAZUS_PUSHOVER_TABLE.loc["lambda", lfrs_ew],
            )
            for lfrs_ns, lfrs_ew
            in zip(self.inventory_df["FEMA_lfrs_ns"], 
                    self.inventory_df["FEMA_lfrs_ew"])
        ]

        self.inventory_df["hazus_mu_ductility"] = [
            (
                hazus.HAZUS_DUCTILITY_TABLE.loc[code[0], lfrs_ns],
                hazus.HAZUS_DUCTILITY_TABLE.loc[code[1], lfrs_ew],
            )
            for code, lfrs_ns, lfrs_ew
            in zip(self.inventory_df["code_level"], 
                   self.inventory_df["FEMA_lfrs_ns"], 
                   self.inventory_df["FEMA_lfrs_ew"])
        ]


            

    def determine_seismic_hazard_params(self, year, site_class, location):
        '''
        Temporary holder function to determine seismic hazard.
        
        If available, read it. If not, default to Vancouver
        '''
        if year >= 2025:
            if location == 'vancouver_city_hall':
                return self._get_vancouver_hazard_post2020(
                    hazard_df=self.VANCOUVER_CITY_HALL_DESIGN_HAZARD_2025, 
                    site_class=site_class)
            elif location == 'vancouver_granville41':
                return self._get_vancouver_hazard_post2020(
                    hazard_df=self.VANCOUVER_GRANVILLE41_DESIGN_HAZARD_2025, 
                    site_class=site_class)
        elif year > 2015:
            if location == 'vancouver_city_hall':
                return self._get_vancouver_hazard_post2020(
                    hazard_df=self.VANCOUVER_CITY_HALL_DESIGN_HAZARD_2020, 
                    site_class=site_class)
            elif location == 'vancouver_granville41':
                return self._get_vancouver_hazard_post2020(
                    hazard_df=self.VANCOUVER_GRANVILLE41_DESIGN_HAZARD_2020, 
                    site_class=site_class)
        else:
            if location == 'vancouver_city_hall':
                return self._get_vancouver_hazard_pre2020(year, loc='city_hall')
            elif location == 'vancouver_granville41':
                return self._get_vancouver_hazard_pre2020(year, loc='granville_41')
            else:
                raise ValueError(
                    "Unsupported analysis location."
                )

        # TODO: stiffness-controlled buildings

    def _get_vancouver_hazard_pre2020(self, year, loc='city_hall'):
        '''
        Returns the site-specific hazard parameters for Vancouver, British Columbia.

        This code has 3 eras:
            Seismic Zone era, where only 4 distinct zones were identified, leading to 
            different multipliers on the base shear.

            Za_Zv_era, where calculation was based on acceleration and velocity ratios 
            per zone.

            Early and mid site-specific era, which tabulated spectral accelerations 
            pre-adjustments for site effects (Fa, Fv), which are handled
            within the Vs calculators.

        Parameters
        ----------
        year: int
            NBCC year, redirects to correct set of parameters
        loc: str
            'city_hall' and 'granville_41' are supported. Return hazard at site.
        Returns
        -------
        dict: 
            Dictionary containing relevant seismic hazard parameters
        '''

        seismic_zone_era = [1941, 1953, 1960, 1965, 1970, 1975, 1977, 1980]
        za_zv_era = [1985, 1990, 1995]
        early_site_era = [2005, 2010]
        mid_site_era = [2015]

        if loc == 'city_hall':
            if year in seismic_zone_era:
                seismic_hazard_params = {
                    'seismic_zone': 3
                }
            elif year in za_zv_era:
                seismic_hazard_params = {
                    'Za' : 4,
                    'Zv' : 4,
                    'v_ratio' : 0.20
                }
            elif year in early_site_era:
                seismic_hazard_params = {
                    'Sa_0p2' : 0.94,
                    'Sa_0p5' : 0.64,
                    'Sa_1p0' : 0.33,
                    'Sa_2p0' : 0.17,
                    'Sa_pga' : 0.46,
                }
            elif year in mid_site_era:
                seismic_hazard_params = {
                    'Sa_0p2' : 0.848,
                    'Sa_0p5' : 0.751,
                    'Sa_1p0' : 0.425,
                    'Sa_2p0' : 0.257,
                    'Sa_5p0' : 0.080,
                    'Sa_10p0': 0.029,
                    'Sa_pga' : 0.369,
                    'Sa_pgv' : 0.553,
                }

            elif loc == 'granville_41':
                if year in seismic_zone_era:
                    seismic_hazard_params = {
                        'seismic_zone': 3
                    }
                elif year in za_zv_era:
                    seismic_hazard_params = {
                        'Za' : 4,
                        'Zv' : 4,
                        'v_ratio' : 0.20
                    }
                elif year in early_site_era:
                    seismic_hazard_params = {
                        'Sa_0p2' : 0.95,
                        'Sa_0p5' : 0.65,
                        'Sa_1p0' : 0.34,
                        'Sa_2p0' : 0.17,
                        'Sa_pga' : 0.47,
                    }
                elif year in mid_site_era:
                    seismic_hazard_params = {
                        'Sa_0p2' : 0.863,
                        'Sa_0p5' : 0.765,
                        'Sa_1p0' : 0.432,
                        'Sa_2p0' : 0.261,
                        'Sa_5p0' : 0.081,
                        'Sa_10p0': 0.029,
                        'Sa_pga' : 0.375,
                        'Sa_pgv' : 0.563,
                    }
        return seismic_hazard_params

    def _get_vancouver_hazard_post2020(self, hazard_df, site_class):
        '''
        Use the appropriate pre-calculated site-specific Sa parameters depending on site class

        Post-2020, Fa and Fv coefficients were no longer used, but each location had
        pre-calculated values tabulated and available at NRC Seismic Hazard Data 
        repository.
        Parameters
        ----------
        hazard_df: pd.DataFrame
            Cleaned hazard DataFrame read from the Seismic Hazard Data for the specific location
            Accessible from https://doi.org/10.4224/nqzr-dz38.

            Header row must be the Site Designation row.

        site_class: str
            Site class letter A through E
        
        Returns
        -------
        dict: 
            Dictionary containing relevant seismic hazard parameters
        '''
        site_designation = 'X' + str(site_class)
        seismic_hazard_params = {
            'Sa_0p2' : hazard_df.at[site_designation, '2%/50 Sa(0.2)'],
            'Sa_0p5' : hazard_df.at[site_designation, '2%/50 Sa(0.5)'],
            'Sa_1p0' : hazard_df.at[site_designation, '2%/50 Sa(1.0)'],
            'Sa_2p0' : hazard_df.at[site_designation, '2%/50 Sa(2.0)'],
            'Sa_5p0' : hazard_df.at[site_designation, '2%/50 Sa(5.0)'],
            'Sa_10p0': hazard_df.at[site_designation, '2%/50 Sa(10.0)'],
            'Sa_pga' : hazard_df.at[site_designation, '2%/50  PGA'],
            'Sa_pgv' : hazard_df.at[site_designation, '2%/50  PGV'],
        }

        return seismic_hazard_params

# TODO: document that this has not accounted for low-seismic exemptions often seen in NBCC     
# TODO: raise flag if irregularity and post-disaster

        


### validation/assertions



