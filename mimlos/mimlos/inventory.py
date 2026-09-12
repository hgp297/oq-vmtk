from pathlib import Path
import pandas as pd
import numpy as np
from . import properties

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
        years = self.inventory_df["Seismic Upgrade Year"].astype("string").str.findall(r"\d{4}")

        self.inventory_df["latest_seismic_upgrade_year"] = (
            years
            .explode()
            .astype("Int64")
            .groupby(level=0)
            .max()
            .reindex(self.inventory_df.index)
        )

        self.inventory_df["original_nbcc_year"] = pd.to_numeric(
            self.inventory_df["NBC Code Year (Original Building)"].str.extract(r"(\d+\.?\d*)")[0], errors="coerce"
        ).fillna(self.inventory_df["Year Built"]).astype("Int64")

### calculation functions

    def estimate_Vs(self):
        '''
        Estimate the lateral strength of the building by using the 
        static lateral base shear estimate of the code at the time 
        of the construction or latest upgrade of the building.


        '''

        # clean upgrade year column
        self.latest_seismic_upgrade_year()

        # determine latest year of seismic code
        self.inventory_df['effective_nbcc_year'] = properties.determine_effective_nbcc_year(
            self.inventory_df["original_nbcc_year"],
            self.inventory_df["latest_seismic_upgrade_year"]
        )

        # calculate the original base shear
        def calc_code_strength(row):
            # determine hazard based on year, site class, and location
            # store as dict to pass to code-calculation function
            seismic_hazard_dict = self.determine_seismic_hazard_params(
                year=int(row['effective_nbcc_year']),
                site_class=row["Site Class"],
                location='vancouver_city_hall'
            )

            # calculate code strength 
            return properties.determine_code_strength(row, seismic_hazard_params=seismic_hazard_dict)

        # apply to every row
        self.inventory_df['original_nbcc_unfactored_V'] = self.inventory_df.apply(
            calc_code_strength, axis=1
        )

        # factor load using best guess at working stress/ultimate stress/limit state design
        # at the time
        self.inventory_df['original_nbcc_factored_V'] = self.inventory_df.apply(
            properties.factor_lateral_earthquake_load, axis=1
        )

    def determine_seismic_hazard_params(self, year, site_class, location):
        '''
        Temporary holder function to determine seismic hazard.
        
        If available, read it. If not, default to Vancouver
        '''
        if year > 2015:
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


        # TODO: weight function
        # TODO: distribution of forces
        # TODO: load factors, before 1965 working stress design was used
        # TODO: overstrength
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



