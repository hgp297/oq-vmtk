from pathlib import Path
import pandas as pd
import numpy as np
from . import properties

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

        # TODO: weight function
        # TODO: distribution of forces
        # TODO: load factors, before 1965 working stress design was used
        # TODO: overstrength

        
# TODO: raise flag if irregularity and post-disaster

        


### validation/assertions



