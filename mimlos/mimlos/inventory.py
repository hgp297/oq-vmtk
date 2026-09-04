from pathlib import Path
import pandas as pd


class Inventory:
    """
    Manage the inventory database for a portfolio.

    This class contains the main data, including inputs with survey
    information, translations to modelling parameters.

    Methods hosted include preprocessing functions and step-by-step
    functions to go through the assessment methodology.

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
        self.df_clean = self.filter_reviewed(df)

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

    def filter_reviewed(self, inv_df: pd.DataFrame):
        '''
        Filter the data to only engineer-reviewed buildings to ensure
        that the df row are more completely surveyed.

        Parameters
        ----------
        inv_df: Dataframe
            Raw unprocessed inventory dataframe

        Returns
        -------
        pd.DataFrame with only non-empty "Approved?" rows.
        '''
        return inv_df[inv_df['Approved?'].notna()]