from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
import sys


project_root = Path(__file__).resolve().parent.parent.parent  
sys.path.append(str(project_root))

from src.config.paths import (
    # ENTEROCOCCI_DATA_PATH,
    # METEOROLOGICAL_DATA_PATH,
    # SITE_METADATA_PATH,
    TRAINING_DATA_PATH
)

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class FeatureEngineer:
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path) if config_path else {}

    def _load_config(self, config_path: Path) -> Dict:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dictionary containing configuration parameters
        """
        logger.info(f"Loading configuration from {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise


    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features from the raw data.
        
        Args:
            data: Raw data
        
        Returns:
            Processed data with engineered features
        """
        logger.info("Engineering features")
        
        # Add new features here
        data = self.temporal_features(data)
        data = self.wind_shoretype_feature(data)
        data = self.lagged_enterococci_features(data)
        
        logger.info("Feature engineering complete")
        return data
    


    def temporal_features(self, df):
        # Convert DateTime column to datetime with the specified format
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='%m/%d/%y %H:%M')
        
        # Extracting month and week features
        df['YEAR'] = df['DateTime'].dt.year
        df['MONTH'] = df['DateTime'].dt.month
        df['WEEK'] = df['DateTime'].dt.isocalendar().week
        df['DAY_OF_WEEK'] = df['DateTime'].dt.dayofweek  # Monday=0, Sunday=6
        df['WEEKEND'] = df['DateTime'].dt.dayofweek.isin([5, 6]).astype(int)
        df['TIME_OF_DAY'] = df['DateTime'].dt.hour  # Extracting time of day

        # Define New Zealand public holidays as a dictionary with date ranges
        HOLIDAY_DATE_DATA = {
            'New Year\'s Day': ('01-01', '01-06'),
            'Waitangi Day': ('01-31', '02-10'),
            'Good Friday': ('04-14', '04-22'),
            'Easter Monday': ('04-17', '04-25'),
            'Anzac Day': ('04-20', '04-30'),
            'Queen\'s Birthday': ('05-29', '06-05'),
            'Labour Day': ('10-23', '10-30'),
            'Christmas Day': ('12-20', '12-30'),
            'Boxing Day': ('12-21', '12-31')
        }

        # Adding a binary event flag feature
        df['HOLIDAY_FLAG'] = 0
        for start, end in HOLIDAY_DATE_DATA.values():
            for year in df['YEAR'].unique():
                start_date = pd.to_datetime(start, format='%m-%d').replace(year=year)
                end_date = pd.to_datetime(end, format='%m-%d').replace(year=year)

                # Adjust for holidays that span the end of the year
                if start_date > end_date:
                    df.loc[(df['DateTime'] >= start_date) | (df['DateTime'] <= end_date.replace(year=year+1)), 'HOLIDAY_FLAG'] = 1
                else:
                    df.loc[(df['DateTime'] >= start_date) & (df['DateTime'] <= end_date), 'HOLIDAY_FLAG'] = 1

        return df
    
    def wind_shoretype_feature(self, X):
        time_windows = ['3h', '6h', '12h', '24h']
        
        for window in time_windows:
            wind_col = f'wind_direction_{window}'
            shore_col = f'wind_shore_{window}'
            
            if wind_col not in X.columns:
                continue
                
            # Calculate OPPOSITE beach direction (where wind would be coming FROM land)
            beach_from_direction = (X['beach_orientation_angle'] + 180) % 360
            
            # Angular difference between wind direction and OPPOSITE beach direction
            angular_diff = np.abs(X[wind_col] - beach_from_direction)
            angular_diff = np.minimum(angular_diff, 360 - angular_diff)
            
            conditions = [
                angular_diff < 45,    # Onshore: Wind from sea
                angular_diff > 135,   # Offshore: Wind from land
                (angular_diff >= 45) & (angular_diff <= 135)  # Alongshore
            ]
            
            X[shore_col] = np.select(conditions, ['Onshore', 'Offshore', 'Alongshore'], default='Unknown')
            X[shore_col] = X[shore_col].astype('category')

        return X
    
    def lagged_enterococci_features(self, data):
        """
        Adds Site_Season_Average (rolling mean of last 5, shifted) and 
        Site_Historical_Exceedance_Rate (manual proportion of previous exceedances per site-season, shifted)
        to the provided DataFrame. Matches inference logic.

        Parameters:
        data (pd.DataFrame): Must contain columns ['SITE_NAME', 'DateTime', 'Season', 'Enterococci']

        Returns:
        pd.DataFrame: Updated with lagged features.
        """

        data = data.copy()
        data["DateTime"] = pd.to_datetime(data["DateTime"])
        data["Season"] = data["Season"].astype("category")
        data["SITE_NAME"] = data["SITE_NAME"].astype("category")
        data = data.sort_values(by=["SITE_NAME", "Season", "DateTime"])

        # Prepare new columns
        data["Site_Season_Average"] = np.nan
        data["Site_Historical_Exceedance_Rate"] = np.nan

        # Manual rolling/expanding for each group (site, season)
        for (site, season), group in data.groupby(["SITE_NAME", "Season"]):
            idxs = group.index
            for i, idx in enumerate(idxs):
                prior = group.loc[idxs[:i], 'Enterococci']
                if len(prior) > 0:
                    # Rolling mean of previous 5 samples (shifted)
                    data.at[idx, "Site_Season_Average"] = prior[-5:].mean()
                    # Manual exceedance rate
                    n_prev = len(prior)
                    n_exceed = (prior >= 280).sum()
                    data.at[idx, "Site_Historical_Exceedance_Rate"] = n_exceed / n_prev
                # else: remain NaN

        # Restore original order if needed
        data = data.sort_values(by=["SITE_NAME", "DateTime"])

        # (Optional) Remove columns if not needed
        if "YEAR" in data.columns:
            data.drop(columns=["YEAR"], inplace=True)
        if "Season" in data.columns:
            data.drop(columns=["Season"], inplace=True)

        return data


    # def lagged_enterococci_features(self, data):
    #     """
    #     Adds Site Season Average (rolling mean of Enterococci) and 
    #     Site Historical Exceedance Rate (proportion of exceedances per site-season) to X_train and X_test.
        
    #     Parameters:
    #     X_train (pd.DataFrame): Training feature set
    #     X_test (pd.DataFrame): Test feature set
    #     y_train (pd.Series): Training target variable
    #     y_test (pd.Series): Test target variable
        
    #     Returns:
    #     pd.DataFrame, pd.DataFrame, pd.Series, pd.Series: Updated training and test feature sets with new features
    #     """
        
    #     # Ensure correct data types
    #     data["DateTime"] = pd.to_datetime(data["DateTime"])
    #     data["Season"] = data["Season"].astype("category")
    #     data["SITE_NAME"] = data["SITE_NAME"].astype("category")
        
    #     data = data.sort_values(by=["SITE_NAME", "DateTime"])
        
    #     # 1. Site Season Average - Rolling mean using only past values
    #     data["Site_Season_Average"] = data.groupby(["SITE_NAME", "Season"])['Enterococci']\
    #         .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
        
    #     # 2. Site Historical Exceedance Rate - Proportion of past exceedances per season/site
    #     def exceedance_rate_calc(x):
    #         past_values = x.shift(1)
    #         return (past_values >= 280).expanding().mean()
    #         # return (past_values >= 280).rolling(window=len(past_values), min_periods=1).mean()
        
    #     data["Site_Historical_Exceedance_Rate"] = data.groupby(["SITE_NAME", "Season"])['Enterococci'].transform(exceedance_rate_calc)
        
    #     # Restore original order using SITE_NAME and DateTime
    #     data = data.sort_values(by=["SITE_NAME", "DateTime"])
        
    #     # Reorder according to the original order in X_train and X_test
    #     data = data[["SITE_NAME", "DateTime"]].merge(data, on=["SITE_NAME", "DateTime"], how="left")

    #     data.drop(columns=["Season", "YEAR"], inplace=True)
        
    #     return data
    
    # def lagged_enterococci_features(self, data):
    #     """
    #     Adds only Site Historical Exceedance Rate (proportion of exceedances per site-season)
    #     to the dataframe. No Site Season Average.

    #     Parameters
    #     ----------
    #     data : pd.DataFrame
    #         Input data with columns ['SITE_NAME', 'Season', 'DateTime', 'Enterococci']

    #     Returns
    #     -------
    #     pd.DataFrame
    #         Updated data with only the Site_Historical_Exceedance_Rate feature
    #     """
    #     import pandas as pd

    #     # Ensure correct data types
    #     data["DateTime"] = pd.to_datetime(data["DateTime"])
    #     data["Season"] = data["Season"].astype("category")
    #     data["SITE_NAME"] = data["SITE_NAME"].astype("category")

    #     data = data.sort_values(by=["SITE_NAME", "DateTime"])

    #     # Only: Site Historical Exceedance Rate (rolling rate using only past values)
    #     def exceedance_rate_calc(x):
    #         past_values = x.shift(1)
    #         return (past_values >= 280).rolling(window=len(past_values), min_periods=1).mean()

    #     data["Site_Historical_Exceedance_Rate"] = (
    #         data.groupby(["SITE_NAME", "Season"])['Enterococci']
    #         .transform(exceedance_rate_calc)
    #     )

    #     # (Optional) Drop any unneeded columns if you want
    #     data.drop(columns=["Season", "YEAR"], inplace=True, errors="ignore")

    #     return data
