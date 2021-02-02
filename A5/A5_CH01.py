"""
02.02.2021

Christian Hilscher
UniMA Matr. #: 1570550
UniHD #: 6000506

Final version of solving Assignment 5 which is to be handed in
First the individual functions are defined.

"""


from pathlib import Path
import numpy as np
import pandas as pd

###############################################################################
# Specifying paths
def make_paths():
    dir = Path.cwd()
    current_folder = dir / "A5"

    output = current_folder / "output"
    output.mkdir(parents=True, exist_ok=True)
    
    return output

###############################################################################

def adapt_merge_frames(dataf_unemp, dataf_forecast):
    dataf_unemp = dataf_unemp.copy()
    dataf_forecast = dataf_forecast.copy()
    
    # Adapt UE DataFrame
    dataf_unemp["DATE"] = pd.PeriodIndex(dataf_unemp["DATE"], freq='Q')
    # Date automatically set as index with groupby
    dataf_unemp = dataf_unemp.groupby("DATE").mean()

    # Adapt Forecast DataFrame
    dataf_forecast["DATE"] = dataf_forecast["YEAR"].astype("str") \
                            + "Q" \
                            + dataf_forecast["QUARTER"].astype("str")

    dataf_forecast["DATE"] = pd.PeriodIndex(dataf_forecast["DATE"], freq='Q')
    dataf_forecast.set_index("DATE", inplace=True)

    # Merging
    dataf_out = dataf_forecast.merge(dataf_unemp, 
                                        how="inner",
                                        on="DATE")
    
    return dataf_out
###############################################################################


df_unemp = pd.read_csv("UNRATE.csv")
df_forecast = pd.read_csv("Mean_UNEMP_Level.csv")

df = adapt_merge_frames(df_unemp, df_forecast)

df