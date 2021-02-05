"""
03.02.2021

Christian Hilscher
UniMA Matr. #: 1570550
UniHD #: 6000506

Final version of solving Assignment 5 which is to be handed in
First the individual functions are defined.

"""

from pathlib import Path
import pandas as pd
import statsmodels.api as sm

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

## Task 1-2
# Reading in files, saved the forecast as a csv instead of an Excel file.
df_unemp = pd.read_csv("UNRATE.csv")
df_forecast = pd.read_csv("Mean_UNEMP_Level.csv")

###############################################################################
## Task 3-5
# Merge dataframes
df = adapt_merge_frames(df_unemp, df_forecast)

###############################################################################
## Task 6
# Shifting around
df["y"] = df["UNRATE"].shift(-3) - df["UNEMP5"]
df["X"] = df["UNEMP5"] - df["UNEMP6"].shift(+1)

# Drop missings
missing = df["X"].isna() | df["y"].isna()
df = df[~missing]

###############################################################################
## Task 7
# Define arrays for estimation
y = df["y"]
X = df["X"]
X = sm.add_constant(X)

# Estimation
modl = sm.OLS(y, X)
res = modl.fit(cov_type = 'HAC', cov_kwds = {'maxlags' : 3})
print(res.summary())

###############################################################################
# Task 8

"""
As in Coibion and Gorodnychenko (2015) the coefficient of the aggregate
forecast revisions is highly significant and positive. With RE holding,
the coefficient should be not statistically different from 0. Therefore one
can conclude from this regression, that RE do not hold when it comes to the 
forecast of the unemployment rate and that information rigidities exist.
"""