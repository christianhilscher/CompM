'''
Assignment 5 - Computational Macro
Florian Fickler -- 1545713
'''
# Import packages
import pandas as pd
from statsmodels.formula.api import ols

# 1.2 Read in the data
# Forecast Data
forecast = pd.read_csv(r'C:\Users\Flori\OneDrive\Documents\GitHub\CompM\A5\\Mean_UNEMP_Level_csv-export.csv', sep=';', decimal= ',')

# UE-Rate
ue_rate = pd.read_csv(r'C:\Users\Flori\OneDrive\Documents\GitHub\CompM\A5\\FRED_unemprate.csv' , sep=',', decimal='.')

# 1.3 Transfer UE-Data to Quaterly
# Copy dataframe
uer_quater = ue_rate.copy()

# Transfer the Date to quaterly
uer_quater['DATE'] = pd.PeriodIndex(uer_quater['DATE'], freq='Q')

# Set Date as Index
uer_quater=uer_quater.set_index(['DATE'])

# Average Up the monthly Data to Quaterly Averages
uer_quater = ue_rate.groupby(uer_quater.index).mean()

# 1.4 Transfer Forecast Data 
# Copy data frame
forecast_quater = forecast.copy()

# Convert Year and Quater to string
forecast_quater['YEAR'] = forecast_quater['YEAR'].astype(str)
forecast_quater['QUARTER'] = forecast_quater['QUARTER'].astype(str)

# Combine Year and Quater to a Date collumn
forecast_quater['DATE'] = forecast_quater['YEAR']+ 'Q' + forecast_quater['QUARTER']

# Transfer the Date to quaterly
forecast_quater['DATE'] = pd.PeriodIndex(forecast_quater['DATE'], freq='Q')

# Set Date as Index
forecast_quater=forecast_quater.set_index(['DATE'])

# Drop now unused columns
forecast_quater = forecast_quater.drop(['YEAR' ,'QUARTER'], axis = 1)

# 1.5 Merge both Datasets
A5_data = pd.concat([uer_quater, forecast_quater], axis=1)

# 1.6 Create new Variables
# Create X_{t+3} - F(x_{t+3})_t = forecast error
# Shift UE rate
A5_data['UNRATE_lead'] = A5_data['UNRATE'].shift(-3)

A5_data['fc_error'] = A5_data['UNRATE_lead'] - A5_data['UNEMP5']

# Creat F(x_{t+3})_t - F(x_{t+3})_{t-1}
# Create a lag for the forcast
A5_data['UN_EMP5_lag'] = A5_data['UNEMP6'].shift(+1)

# Use the lag to create the revision
A5_data['fc_revision'] = A5_data['UNEMP5'] - A5_data['UN_EMP5_lag']

# 1.7 OLS regression
# Define OLS function
reg = ols(formula = 'fc_error ~ fc_revision', data = A5_data)

# Fit the OLS with robust SE
fit = reg.fit(cov_type = 'HAC', cov_kwds = {'maxlags' : 3})

# Print summary statistics
print(fit.summary())

# 1.8 Interpretation

'''
In line with the paper by Coibion and Gordnychenko (2015) henceforth CG, I find a positive estimate for beta which is highly significant.
Even though the estimate is somewhat smaller, it still is larger than 0, and therefore indicates the presence of information rigidities.
In the context of sticky-information models, my estimate would translate into an informational friction lambda of roughly 0.42.
Comparing this to the estimated information friction of 0.55 from CG, one could see an evidence in this, that there are less informational friction present, when it comes to the forecasting of unemployment rates compared to inflation rates.
Furthermore, finding a positive estimate for this data, can also be seen as additional validation for the microfoundations used in CG.
Taking all of this together, I would argue, that my restults are in line with primary findings of CG.

A simple way of interpreting beta, would be to caracterise it as the ammount of information rigidities that are present when it comes to forecasting macroeconomic variables. In other words, it is an indicator for the failing of the assumptions, that agents have full information or a rational updating behavior of their expectations when they receive new information.

The latter one can be explained with the use of sticky information model (e.g. Mankiw and Reis (2002)), where as written above, the beta can be mapped into a degree of infomation rigidity lambda. Relatively simple, lambda stands for the probabilty of aquring new information. An increase in beta, then simply reduces the probabilty, that an agent will acquire new information in a given period, and makes it more likely, that he will miss out on updating his expectations.

In imperfect information models (e.g. Woodford (2001)) however, we can map beta into a different meassure of information rigidities. Assuming that agents update, but rather lack some information, beta can then be seen as a meassure of how much of this information is missing. Therefore a larger beta would translate in more imperfect information and in less informed agents.

'''



























