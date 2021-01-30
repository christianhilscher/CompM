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
forecast_quater['QUATER'] = forecast_quater['QUATER'].astype(str)

# Combine Year and Quater to a Date collumn
forecast_quater['Date'] = forecast_quater['YEAR']+ 'Q' + forecast_quater['QUARTER']

# Drop now unused columns
forecast_quater.drop(['Year' ,'Quater'])

# Set Date as Index
forecast_quater['DATE'] = pd.PeriodIndex(forecast_quater['DATE'], freq='Q')

# 1.5 Merge both Datasets
A5_data = pd.concat([uer_quater, forecast_quater], axis=0)

# 1.6 Create new Variables
# Create X_{t+3} - F(x_{t+3})_t = forecast error
A5_data['fc_error'] = A5_data['UE'] - A5_data['UN-EMP5']

# Creat F(x_{t+3})_t - F(x_{t+3})_{t-1}
# Create a lag for the forcast
A5_data['UN_EMP5_lag'] = A5_data['UN_EMP5'].shift(+1)

# Use the lag to create the revision
A5_data['fc_revision'] = A5_data['UN_EMP5'] - A5_data['UN_EMP5_lag']

# 1.7 OLS regression
# Define OLS function
reg = ols(formula = 'A5_data[fc_error] ~ A5_data[fc_revision]', data = A5_data)

# Fit the OLS with robust SE
fit = reg.fit(cov_type = 'HAC', cov_kwds = {'maxlags' : 3})

# Print summary statistics
print(fit.summary())

# 1.8 Interpretation

'''

Coibion and Gorodnychenko (2015) find a sig. and positive relationship  with inflation, compare and interprete this one:

'''



























