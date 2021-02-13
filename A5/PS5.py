import os
import pandas as pd 
from statsmodels.formula.api import ols

#*set path
wd_lc = '/Users/llccf/OneDrive/Dokumente/3. Semester/Comp Macro (HD)/A5'
os.chdir(wd_lc)

#*##############
#! FUNCTIONS 
#*##############
def reduce_freq(df, freq = 'Q', datecol = 'DATE'):
    '''
    Turn monthly into lower frequency (e.g. quarterly) dataframe and get averages.
    
    *df = monthly dataset to be transformed
    *freq = freq to be transformed into 
    *datecol = name of column that contains date variable
    '''
    df = df.set_index(datecol)
    df.index = pd.PeriodIndex(df.index, freq = freq)
    df = df.groupby(df.index).mean()
    
    return df

def regress(formula, df, cov = 'H1'):
    '''
    Run a simple OLS regression.
    
    *formula = regression formula
    *df = dataframe containing data 
    *cov = covariance type used in estimation, default is heteroskedastic SEs; note: if cov = 'HAC', cov_kwds defaults to {'maxlags': 3}
    
    '''
    if cov == 'HAC':
        cov_kwds = {'maxlags': 3}
    else:
        cov_kwds = None 
    mod = ols(formula, df)
    results = mod.fit(cov_type = cov, cov_kwds = cov_kwds)
    print(results.summary())    
    
    return results

#*##############
#! DATA
#*##############
#read in unemployment data 
actual = pd.read_csv('data/FRED_unemprate.csv')
forecasts = pd.read_csv('data/mean_forecasts_unemp.csv', sep = ';')

#* 3. 
#turn actual UR to quarterly averages 
actual_q = reduce_freq(actual)

#* 4. 
#create date column in forecast data 
forecasts['DATE'] = forecasts['YEAR'].astype(str) + 'Q' + forecasts['QUARTER'].astype(str)
forecasts = forecasts.set_index('DATE')
forecasts.index = pd.PeriodIndex(forecasts.index, freq = 'Q')
forecasts = forecasts.drop(['YEAR', 'QUARTER'], axis = 1)

#* 5. 
#merge the two dfs into one
complete = actual_q.merge(forecasts, on = 'DATE')

#* 6.
#construct new variables 
#the unemployment columns are strings, so first need to turn them into floats
'''
first, let's start with the dependent variable; 'shift' shifts the index x periods, so here shift 3 periods back such that the value of UNRATE 3 periods ahead is at the same index as the forecast for 3 periods ahead (namely, point t)
'''
complete['dependent'] = complete['UNRATE'].shift(-3) - complete['UNEMP5'] 
'''
Now the independent variable: the forecast of x_(t+3) in period t-1 is 4 (!) quarters ahead, not 3 (as a 3 quarters ahead forecast in t-1 is about t+2). Thus, use UNEMP6 and shift it by 1 period.
We lose 1 period here obviously (the first one).
'''
complete['independent'] = complete['UNEMP5'] - complete['UNEMP6'].shift(1)

#* 7.
# Estimation
form = 'dependent ~ independent'
results = regress(form, complete, cov = 'HAC')

'''
Similarly as in Coibion and Gorodnynchenko (2015) (henceforth CG) we regress the aggregate forecast error of 3 quarters ahead forecasts on the forecast revisions of these forecasts within one period. The OLS coefficient is (assuming that it is unbiased) equal to the correlation between these two measures. Our results show that as in CG the coefficient is indeed significant at the 5% level. This suggests that there is a relationship between these two variables that under rational expectations should not exist. Under rational expecations we would assume that the updating process of the forecast from t-1 to t itself does not affect by how much the forecast errs. 
Intuitively, the coefficient should show by how much the forecast error is changed when the forecast is updated by one unit and should therefore show some kind of measure for the deviation from the rationally expected updating behavior. 
'''