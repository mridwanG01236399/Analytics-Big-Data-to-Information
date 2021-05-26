# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 03:10:49 2020

@author: iwanJ
"""

import seaborn as sns
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os  # needed for directory access
from statsmodels.graphics.gofplots import ProbPlot

#read directory
os.chdir('D:/George Mason University/2. Spring 2020/Analytics Big Data to Information/Assignment 8')
os.getcwd()

#Import the data1
data1 = pd.read_csv('COVID-19-geographic-disbtribution-worldwide-2020-04-16.csv', sep = ',')
data1
data1.head(10)
data1.info()
data1['countriesAndTerritories']

#setdata1 Indonesia
data1_indonesia = data1.query('countriesAndTerritories == "Indonesia"')
len(data1_indonesia.index)
data1_indonesia2 = data1_indonesia.sort_values(by='dateRep')
data1_indonesia2.info()
data1_indonesia3 = data1_indonesia2[['dateRep', 'day', 'month', 'year', 'cases', 'deaths', 'popData2018', 'countriesAndTerritories']]
data1_indonesia3.head(50)

#test_data1_indonesia3 = data1_indonesia3.query('cases > 0')
#test_data1_indonesia3 = test_data1_indonesia3[['dateRep', 'day', 'month', 'year', 'cases', 'deaths', 'popData2018']]

#convert to array for sorting
dateRep1 = np.array(data1_indonesia3['dateRep'])
day1 = np.array(data1_indonesia3['day'])
month1 = np.array(data1_indonesia3['month'])
year1 = np.array(data1_indonesia3['year'])
cases1 = np.array(data1_indonesia3['cases'])
deaths1 = np.array(data1_indonesia3['deaths'])
popData1 = np.array(data1_indonesia3['popData2018'])
countriesAndTerritories1 = np.array(data1_indonesia3['countriesAndTerritories'])

###############################################################################
#sort based on year
###############################################################################
i = 0
while (i < (len(data1_indonesia3) - 1)):
    j = 0
    while (j < (len(data1_indonesia3)-i-1)):
        
        if (year1[j] > year1[j + 1]):
            temp1 = year1[j]
            year1[j] = year1[j + 1]
            year1[j + 1] = temp1
                    
            temp2 = day1[j]
            day1[j] = day1[j + 1]
            day1[j + 1] = temp2
                    
            temp3 = month1[j]
            month1[j] = month1[j + 1]
            month1[j + 1] = temp3
                    
            temp4 = cases1[j]
            cases1[j] = cases1[j + 1]
            cases1[j + 1] = temp4
                    
            temp5 = deaths1[j]
            deaths1[j] = deaths1[j + 1]
            deaths1[j + 1] = temp5
            
            temp6 = dateRep1[j]
            dateRep1[j] = dateRep1[j + 1]
            dateRep1[j + 1] = temp6
            
            temp7 = popData1[j]
            popData1[j] = popData1[j + 1]
            popData1[j + 1] = temp7
                  
        j = j + 1        
    i = i + 1
###############################################################################
#sort based on month
###############################################################################    
i = 0
while (i < (len(data1_indonesia3) - 1)):
    if (year1[i] == year1[i + 1]):
        print("same year")
        
        j = i;
        while (j < (len(data1_indonesia3) - 1)):
            k = j
            while (k < (len(data1_indonesia3)-i-1)):
                print('sorting process')
                if (month1[k] > month1[k + 1]):
                    temp1 = year1[k]
                    year1[k] = year1[k + 1]
                    year1[k + 1] = temp1
                            
                    temp2 = day1[k]
                    day1[k] = day1[k + 1]
                    day1[k + 1] = temp2
                            
                    temp3 = month1[k]
                    month1[k] = month1[k + 1]
                    month1[k + 1] = temp3
                            
                    temp4 = cases1[k]
                    cases1[k] = cases1[k + 1]
                    cases1[k + 1] = temp4
                            
                    temp5 = deaths1[k]
                    deaths1[k] = deaths1[k + 1]
                    deaths1[k + 1] = temp5  
                    
                    temp6 = dateRep1[k]
                    dateRep1[k] = dateRep1[k + 1]
                    dateRep1[k + 1] = temp6
                    
                    temp7 = popData1[k]
                    popData1[k] = popData1[k + 1]
                    popData1[k + 1] = temp7
                k = k + 1                
            j = j + 1
    else:
        print("different year")
    i = i + 1
###############################################################################
#sort based on day
###############################################################################        
i = 0
while (i < (len(data1_indonesia3) - 1)):
    if (year1[i] == year1[i + 1]):
        print("same year")
        
        j = i;
        while (j < (len(data1_indonesia3) - 1)):
            k = j
            while (k < (len(data1_indonesia3)-i-1)):                
                if (month1[k] == month1[k + 1]):
                    print("same month")
                    
                    l = k
                    while ((l < (len(data1_indonesia3)-i)) and (month1[l] == month1[l + 1])):
                        if (day1[l] > day1[l + 1]):
                            temp1 = year1[l]
                            year1[l] = year1[l + 1]
                            year1[l + 1] = temp1
                                    
                            temp2 = day1[l]
                            day1[l] = day1[l + 1]
                            day1[l + 1] = temp2
                                    
                            temp3 = month1[l]
                            month1[l] = month1[l + 1]
                            month1[l + 1] = temp3
                                    
                            temp4 = cases1[l]
                            cases1[l] = cases1[l + 1]
                            cases1[l + 1] = temp4
                                    
                            temp5 = deaths1[l]
                            deaths1[l] = deaths1[l + 1]
                            deaths1[l + 1] = temp5

                            temp6 = dateRep1[l]
                            dateRep1[l] = dateRep1[l + 1]
                            dateRep1[l + 1] = temp6
                            
                            temp7 = popData1[l]
                            popData1[l] = popData1[l + 1]
                            popData1[l + 1] = temp7                                
                        l = l + 1                    
                else:
                    print("different month")
                k = k + 1                
            j = j + 1
    else:
        print("different year")
    i = i + 1
###############################################################################
#New Data Frame for setdata1 Indonesia
############################################################################### 
data1_indonesia4 = pd.DataFrame({'dateRep' : dateRep1,
                                 'day' : day1, 
                                 'month' : month1,
                                 'year' : year1,
                                 'cases' : cases1,
                                 'deaths' : deaths1,
                                 'popData2018' : popData1,
                                 'countriesAndTerritories' : countriesAndTerritories1})

data1_indonesia4 = data1_indonesia4.query('cases > 0').reset_index(drop=False).reset_index(drop=False)

#data1_indonesia4 = data1_indonesia4[['level_0', 'dateRep', 'day', 'month', 'year', 'cases', 'deaths', 'popData2018']]
data1_indonesia4 = data1_indonesia4[['level_0', 'dateRep', 'day', 'month', 'year', 'cases']]

#data1_indonesia4.columns = ['Timestep', 'dateRep', 'day', 'month', 'year', 'cases', 'deaths', 'popData2018']
data1_indonesia4.columns = ['Timestep', 'dateRep', 'day', 'month', 'year', 'cases']

data1_indonesia4.head(16)
###############################################################################
#DIAGNOSTIC PLOT FOR DATA1
###############################################################################
X = data1_indonesia4.Timestep
X = sm.add_constant(X)

y1 = data1_indonesia4.cases


model1 = sm.OLS(y1, X)
model_fit1 = model1.fit()

print(model_fit1.summary())
dataframe = pd.concat([X, y1], axis=1)

###############################################################################
#Residuals vs Fitted
###############################################################################
# model values
model_fitted_y = model_fit1.fittedvalues

# model residuals
model_residuals = model_fit1.resid

# normalized residuals
model_norm_residuals = model_fit1.get_influence().resid_studentized_internal

# absolute squared normalized residuals
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

# absolute residuals
model_abs_resid = np.abs(model_residuals)

# leverage, from statsmodels internals
model_leverage = model_fit1.get_influence().hat_matrix_diag

# cook's distance, from statsmodels internals
model_cooks = model_fit1.get_influence().cooks_distance[0]

plot_lm_1 = plt.figure()
plot_lm_1.axes[0] = sns.residplot(model_fitted_y, dataframe.columns[-1], data=dataframe, lowess=True, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
plot_lm_1.axes[0].set_title('Residuals vs Fitted')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals');
###############################################################################
#Normal Q-Q Plot
###############################################################################
QQ = ProbPlot(model_norm_residuals)
plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
plot_lm_2.axes[0].set_title('Normal Q-Q')
plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
plot_lm_2.axes[0].set_ylabel('Standardized Residuals');
# annotations
abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
abs_norm_resid_top_3 = abs_norm_resid[:3]
for r, i in enumerate(abs_norm_resid_top_3):
    plot_lm_2.axes[0].annotate(i, xy=(np.flip(QQ.theoretical_quantiles, 0)[r], model_norm_residuals[i]));
###############################################################################
#Scale-Location
###############################################################################
plot_lm_3 = plt.figure()
plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5);
sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt, scatter=False, ci=False, lowess=True, line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
plot_lm_3.axes[0].set_title('Scale-Location')
plot_lm_3.axes[0].set_xlabel('Fitted values')
plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');

# annotations
abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
for i in abs_norm_resid_top_3:
    plot_lm_3.axes[0].annotate(i, xy=(model_fitted_y[i], model_norm_residuals_abs_sqrt[i]));
###############################################################################
#Residuals vs Leverage
###############################################################################
plot_lm_4 = plt.figure();
plt.scatter(model_leverage, model_norm_residuals, alpha=0.5);
sns.regplot(model_leverage, model_norm_residuals, scatter=False, ci=False, lowess=True, line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
plot_lm_4.axes[0].set_xlim(0, max(model_leverage)+0.01)
plot_lm_4.axes[0].set_ylim(-3, 5)
plot_lm_4.axes[0].set_title('Residuals vs Leverage')
plot_lm_4.axes[0].set_xlabel('Leverage')
plot_lm_4.axes[0].set_ylabel('Standardized Residuals');

# annotations
leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]
for i in leverage_top_3:
    plot_lm_4.axes[0].annotate(i, xy=(model_leverage[i], model_norm_residuals[i]));
###############################################################################
#create the plot of COVID-19 case in Indonesia
plt.scatter(data1_indonesia4.Timestep, data1_indonesia4.cases)
plt.title('Number of COVID-19 case in Indonesia')
plt.xlabel('Time (Day)')
plt.ylabel('Cases')
plt.legend([ 'Real Data', 'Exponential Model'])

#basic model function = 
#y = (-31.6437) + 9.0163X
def linear_predictions(t):
    return (-31.6437) + (9.0163 * t)

#get the prediction result
data1_indonesia4['PredictedCases'] = linear_predictions(data1_indonesia4.Timestep)
data1_indonesia4

#create the plot of real data vs predicted case of COVID-19 in Indonesia
plt.scatter(data1_indonesia4.Timestep, data1_indonesia4.cases)
plt.plot(data1_indonesia4.Timestep, linear_predictions(data1_indonesia4.Timestep), 'blue')
plt.title('Predicted number vs real number of COVID-19 growth case in Indonesia')
plt.xlabel('Time (Day)')
plt.ylabel('Cases')
plt.legend([ 'Basic Model', 'Real Data'])