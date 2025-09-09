# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

#%% The purpose of this script is to perform Data Feature Engineering and Data Exploration before 
# Conducting a Discrete Event Simulation (DES) for Hurricane Idalia

# Intalling Packages
! pip install statsmodels
! pip install tabulate
#%%
# Importing Pacakges
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
from itertools import product
from tabulate import tabulate

pd.options.display.max_rows = 9999

#%%
# Reading in Mobile_Logistic_Nodes,Planned Network Path Locations
# And Logistic Nodes CSV Files
mln_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Updated_CSV/Mobile_Logistic_Nodes/Mobile_Logistic_Nodes.csv')
pnpl_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Updated_CSV/Planned Network Path Locations/Planned Network Path Locations.csv')
ln_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Updated_CSV/Logistic Nodes/Logistic Nodes.csv')
lnh_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Updated_CSV/Logistic Nodes/Logistic Nodes Hospitals.csv')
lns_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Updated_CSV/Logistic Nodes/Logistic Nodes Shelters.csv')
em_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Updated_CSV/Evacuation Mean And Nodes/Evacuation_Mean_(mobile_speed_capacity).csv')
en_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Updated_CSV/Evacuation Mean And Nodes/Evacuation_Nodes(evacuation_zones_by_county).csv')
emn_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Updated_CSV/Evacuation Mean And Nodes/Evacuation_Mean_(numbers).csv')

#%%
# Adding an id column in mln_df, emn_df, and pnpl_df to merge mln_df to emn_df 
# And to merge pnpl_df to emn_df
mln_df['id'] = 1
pnpl_df['id'] = 1
emn_df['id'] = 1

# Conducting inner joins on the id column, dropping duplicates, dropping the id column, and renaming columns
mst_df = pd.merge(mln_df,emn_df, how='inner', on='id').drop_duplicates().drop(columns = ['id'])\
    .rename(columns={'Throughput (Distance)' : 'Throughput'})
pnst_df = pd.merge(pnpl_df,emn_df, how='inner', on='id').drop_duplicates().drop(columns = ['id'])\
    .rename(columns={'Throughput (Distance)' : 'Throughput'})
    
# Dropping unnecessary columns from pnst_df
pnst_df = pnst_df.drop(columns=['fb_Longitude', 'fb_Latitude','Unnamed: 0'])

# Dropping rows with NA values
#pnst_df = pnst_df.dropna()

#%%
##############################################Data Feature Engineering######################################################
# To conduct Data Feature Engineering Interaction Feature Plots were generated for mln_df(Mobile Logistics)
# pnpl_df(Planned Network Path Locations) and em_df(Capabilities', 'Capacity', 'Speed', 'Throughput (Distance)

# Identifying the interaction among Mobile Logistics, Speed and Throughput
mst1_df = sm.OLS.from_formula('Throughput ~ FACILITY_TYPE * Speed', data=mst_df).fit()

# Printing model summaries
print(mst1_df.summary())

# Creating a DataFrame of all combinations for predictions
mobileln = mst_df['FACILITY_TYPE'].unique()
speeds = np.sort(mst_df['Speed'].unique())
pred_grid = pd.DataFrame(list(product(mobileln, speeds)), columns=['FACILITY_TYPE', 'Speed'])

# Predicting throughput and getting confidence intervals
predictions = mst1_df.get_prediction(pred_grid)
pred_summary = predictions.summary_frame(alpha=0.05)  # 95% CI
pred_grid['Predicted_Throughput'] = pred_summary['mean']
pred_grid['CI_lower'] = pred_summary['mean_ci_lower']
pred_grid['CI_upper'] = pred_summary['mean_ci_upper']

# Extracting interaction p-value
interaction_terms = mst1_df.pvalues.filter(like=':')  # all interaction terms
interaction_significance = "Significant" if (interaction_terms < 0.05).any() else "Not significant"

# Determine Moble Logistic Nodes with largest slope effect
slopes = {}
for ft in mobileln:
    subset = pred_grid[pred_grid['FACILITY_TYPE'] == ft]
    slope = np.polyfit(subset['Speed'], subset['Predicted_Throughput'], 1)[0]
    slopes[ft] = slope
strongest_ft = max(slopes, key=slopes.get)

# Plottig raw data and predicted lines with confidence intervals
plt.figure(figsize=(10,6))

# Raw data points
sns.scatterplot(
    data=mst_df,
    x='Speed',
    y='Throughput',
    hue='FACILITY_TYPE',
    style='FACILITY_TYPE',
    alpha=0.6,
    s=70
)

# Predicted lines with shaded confidence intervals
for ft in mobileln:
    subset = pred_grid[pred_grid['FACILITY_TYPE'] == ft]
    plt.plot(subset['Speed'], subset['Predicted_Throughput'], marker='o', label=f'{ft} (predicted)')
    plt.fill_between(
        subset['Speed'],
        subset['CI_lower'],
        subset['CI_upper'],
        alpha=0.2
    )

# Adding textual summary
plt.text(
    x=min(speeds),
    y=max(pred_grid['Predicted_Throughput']) * 0.95,
    s=f'Interaction: {interaction_significance}\nMobile Logistic Nodes with strongest Speed effect: {strongest_ft}',
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.6)
)

plt.title('Interaction of Mobile Logistic Nodes and Speed on Throughput\nRaw Data, Model Predictions & 95% CI')
plt.xlabel('Speed')
plt.ylabel('Throughput')
plt.legend(title='Mobile Logistic Nodes')
plt.tight_layout()
plt.show()

""""""

# Identifying the interaction among Mobile Logistics, Speed and Capabilities
mst2_df = sm.OLS.from_formula('Capabilities~ FACILITY_TYPE * Speed', data = mst_df).fit()

# Printing model summaries
print(mst2_df.summary())

# Creating a DataFrame of all combinations for predictions
mobileln = mst_df['FACILITY_TYPE'].unique()
speeds = np.sort(mst_df['Speed'].unique())
pred_grid = pd.DataFrame(list(product(mobileln, speeds)), columns=['FACILITY_TYPE', 'Speed'])

# Predicting Capabilities and getting confidence intervals
predictions = mst2_df.get_prediction(pred_grid)
pred_summary = predictions.summary_frame(alpha=0.05)  # 95% CI
pred_grid['Predicted_Capabilities'] = pred_summary['mean']
pred_grid['CI_lower'] = pred_summary['mean_ci_lower']
pred_grid['CI_upper'] = pred_summary['mean_ci_upper']

# Extracting interaction p-value
interaction_terms = mst2_df.pvalues.filter(like=':')  # all interaction terms
interaction_significance = "Significant" if (interaction_terms < 0.05).any() else "Not significant"

# Determine Moble Logistic Nodes with largest slope effect
slopes = {}
for ft in mobileln:
    subset = pred_grid[pred_grid['FACILITY_TYPE'] == ft]
    slope = np.polyfit(subset['Speed'], subset['Predicted_Capabilities'], 1)[0]
    slopes[ft] = slope
strongest_ft = max(slopes, key=slopes.get)

# Plottig raw data and predicted lines with confidence intervals
plt.figure(figsize=(10,6))

# Raw data points
sns.scatterplot(
    data=mst_df,
    x='Speed',
    y='Capabilities',
    hue='FACILITY_TYPE',
    style='FACILITY_TYPE',
    alpha=0.6,
    s=70
)

# Predicted lines with shaded confidence intervals
for ft in mobileln:
    subset = pred_grid[pred_grid['FACILITY_TYPE'] == ft]
    plt.plot(subset['Speed'], subset['Predicted_Capabilities'], marker='o', label=f'{ft} (predicted)')
    plt.fill_between(
        subset['Speed'],
        subset['CI_lower'],
        subset['CI_upper'],
        alpha=0.2
    )

# Adding textual summary
plt.text(
    x=min(speeds),
    y=max(pred_grid['Predicted_Capabilities']) * 0.95,
    s=f'Interaction: {interaction_significance}\nMobile Logistic Nodes with strongest Speed effect: {strongest_ft}',
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.6)
)

plt.title('Interaction of Mobile Logistic Nodes and Speed on Capabilities\nRaw Data, Model Predictions & 95% CI')
plt.xlabel('Speed')
plt.ylabel('Capabilities')
plt.legend(title='Mobile Logistic Nodes')
plt.tight_layout()
plt.show()

""""""

# Identifying the interaction among Mobile Logistics, Capacity and Capabilities
mst3_df = sm.OLS.from_formula('Capacity ~ FACILITY_TYPE * Capabilities', data = mst_df).fit()

# Printing model summaries
print(mst3_df.summary())

# Creating a DataFrame of all combinations for predictions
mobileln = mst_df['FACILITY_TYPE'].unique()
Capabilities = np.sort(mst_df['Capabilities'].unique())
pred_grid = pd.DataFrame(list(product(mobileln, Capabilities)), columns=['FACILITY_TYPE', 'Capabilities'])

# Predicting Capacity and getting confidence intervals
predictions = mst3_df.get_prediction(pred_grid)
pred_summary = predictions.summary_frame(alpha=0.05)  # 95% CI
pred_grid['Predicted_Capacity'] = pred_summary['mean']
pred_grid['CI_lower'] = pred_summary['mean_ci_lower']
pred_grid['CI_upper'] = pred_summary['mean_ci_upper']

# Extracting interaction p-value
interaction_terms = mst3_df.pvalues.filter(like=':')  # all interaction terms
interaction_significance = "Significant" if (interaction_terms < 0.05).any() else "Not significant"

# Determine Moble Logistic Nodes with largest slope effect
slopes = {}
for ft in mobileln:
    subset = pred_grid[pred_grid['FACILITY_TYPE'] == ft]
    slope = np.polyfit(subset['Capabilities'], subset['Predicted_Capacity'], 1)[0]
    slopes[ft] = slope
strongest_ft = max(slopes, key=slopes.get)

# Plottig raw data and predicted lines with confidence intervals
plt.figure(figsize=(10,6))

# Raw data points
sns.scatterplot(
    data=mst_df,
    x='Capabilities',
    y='Capacity',
    hue='FACILITY_TYPE',
    style='FACILITY_TYPE',
    alpha=0.6,
    s=70
)

# Predicted lines with shaded confidence intervals
for ft in mobileln:
    subset = pred_grid[pred_grid['FACILITY_TYPE'] == ft]
    plt.plot(subset['Capabilities'], subset['Predicted_Capacity'], marker='o', label=f'{ft} (predicted)')
    plt.fill_between(
        subset['Capabilities'],
        subset['CI_lower'],
        subset['CI_upper'],
        alpha=0.2
    )

# Adding textual summary
plt.text(
    x=min(Capabilities),
    y=max(pred_grid['Predicted_Capacity']) * 0.95,
    s=f'Interaction: {interaction_significance}\nMobile Logistic Nodes with strongest Capabilities effect: {strongest_ft}',
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.6)
)

plt.title('Interaction of Mobile Logistic Nodes and Capabilities on Capacity\nRaw Data, Model Predictions & 95% CI')
plt.xlabel('Capabilities')
plt.ylabel('Capacity')
plt.legend(title='Mobile Logistic Nodes')
plt.tight_layout()
plt.show()

""""""

# Identifying the interaction among Mobile Logistics, Capacity and Throughput
mst4_df = sm.OLS.from_formula('Throughput ~ FACILITY_TYPE * Capacity', data = mst_df).fit()

# Printing model summaries
print(mst4_df.summary())

# Creating a DataFrame of all combinations for predictions
mobileln = mst_df['FACILITY_TYPE'].unique()
Capacity = np.sort(mst_df['Capacity'].unique())
pred_grid = pd.DataFrame(list(product(mobileln, Capacity)), columns=['FACILITY_TYPE', 'Capacity'])

# Predicting throughput and getting confidence intervals
predictions = mst4_df.get_prediction(pred_grid)
pred_summary = predictions.summary_frame(alpha=0.05)  # 95% CI
pred_grid['Predicted_Throughput'] = pred_summary['mean']
pred_grid['CI_lower'] = pred_summary['mean_ci_lower']
pred_grid['CI_upper'] = pred_summary['mean_ci_upper']

# Extracting interaction p-value
interaction_terms = mst3_df.pvalues.filter(like=':')  # all interaction terms
interaction_significance = "Significant" if (interaction_terms < 0.05).any() else "Not significant"

# Determine Moble Logistic Nodes with largest slope effect
slopes = {}
for ft in mobileln:
    subset = pred_grid[pred_grid['FACILITY_TYPE'] == ft]
    slope = np.polyfit(subset['Capacity'], subset['Predicted_Throughput'], 1)[0]
    slopes[ft] = slope
strongest_ft = max(slopes, key=slopes.get)

# Plottig raw data and predicted lines with confidence intervals
plt.figure(figsize=(10,6))

# Raw data points
sns.scatterplot(
    data=mst_df,
    x='Capacity',
    y='Throughput',
    hue='FACILITY_TYPE',
    style='FACILITY_TYPE',
    alpha=0.6,
    s=70
)

# Predicted lines with shaded confidence intervals
for ft in mobileln:
    subset = pred_grid[pred_grid['FACILITY_TYPE'] == ft]
    plt.plot(subset['Capacity'], subset['Predicted_Throughput'], marker='o', label=f'{ft} (predicted)')
    plt.fill_between(
        subset['Capacity'],
        subset['CI_lower'],
        subset['CI_upper'],
        alpha=0.2
    )

# Adding textual summary
plt.text(
    x=min(Capacity),
    y=max(pred_grid['Predicted_Throughput']) * 0.95,
    s=f'Interaction: {interaction_significance}\nMobile Logistic Nodes with strongest Throughput effect: {strongest_ft}',
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.6)
)

plt.title('Interaction of Mobile Logistic Nodes and Capacity on Throughput\nRaw Data, Model Predictions & 95% CI')
plt.xlabel('Capacity')
plt.ylabel('Throughput')
plt.legend(title='Mobile Logistic Nodes')
plt.tight_layout()
plt.show()


#%%
# Identifying the interaction among Planned Network Path Locations, Speed, and Throughput
pnst1_df = sm.OLS.from_formula('Throughput ~ FACILITY_TYPE * Speed', data = pnst_df).fit()

# Creating the interaction plots
fig5, ax = plt.subplots()
sm.graphics.interaction_plot(pnst_df['FACILITY_TYPE'], pnst_df['Speed'], pnst_df['Throughput'], ax=ax)
plt.show()

# Identifying the interaction among Planned Network Path Locations, Speed, and Capabilities
pnst2_df = sm.OLS.from_formula('Capabilities~ FACILITY_TYPE * Speed', data = pnst_df).fit()

# Creating the interaction plots
fig6, ax = plt.subplots()
sm.graphics.interaction_plot(pnst_df['FACILITY_TYPE'], pnst_df['Speed'], pnst_df['Capabilities'], ax=ax)
plt.show()

# Identifying the interaction among Planned Network Path Locations, Capacity, and Capabilities
pnst3_df = sm.OLS.from_formula('Capacity ~ FACILITY_TYPE * Capabilities', data = pnst_df).fit()

# Creating the interaction plots
fig7, ax = plt.subplots()
sm.graphics.interaction_plot(pnst_df['FACILITY_TYPE'], pnst_df['Capabilities'], pnst_df['Capacity'], ax=ax)
plt.show()

# Identifying the interaction among Planned Network Path Locations, Capacity, and Throughput
pnst8_df = sm.OLS.from_formula('Throughput ~ FACILITY_TYPE * Capacity', data = pnst_df).fit()

# Creating the interaction plots
fig8, ax = plt.subplots()
sm.graphics.interaction_plot(pnst_df['FACILITY_TYPE'], pnst_df['Capacity'], pnst_df['Throughput'], ax=ax)
plt.show()

#%%
# Creating interaction features manually for speed and throughput for both data frames (mst_df and pnst_df)
mst_df['interaction'] = mst_df['Speed'] * mst_df['Throughput']
pnst_df['interaction'] = pnst_df['Speed'] * pnst_df['Throughput']

#%%
##################################################Data Exploration#########################################################
# Generating five number summaries for Mobile Logistic Nodes, Planned Network Path Locations,
# Evacuation Mean(mobile speed and capacity), Evacuation Nodes(evacuation zones by county), 
# And Logistic Nodes and formatting the results into a table, Boxplots, and dropping unnecessary columns.

# Dropping Unessary Columns
mln_df = mln_df.drop(columns=['Unnamed: 0'])

# Generating Five-Number Summaries
mln_summary = mln_df.describe()

# Convert the description to a list of lists for tabulate
mln_table = mln_summary.reset_index().values.tolist()

# Print the table using tabulate
print(tabulate(mln_table, headers='firstrow', tablefmt='grid'))

# Creating a box plot
mln_df.boxplot(figsize=(10, 6))

# Setting title and labels
plt.title('Box Plot of Moble Logistic Nodes')
plt.xlabel('Columns')
plt.ylabel('Values')

# Show the plot
plt.show()

""""""""""""""""""

# Dropping Unessary Columns
pnpl_df = pnpl_df.drop(columns=['Unnamed: 0'])

# Generating Five-Number Summaries
pnpl_summary = pnpl_df.describe()

# Convert the description to a list of lists for tabulate
pnpl_table = pnpl_summary.reset_index().values.tolist()

# Print the table using tabulate
print(tabulate(pnpl_table, headers='firstrow', tablefmt='grid'))

# Creating a box plot
pnpl_df.boxplot(figsize=(10, 6))

# Setting title and labels
plt.title('Box Plot of Planned Network Path Locations')
plt.xlabel('Columns')
plt.ylabel('Values')

# Show the plot
plt.show()

""""""""""""""""""

# Generating Five-Number Summaries
em_summary = em_df.describe()

# Convert the description to a list of lists for tabulate
em_table = em_summary.reset_index().values.tolist()

# Print the table using tabulate
print(tabulate(em_table, headers='firstrow', tablefmt='grid'))

# Creating a box plot
em_df.boxplot(figsize=(10, 6))

# Setting title and labels
plt.title('Box Plot of Evacuation Mean')
plt.xlabel('Columns')
plt.ylabel('Values')

# Show the plot
plt.show()

""""""""""""""""""

# Generating Five-Number Summaries
en_summary = en_df.describe()

# Convert the description to a list of lists for tabulate
en_table = en_summary.reset_index().values.tolist()

# Print the table using tabulate
print(tabulate(en_table, headers='firstrow', tablefmt='grid'))

# Creating a box plot
en_df.boxplot(figsize=(10, 6))

# Setting title and labels
plt.title('Box Plot of Evacuation Mean Numbers')
plt.xlabel('Columns')
plt.ylabel('Values')

# Show the plot
plt.show()

""""""""""""""""""
# Dropping Unessary Columns
ln_df = ln_df.drop(columns=['Unnamed: 0'])

# Generating Five-Number Summaries
ln_summary = ln_df.describe()

# Convert the description to a list of lists for tabulate
ln_table = ln_summary.reset_index().values.tolist()

# Print the table using tabulate
print(tabulate(ln_table, headers='firstrow', tablefmt='grid'))

# Creating a box plot
ln_df.boxplot(figsize=(10, 6))

# Setting title and labels
plt.title('Box Plot of Logistic Nodes')
plt.xlabel('Columns')
plt.ylabel('Values')

# Show the plot
plt.show()

""""""""""""""""""

# Dropping Unessary Columns
lnh_df = lnh_df.drop(columns=['Unnamed: 0'])

# Generating Five-Number Summaries
lnh_summary = lnh_df.describe()

# Convert the description to a list of lists for tabulate
lnh_table = lnh_summary.reset_index().values.tolist()

# Print the table using tabulate
print(tabulate(lnh_table, headers='firstrow', tablefmt='grid'))

# Creating a box plot
lnh_df.boxplot(figsize=(10, 6))

# Setting title and labels
plt.title('Box Plot of Logistic Nodes Hospitals')
plt.xlabel('Columns')
plt.ylabel('Values')

# Show the plot
plt.show()

""""""""""""""""""

# Dropping Unessary Columns
lns_df = lns_df.drop(columns=['Unnamed: 0'])

# Generating Five-Number Summaries
lns_summary = lns_df.describe()

# Convert the description to a list of lists for tabulate
lns_table = lns_summary.reset_index().values.tolist()

# Print the table using tabulate
print(tabulate(lns_table, headers='firstrow', tablefmt='grid'))

# Creating a box plot
lns_df.boxplot(figsize=(10, 6))

# Setting title and labels
plt.title('Box Plot of Logistic Nodes Shelters')
plt.xlabel('Columns')
plt.ylabel('Values')

# Show the plot
plt.show()

#%%
