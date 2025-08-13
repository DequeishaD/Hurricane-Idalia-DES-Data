# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

#%% The purpose of this script is to perform Data Feature Engineering and Data Exploration before 
# Conducting a Discrete Event Simulation (DES) for Hurricane Idalia

#%%
# Importing Pacakges
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
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
mst1_df = sm.OLS.from_formula('Throughput ~ FACILITY_TYPE * Speed', data = mst_df).fit()

# Creating the interaction plots
fig1, ax = plt.subplots()
sm.graphics.interaction_plot(mst_df['FACILITY_TYPE'], mst_df['Speed'], mst_df['Throughput'], ax=ax)
plt.show()

# Identifying the interaction among Mobile Logistics, Speed and Capabilities
mst2_df = sm.OLS.from_formula('Capabilities~ FACILITY_TYPE * Speed', data = mst_df).fit()

# Creating the interaction plots
fig2, ax = plt.subplots()
sm.graphics.interaction_plot(mst_df['FACILITY_TYPE'], mst_df['Speed'], mst_df['Capabilities'], ax=ax)
plt.show()

# Identifying the interaction among Mobile Logistics, Capacity, and Capabilities
mst3_df = sm.OLS.from_formula('Capacity ~ FACILITY_TYPE * Capabilities', data = mst_df).fit()

# Creating the interaction plots
fig3, ax = plt.subplots()
sm.graphics.interaction_plot(mst_df['FACILITY_TYPE'], mst_df['Capabilities'], mst_df['Capacity'], ax=ax)
plt.show()

# Identifying the interaction among Mobile Logistics, Capacity, and Throughput
mst4_df = sm.OLS.from_formula('Throughput ~ FACILITY_TYPE * Capacity', data = mst_df).fit()

# Creating the interaction plots
fig4, ax = plt.subplots()
sm.graphics.interaction_plot(mst_df['FACILITY_TYPE'], mst_df['Capacity'], mst_df['Throughput'], ax=ax)
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
