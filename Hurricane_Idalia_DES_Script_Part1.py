# -*- coding: utf-8 -*-
"""
Spyder Editor
"""

#%% The purpose of this script is to conduct Data Integration & Transformation, Data Cleaning, &
# Data Preparation before developing Hurricane Idalia's Discrete Event Simulation (DES) Model 

#%%
# Importing Pacakges
import pandas as pd
import numpy as np
pd.options.display.max_rows = 9999

#%%
###############################################Data Intergration & Transformation######################################
# Importing Critcal Facilities (mobile logistic nodes (mln_df), planned network path locations (pnpl_df)
# logistic nodes (ln_df), evacuation nodes (en_df), and evacuation mean (em_df) ), Highwater data (hw)
# and Sensor data (sn)) csv files
hw_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Updated_CSV/HI_Highwater And Sensor/HI_Highwater.csv')
sn_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Updated_CSV/HI_Highwater And Sensor/HI_Sensor.csv')
mln_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Critical Facilities/Mobile_Logistic_Nodes/Bus Terminals.csv')
em_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Critical Facilities/Evacuation_Nodes/Evacuation_Mean_(mobile_speed_capacity).csv')
en_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Critical Facilities/Evacuation_Nodes/Evacuation_Nodes(evacuation_zones_by_county).csv')

#%%
# CSV's for final table pnpl_df = coast guard (cg_df), fire stations (fs_df), food banks (fb_df)
# and national guard (ng_df) datasets
cg_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Critical Facilities/Planned_Network_Path_Loctions/Coast Guard.csv')
fs_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Critical Facilities/Planned_Network_Path_Loctions/Fire Stations.csv')
fb_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Critical Facilities/Planned_Network_Path_Loctions/Food Banks.csv')
ng_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Critical Facilities/Planned_Network_Path_Loctions/National Guard.csv')

#%%
# CSV's for final table ln_df = Airports (private) (airpri_df), airports public (airpub_df)
# disaster recovery centers mobile (drcm_df), disaster recovery centers (drc_df), 
# private schools (prisch_df), public schools (pubsch_df), emergency medical facilities (emf_df)
# emergency operational centers (eoc_df), healthcare facilities (hf_df), hospitals (hos_df), 
# relief agencies (ra_df), risk shelter inventory general (rsig_df), risk shelter inventory pet friendly (rsipf_df)
# and and risk shelter inventory special needs (rsisn_df) 
airpri_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Critical Facilities/Logistic_Nodes/Airports - Private.csv')
airpub_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Critical Facilities/Logistic_Nodes/Airports - Public.csv')
drcm_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Critical Facilities/Logistic_Nodes/Disaster Recovery Centers - Mobile.csv')
drc_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Critical Facilities/Logistic_Nodes/Disaster Recovery Centers.csv')
prisch_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Critical Facilities/Logistic_Nodes/Emergency Medical Services.csv')
pubsch_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Critical Facilities/Logistic_Nodes/Public Schools.csv')
emf_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Critical Facilities/Logistic_Nodes/Emergency Medical Services.csv')
eoc_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Critical Facilities/Logistic_Nodes/Emergency Operations Center.csv')
hf_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Critical Facilities/Logistic_Nodes/Healthcare Facilities.csv')
hos_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Critical Facilities/Logistic_Nodes/Hospitals.csv')
ra_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Critical Facilities/Logistic_Nodes/Relief Agencies.csv')
rsig_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Critical Facilities/Logistic_Nodes/Risk Shelter Inventory - General.csv')
rsipf_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Critical Facilities/Logistic_Nodes/Risk Shelter Inventory - Pet Friendly.csv')
rsisn_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Critical Facilities/Logistic_Nodes/Risk Shelter Inventory - Special Needs.csv')

#%%
############################################Data Cleaning & Data Prepration############################################
# Conducting Data Cleaning Processes removing unnessary columns, filtering only nessary columns
# renaming columns, and reformating lat long for hw_df, sn_df, mln_df, and CSV files for ppl_df
hw_df = hw_df.drop(columns =['latitude_dd', 'longitude_dd', 'site', 'vertical_datums', 'flag_member', 'Links'
                            , 'approval', 'marker', 'survey_member', 'peak_summary', 'verticalDatumName', 'verticalMethodName'
                            , 'approvalMember', 'markerName', 'flagMemberName', 'site_no', 'sitePriorityName', 'networkNames'
                            , 'survey_date', 'approval_id', 'marker_id', 'peak_summary_id', 'survey_member_id', 'last_updated'
                            , 'last_updated_by', 'horizontal_datums', 'vertical_collect_methods', 'horizontal_collect_methods'
                            , 'hwm_types', 'hwm_qualities', 'event', 'siteZone','surveyMemberName', 'files', 'bank', 'flag_member_id'])
sn_df = sn_df.drop(columns = ['housingType', 'Links', 'site', 'event', 'files', 'data_files', 'instrument_status', 'deployment_type'
                              , 'siteHDatum', 'sitePriorityName', 'siteZone', 'deployment_type_id', 'location_description', 'serial_number'
                              , 'housing_serial_number', 'interval', 'inst_collection_id', 'housing_type_id', 'sensor_brand_id', 'vented'
                              , 'last_updated', 'last_updated_by', 'instr_collection_conditions', 'sensor_brand', 'sensor_type', 'housing_type', 'event_id'
                              ,'site_id', 'instrument_id'])
mln_df = mln_df.drop(columns = ['Address', 'Zip', 'USNG', 'OWNERSHIP', 'PARCELID', 'x', 'y', 'LONGLAT'])
cg_df = cg_df.drop(columns = ['Address', 'Zip', 'USNG', 'OWNERSHIP','PARCELID', 'x', 'y', 'LONGLAT'])

# Renaming X and Y columns to Longitude (X) and Latitude (Y)
mln_df = mln_df.rename(columns = {'X':'Longitude', 'Y': 'Latitude'})
cg_df = cg_df.rename(columns = {'X':'Longitude', 'Y': 'Latitude', 'City' : 'CITY'})
                             

#%%
# Filtering fs_df to keep only the nessary columns
fs_df = fs_df.loc[:, ['CITY', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE']]

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
fs_Longitude = []
fs_Latitude = []

# For each row in a varible,
for row in fs_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to fs_Longitude
        fs_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to fs_Latitude
        fs_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to fs_Longitide
        fs_Longitude.append(np.NaN)
        # append a missing value to fs_Latitude
        fs_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
fs_df['fs_Longitude'] = fs_Longitude
fs_df['fs_Latitude'] = fs_Latitude

# Dropping unnessary columns
fs_df = fs_df.drop(columns = ['LONGLAT'])

#%%
# Filtering fb_df to keep only the nessary columns
fb_df = fb_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE']]

# Renaming columns so all dataframes are formated the same
fb_df = fb_df.rename(columns = {'City':'CITY'})

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
fb_Longitude = []
fb_Latitude = []

# For each row in a varible,
for row in fb_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to fb_Longitude
        fb_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to fb_Latitude
        fb_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to fb_Longitide
        fb_Longitude.append(np.NaN)
        # append a missing value to fb_Latitude
        fb_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
fb_df['fb_Longitude'] = fb_Longitude
fb_df['fb_Latitude'] = fb_Latitude

# Dropping unnessary columns
fb_df = fb_df.drop(columns = ['LONGLAT'])

#%%
# Filtering to keep only the nessary columns
ng_df = ng_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE']]

# Renaming columns so all dataframes are formated the same
ng_df = ng_df.rename(columns = {'City':'CITY'})

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
ng_Longitude = []
ng_Latitude = []

# For each row in a varible,
for row in ng_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to ng_Longitude
        ng_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to ng_Latitude
        ng_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to ng_Longitide
        ng_Longitude.append(np.NaN)
        # append a missing value to ng_Latitude
        ng_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
ng_df['ng_Longitude'] = ng_Longitude
ng_df['ng_Latitude'] = ng_Latitude

# Dropping unnessary columns
ng_df = ng_df.drop(columns = ['LONGLAT'])

#%%
# Filtering CSV files for the ln_df datasets to keep only the nessary columns and Renaming columns
airpri_df = airpri_df.loc[:, ['City', 'COUNTY', 'LAT','LONG','NAME_OTHER', 'FACILITY_TYPE'
                      , 'DESCRIPTION', 'FAC_STATUS']]

# Renaming columns so all dataframes are formated the same
airpri_df = airpri_df.rename(columns = {'City':'CITY'})

airpub_df = airpub_df.loc[:, ['City', 'COUNTY', 'LAT','LONG','NAME_OTHER', 'FACILITY_TYPE'
                      , 'DESCRIPTION', 'FAC_STATUS']]

# Renaming columns so all dataframes are formated the same
airpub_df = airpub_df.rename(columns = {'City':'CITY'})

#%%
# Filtering to keep only the nessary columns
drcm_df = drcm_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE']]

# Renaming columns so all dataframes are formated the same
drcm_df = drcm_df.rename(columns = {'City':'CITY'})

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
drcm_Longitude = []
drcm_Latitude = []

# For each row in a varible,
for row in drcm_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to drcm_Longitude
        drcm_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to drcm_Latitude
        drcm_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to drcm_Longitide
        drcm_Longitude.append(np.NaN)
        # append a missing value to drcm_Latitude
        drcm_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
drcm_df['drcm_Longitude'] = drcm_Longitude
drcm_df['drcm_Latitude'] = drcm_Latitude

# Dropping unnessary columns
drcm_df = drcm_df.drop(columns = ['LONGLAT'])

#%%
# Filtering to keep only the nessary columns
drc_df = drc_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE']]

# Renaming columns so all dataframes are formated the same
drc_df = drc_df.rename(columns = {'City':'CITY'})

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
drc_Longitude = []
drc_Latitude = []

# For each row in a varible,
for row in drc_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to drc_Longitude
        drc_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to drc_Latitude
        drc_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to drc_Longitide
        drc_Longitude.append(np.NaN)
        # append a missing value to drc_Latitude
        drc_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
drc_df['drc_Longitude'] = drc_Longitude
drc_df['drc_Latitude'] = drc_Latitude

# Dropping unnessary columns
drc_df = drc_df.drop(columns = ['LONGLAT'])

#%%
# Filtering to keep only the nessary columns
prisch_df = prisch_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE']]

# Renaming columns so all dataframes are formated the same
prisch_df = prisch_df.rename(columns = {'City':'CITY'})

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
prisch_Longitude = []
prisch_Latitude = []

# For each row in a varible,
for row in prisch_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to prisch_Longitude
        prisch_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to prisch_Latitude
        prisch_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to prisch_Longitide
        prisch_Longitude.append(np.NaN)
        # append a missing value to prisch_Latitude
        prisch_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
prisch_df['prisch_Longitude'] = prisch_Longitude
prisch_df['prisch_Latitude'] = prisch_Latitude

# Dropping unnessary columns
prisch_df = prisch_df.drop(columns = ['LONGLAT'])

#%%
# Filtering to keep only the nessary columns
pubsch_df = pubsch_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE']]

# Renaming columns so all dataframes are formated the same
pubsch_df = pubsch_df.rename(columns = {'City':'CITY'})

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
pubsch_Longitude = []
pubsch_Latitude = []

# For each row in a varible,
for row in pubsch_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to pubsch_Longitude
        pubsch_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to pubsch_Latitude
        pubsch_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to pubsch_Longitide
        pubsch_Longitude.append(np.NaN)
        # append a missing value to pubsch_Latitude
        pubsch_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
pubsch_df['pubsch_Longitude'] = pubsch_Longitude
pubsch_df['pubsch_Latitude'] = pubsch_Latitude

#%%
# Filtering to keep only the nessary columns
emf_df = emf_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE']]

# Renaming columns so all dataframes are formated the same
emf_df = emf_df.rename(columns = {'City':'CITY'})

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
emf_Longitude = []
emf_Latitude = []

# For each row in a varible,
for row in emf_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to emf_Longitude
        emf_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to emf_Latitude
        emf_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to emf_Longitide
        emf_Longitude.append(np.NaN)
        # append a missing value to emf_Latitude
        emf_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
emf_df['emf_Longitude'] = emf_Longitude
emf_df['emf_Latitude'] = emf_Latitude

# Dropping unnessary columns
emf_df = emf_df.drop(columns = ['LONGLAT'])

#%%
# Filtering to keep only the nessary columns
eoc_df = eoc_df.loc[:, ['City', 'County', 'LONGLAT', 'FACILITY_TYPE']]

# Renaming columns so all dataframes are formated the same
eoc_df =eoc_df.rename(columns = {'City':'CITY', 'County':'COUNTY'})

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
eoc_Longitude = []
eoc_Latitude = []

# For each row in a varible,
for row in eoc_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to eoc_Longitude
        eoc_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to eoc_Latitude
        eoc_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to eoc_Longitide
        eoc_Longitude.append(np.NaN)
        # append a missing value to eoc_Latitude
        eoc_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
eoc_df['eoc_Longitude'] = eoc_Longitude
eoc_df['eoc_Latitude'] = eoc_Latitude

# Dropping unnessary columns
eoc_df = eoc_df.drop(columns = ['LONGLAT'])

#%%
# Filtering to keep only the nessary columns
hf_df = hf_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE', 'TTL_BEDS']]

# Renaming columns so all dataframes are formated the same
hf_df = hf_df.rename(columns = {'City':'CITY'})

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
hf_Longitude = []
hf_Latitude = []

# For each row in a varible,
for row in hf_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to fb_Longitude
        hf_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to hf_Latitude
        hf_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to hf_Longitide
        hf_Longitude.append(np.NaN)
        # append a missing value to hf_Latitude
        hf_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
hf_df['hf_Longitude'] = hf_Longitude
hf_df['hf_Latitude'] = hf_Latitude

# Dropping unnessary columns
hf_df = hf_df.drop(columns = ['LONGLAT'])

#%%
# Filtering to keep only the nessary columns
hos_df = hos_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE', 'TTL_BEDS']]

# Renaming columns so all dataframes are formated the same
hos_df = hos_df .rename(columns = {'City':'CITY'})

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
hos_Longitude = []
hos_Latitude = []

# For each row in a varible,
for row in hos_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to hos_Longitude
        hos_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to hos_Latitude
        hos_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to hos_Longitide
        hos_Longitude.append(np.NaN)
        # append a missing value to hos_Latitude
        hos_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
hos_df['hos_Longitude'] = hos_Longitude
hos_df['hos_Latitude'] = hos_Latitude

# Dropping unnessary columns
hos_df = hos_df.drop(columns = ['LONGLAT'])

#%%
# Filtering to keep only the nessary columns
ra_df = ra_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE']]

# Renaming columns so all dataframes are formated the same
ra_df = ra_df.rename(columns = {'City':'CITY'})

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
ra_Longitude = []
ra_Latitude = []

# For each row in a varible,
for row in ra_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to ra_Longitude
        ra_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to ra_Latitude
        ra_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to ra_Longitide
        ra_Longitude.append(np.NaN)
        # append a missing value to ra_Latitude
        ra_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
ra_df['ra_Longitude'] = ra_Longitude
ra_df['ra_Latitude'] = ra_Latitude

# Dropping unnessary columns
ra_df = ra_df.drop(columns = ['LONGLAT'])

#%%
# Filtering to keep only the nessary columns
rsig_df = rsig_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE', 'SHELTER_TYPE', 'Risk_Capacity_Spaces',
                          'Evacuation_Zone', 'General_Pop', 'SPECIAL_NEEDS', 'Pet_Friendly']]

# Renaming columns so all dataframes are formated the same
rsig_df = rsig_df.rename(columns = {'City':'CITY'})

# Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
rsig_Longitude = []
rsig_Latitude = []

# For each row in a varible,
for row in rsig_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to rsig_Longitude
        rsig_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to rsig_Latitude
        rsig_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to rsig_Longitide
        rsig_Longitude.append(np.NaN)
        # append a missing value to rsig_Latitude
        rsig_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
rsig_df['rsig_Longitude'] = rsig_Longitude
rsig_df['rsig_Latitude'] = rsig_Latitude

# Dropping unnessary columns
rsig_df = rsig_df.drop(columns = ['LONGLAT'])

#%%
# Filtering to keep only the nessary columns
rsipf_df = rsipf_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE', 'SHELTER_TYPE', 'Risk_Capacity_Spaces',
                          'Evacuation_Zone', 'General_Pop', 'SPECIAL_NEEDS', 'Pet_Friendly']]

# Renaming columns so all dataframes are formated the same
rsipf_df = rsipf_df.rename(columns = {'City':'CITY'})

# Reformating LONGLAT so that Longitude and Latitude columns are separate
# Create two lists for the loop results to be placed
Longitude = []
Latitude = []

# For each row in a varible,
for row in rsipf_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to Longitude
        Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to Latitude
        Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to Longitide
        Longitude.append(np.NaN)
        # append a missing value to Latitude
        Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
rsipf_df['Longitude'] = Longitude
rsipf_df['Latitude'] = Latitude

# Dropping unnessary columns
rsipf_df = rsipf_df.drop(columns = ['LONGLAT'])

#%%
# Filtering to keep only the nessary columns
rsisn_df = rsisn_df.loc[:, ['City', 'COUNTY', 'LONGLAT', 'FACILITY_TYPE', 'SHELTER_TYPE', 'Risk_Capacity_Spaces',
                          'Evacuation_Zone', 'General_Pop', 'SPECIAL_NEEDS', 'Pet_Friendly']]

# Renaming columns so all dataframes are formated the same
rsisn_df = rsisn_df.rename(columns = {'City':'CITY'})

## Reformating LONGLAT so that Longitude and Latitude columns are seperate
# Create two lists for the loop results to be placed
rsisn_Longitude = []
rsisn_Latitude = []

# For each row in a variable,
for row in rsisn_df['LONGLAT']:
    # Try to,
    try:
        # Split the row by comma and append
        # everything before the comma to rsisn_Longitude
        rsisn_Longitude.append(row.split(',')[0])
        # Split the row by comma and append
        # everything after the comma to rsisn_Latitude
        rsisn_Latitude.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to rsisn_Longitide
        rsisn_Longitude.append(np.NaN)
        # append a missing value to rsisn_Latitude
        rsisn_Latitude.append(np.NaN)

# Create two new columns from Longitude and Latitude
rsisn_df['rsisn_Longitude'] = rsisn_Longitude
rsisn_df['rsisn_Latitude'] = rsisn_Latitude

# Dropping unnessary columns
rsisn_df = rsisn_df.drop(columns = ['LONGLAT'])

#%%
# Merging coast guard (cg_df), fire stations (fs_df), food banks (fb_df)
# and national guard (ng_df) data frames to pnpl_df
pnpl_df = cg_df.merge(fs_df, on = ['FACILITY_TYPE', 'COUNTY', 'CITY'], how = 'outer')
pnpl_df = pnpl_df.merge(fb_df, on = ['FACILITY_TYPE', 'COUNTY', 'CITY'], how = 'outer')
pnpl_df = pnpl_df.merge(ng_df, on = ['FACILITY_TYPE', 'COUNTY', 'CITY'], how = 'outer')
# Dataframe has NaN values due to tables not being the same size, 
# however all the data is present for each dataframe

#%%
# Merging Airports(private) (airpri_df), airports public (airpub_df)
# disaster recovery centers mobile (drcm_df), disaster recovery centers (drc_df), 
# private schools (prisch_df), public schools (pubsch_df), emergency medical facilities (emf_df)
# emergency operational centers (eoc_df), healthcare facilities (hf_df), hospitals (hos_df), 
# relief agencies (ra_df), risk shelter inventory general (rsig_df), risk shelter inventory pet friendly (rsipf_df)
# and and risk shelter inventory special needs (rsisn_df) data frames to ln_df

ln_df = airpri_df.merge(airpub_df, on = ['FACILITY_TYPE', 'COUNTY', 'CITY', 'NAME_OTHER', 
                                         'FAC_STATUS', 'DESCRIPTION', 'LAT', 'LONG'], how = 'outer')

ln_df = ln_df.merge(drcm_df, on = ['FACILITY_TYPE', 'COUNTY', 'CITY'], how = 'outer')
ln_df = ln_df.merge(drc_df, on = ['FACILITY_TYPE', 'COUNTY', 'CITY'], how = 'outer')
ln_df = ln_df.merge(prisch_df, on = ['FACILITY_TYPE', 'COUNTY', 'CITY'], how = 'outer')
ln_df = ln_df.merge(pubsch_df, on = ['FACILITY_TYPE', 'COUNTY', 'CITY'], how = 'outer')
ln_df = ln_df.merge(emf_df, on = ['FACILITY_TYPE', 'COUNTY', 'CITY'], how = 'outer')
ln_df = ln_df.merge(eoc_df, on = ['FACILITY_TYPE', 'COUNTY', 'CITY'], how = 'outer')
ln_df = ln_df.merge(ra_df, on = ['FACILITY_TYPE', 'COUNTY', 'CITY'], how = 'outer')

# Creating seperate logistic nodes dataframes for hospitals and shelters
# by merging on like columns
lnh_df = hos_df.merge(hf_df, on = ['FACILITY_TYPE', 'COUNTY', 'CITY', 'TTL_BEDS'], how = 'outer')


lns_df = rsig_df.merge(rsipf_df, on = ['FACILITY_TYPE', 'COUNTY', 'CITY', 'SHELTER_TYPE'
                                   , 'Risk_Capacity_Spaces', 'Evacuation_Zone', 'SPECIAL_NEEDS'
                                   ,'Pet_Friendly', 'General_Pop'], how = 'outer')

lns_df = lns_df.merge(rsisn_df, on = ['FACILITY_TYPE', 'COUNTY', 'CITY', 'SHELTER_TYPE'
                                   , 'Risk_Capacity_Spaces', 'Evacuation_Zone', 'SPECIAL_NEEDS'
                                   ,'Pet_Friendly', 'General_Pop'], how = 'outer')
# Dataframe has NaN values due to tables not being the same size, 
# however all the data is present for each dataframe

#%%
# Exporting (mobile logistic nodes (mln_df), planned path locations (pnpl_df)
# logistic nodes (ln_df, lnh_df, and lns_df), Highwater data (hw), and Sensor data (sn)) to csv files

hw_df.to_csv('HI_Highwater.csv', encoding = 'utf-8')
sn_df.to_csv("HI_Sensor.csv", encoding = 'utf-8')
mln_df.to_csv("Mobile_Logistic_Nodes.csv", encoding = 'utf-8')
pnpl_df.to_csv("Planned Network Path Locations.csv", encoding = 'utf-8')
ln_df.to_csv("Logistic Nodes.csv", encoding = 'utf-8')
lnh_df.to_csv("Logistic Nodes Hospitals.csv", encoding = 'utf-8')
lns_df.to_csv("Logistic Nodes Shelters.csv", encoding = 'utf-8')
