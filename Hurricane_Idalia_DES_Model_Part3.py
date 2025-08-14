#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:55:30 2025

@author: dequeishadiggs
"""
#%%
# Installing packages
! pip3 install simpy
! pip install pyshp
! pip install matplotlib
! pip install basemap-data-hires
! pip install adjustText

# Importing Packages
import random
import simpy
import math
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scipy.stats as stats
import seaborn as sns
import shapefile
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from adjustText import adjust_text
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from matplotlib.animation import FuncAnimation
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from os import pipe
pipe
pd.options.display.max_rows = 9999

#%%
# Loading Data
# Reading in Mobile_Logistic_Nodes,Planned Network Path Locations
# And Logistic Nodes CSV Files

mln_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Updated_CSV/Mobile_Logistic_Nodes/Mobile_Logistic_Nodes.csv')
pnpl_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Updated_CSV/Planned Network Path Locations/Planned Network Path Locations.csv')
ln_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Updated_CSV/Logistic Nodes/Logistic Nodes.csv')
lnh_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Updated_CSV/Logistic Nodes/Logistic Nodes Hospitals.csv')
lns_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Updated_CSV/Logistic Nodes/Logistic Nodes Shelters.csv')
em_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Updated_CSV/Evacuation Mean And Nodes/Evacuation_Mean_(mobile_speed_capacity).csv')
en_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Updated_CSV//Evacuation Mean And Nodes/Evacuation_Nodes(evacuation_zones_by_county).csv')
ena_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Updated_CSV/Evacuation Mean And Nodes/Evacuation_Nodes(affected_areas).csv')
emn_df = pd.read_csv('/workspaces/Hurricane-Idalia-DES-Data/Updated_CSV/Evacuation Mean And Nodes/Evacuation_Mean_(numbers).csv')

# Specifying the Path to the Florida Counties Shapefile by using shapefile
shapefile_path = '/workspaces/Hurricane-Idalia-DES-Data/FL_Counties/tl_2023_12_cousub.shp'

# Creating a shapefile reader object
sf = shapefile.Reader(shapefile_path)

# Accessing Shape and Record Data from sf
shapes = sf.shapes()
records = sf.records()

# Accessing the column names from the field names dataframe
fields = sf.fields[1:]

# Extracting coulmn names from the field names dataframe
field_names = [field[0] for field in fields]

# Printing field_names
print(field_names)

#%%
########################################Adding ID Columns###############################################
# Adding an id column in mln_df, emn_df, and pnpl_df to merge mln_df to emn_df 
# And to merge pnpl_df to emn_df

mln_df['id'] = 1
pnpl_df['id'] = 1
emn_df['id'] = 1

#%%
#############################################More Data Cleaning#########################################
# Conducting inner joins on the id column, dropping duplicates, dropping the id column, and renaming columns

mst_df = pd.merge(mln_df,emn_df, how='inner', on='id').drop_duplicates().drop(columns = ['id'])\
    .rename(columns={'Throughput (Distance)' : 'Throughput'})
pnst_df = pd.merge(pnpl_df,emn_df, how='inner', on='id').drop_duplicates().drop(columns = ['id'])\
    .rename(columns={'Throughput (Distance)' : 'Throughput'})
    
# Dropping unnessary columns from pnst_df
pnst_df = pnst_df.drop(columns=['fb_Longitude', 'fb_Latitude','Unnamed: 0'])

#%%
# Filtering lns_df for Shelters that are not in affected Areas (Evacuation Zones)
# lns_df is being used to filter out shelters from the affected areas located in the en_df data frame.
shelters_df = lns_df[~lns_df['COUNTY'].isin(['ALACHUA', 'BAKER', 'VOLUSIA', 'UNION', 'TAYLOR', 'SUWANNEE'
                                             , 'SUMTER', 'ST. JOHNS', 'SARASOTA', 'PUTNAM', 'PINELLAS'
                                             , 'PASCO', 'NASSAU', 'MARION', 'MANATEE', 'MADISON', 'LEVY'
                                             , 'LEON', 'LAFAYETTE', 'JEFFERSON', 'HILLSBOROUGH', 'HERNANDO'
                                             , 'HAMILTON', 'GULF', 'GILCHRIST', 'FRANKLIN', 'FLAGLER', 'DIXIE'
                                             , 'CITRUS', 'WAKULLA'])]

# Adding a latitude and longitude column in em_df and emn_df that will be 
# populated with Kingston Beach Coordinates(Lat, Long)
em_df['latitude'] = 29.9293
em_df['longitude'] = 83.6001
emn_df['latitude'] = 29.9293
emn_df['longitude'] = 83.6001

# Flitering pnpl_df for the Kingston Beach Flordia Taylor County
sites_df = pnpl_df[pnpl_df['COUNTY'].isin(['TAYLOR'])].drop_duplicates()

#%%
####################################Getting the Shape of the Data#######################################
# This portion of the script was added to get the number of rows and columns within each data frame

print(np.shape(mln_df),np.shape(mst_df), np.shape(pnst_df),
      np.shape(ln_df), np.shape(lnh_df), np.shape(lns_df), np.shape(en_df), np.shape(em_df))

#%%
#################################Descriptives(Summaries of the Data Frames)##############################

# Printing Descriptive Summeries of the dataframes
print(mln_df.describe())
print(mst_df.describe())
print(pnst_df.describe())
print(ln_df.describe())
print(lnh_df.describe())
print(lns_df.describe())
print(en_df.describe())
print(em_df.describe())

#%%          
###############################################Simulation################################################
# For variables defined within the function vehical types (capabilities), 
# locations(evacuation locations), site (FACILITY_TYPE), and shelters(COUNTY)
######################Defining the Simulation Class########################################################

# Defining the environment, by creating the environment class and initializing capabilities, locations, 
# sites, and shelters.
# Class for Simulation of Hurricane Idalia
class HurricaneIdaliaSimulation:
    def __init__(self, env, capabilities, evacuation_locations, sites, shelters):
        self.env = env
        self.capabilities = []
        self.evacuation_locations = evacuation_locations
        self.sites = sites
        self.shelters = shelters
        
        # Collecting data for mobile logistics (capabilities) concerning (speed, throughput, capacity)
        # This is information being collected from capabilities
        self.vehicle_speeds = []
        self.vehicle_throughputs = []  
        self.vehicle_capacities = []  
        self.evacuated_people = [] 
            
        # Initializing capabilities in each Evacuation Location
        for loc in evacuation_locations:
            for cap in capabilities:
                vehicle = ResponseVehicle(env, cap['name'], loc['location'], cap['Speed'], cap['Throughput'], cap['Capacity'])
                self.capabilities.append(vehicle)
                self.vehicle_speeds.append(vehicle.speed)
                self.vehicle_throughputs.append(vehicle.throughput)
                self.vehicle_capacities.append(vehicle.capacity)
                self.evacuated_people.append(0)

        # Starting the evacuation process
        self.env.process(self.run_simulation())

    def run_simulation(self):
        # Creating a loop that will process until all people are evacuated
        while any(site['people'] > 0 for site in self.sites):
            for vehicle in self.capabilities:
                if any(site['people'] > 0 for site in self.sites):
                    # Passing self.evacuation_locations to evacuate
                    yield self.env.process(vehicle.evacuate(self.sites, self.shelters, self.evacuation_locations))
            yield self.env.timeout(1)
            
#%%
# Creating a function to calculate distance between two latitudes and longitudes points
# By incorporating the Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 3960  # Calcuating the radius of the Earth in miles
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c  # Calcuating the Distance in miles
    return distance

#%%
# Defining kingston_beach_location and Evacuation and evacuation_locations (Kingston Beach, Florida)
kingston_beach_location = {
    'name': 'Kingston Beach, Taylor County, Florida',
    'latitude': 29.9293,
    'longitude': -83.6001,
    'location': 0
}

evacuation_locations = [
    {
        'location': 0,
        'type': 'Evacuation Point',
        'description': 'Kingston Beach, FL Evacuation Starting Point.',
        'latitude': 29.9293,
        'longitude': -83.6001
    }
]

#%%
# Generating a Class for Response Vehicles
class ResponseVehicle:
    def __init__(self, env, name, evacuation_location, speed, throughput, capacity):
        self.env = env
        self.name = name
        self.evacuation_location = evacuation_location
        self.speed = speed
        self.throughput = throughput
        self.capacity = capacity
        self.path_latitudes = []  # Tracking latitude paths
        self.path_longitudes = []  # Tracking longitude paths
        self.evacuated_people = 0  # Tracking the number of people each vehicle has evacuated 


    def get_location_coordinates(self, location_id, evacuation_locations):
        # Identifying the locations in the evacuation locations list
        location = next((loc for loc in evacuation_locations if loc['location'] == location_id), None)
        # Ensureing latitude and longitude keys are present
        if location:
            return location['latitude'], location['longitude']
        else:
            raise ValueError(f"Location with ID {location_id} is missing latitude or longitude.")                        
        
    def evacuate(self, sites, shelters, evacuation_locations):
        if not any(site['people'] > 0 for site in sites):
            return "No people left to evacuate"
        
        # Randomly choosing sites with people to evacuate
        site = random.choice([s for s in sites if s['people'] > 0])
        # Selecting shelters that are not full to capacity
        shelter = self.shelter_selection(shelters)
        start_lat, start_lon = self.get_location_coordinates(self.evacuation_location, evacuation_locations)
        print(f"{self.env.now}: {self.name} departing from {self.evacuation_location} to {site['name']}")

        # Tracking the vehicle's starting location to show the vehicles paths from the orgin (evacuation location, Kinston Beach, FL) 
        # to sites then the final destination (shelters)
        self.path_latitudes.append(start_lat)
        self.path_longitudes.append(start_lon)
        
        # Simulating travel to the site
        travel_time = self.travel_time(start_lat, start_lon, site['latitude'], site['longitude'])
        yield self.env.timeout(travel_time)
        
        # Update the path for the site to depict the Response Vehicle movement and travel path
        self.path_latitudes.append(site['latitude'])
        self.path_longitudes.append(site['longitude'])
        
        # Evacuating people based on vehicle capacity and site population
        evacuated_people = min(site['people'], self.capacity)
        site['people'] -= evacuated_people
        self.evacuated_people += evacuated_people
        print(f"{self.env.now}: {self.name} evacuating {evacuated_people} people from {site['name']} to {shelter['name']}")
        
        # Simulating travel to the shelters
        travel_time = self.travel_time(site['latitude'], site['longitude'], shelter['latitude'], shelter['longitude'])
        yield self.env.timeout(travel_time)
        print(f"{self.env.now}: {self.name} arrived at shelter {shelter['name']}")
        
        # Getting capacity values and making it default to zero if no values are present
        shelter_capacity = shelter.get('capacity', 0)
        shelter_capacity -= evacuated_people
        
        # Adding a clause that will not allow vehicals to go shelters that are full to Capacity
        if shelter_capacity <= 0:
            shelter['Risk_Capacity_Space'] = 'Full to Capacity'
        shelter['capacity'] = shelter_capacity  # Updating the shelter's capacity with the new value
       
        # Updating shelter path to depict the Response Vehicle movement and travel path
        self.path_latitudes.append(shelter['latitude'])
        self.path_longitudes.append(shelter['longitude'])

    def travel_time(self, start_lat, start_lon, dest_lat, dest_lon):
        # Calculating distances using the haversine formula
        distance = haversine(start_lat, start_lon, dest_lat, dest_lon)
        # Calculating the travel time (Time = Distance / Speed)
        return distance / self.speed 
    
    def shelter_selection(self, shelters):
        # Selecting shelters that are not full to capacity
        available_shelters = [s for s in shelters if s['Risk_Capacity_Space'] != 'Full to Capacity']
        return random.choice(available_shelters) if available_shelters else random.choice(shelters)

#%%
############################################Plotting and Animating the Simulation Results#############################################
"""
# This portion of the script plots and animates the simulation results on a Flordia map 
# that depicts the counties (Grey), affected areas (Light Red), Evacuation Area (Red) and randomly selected shelters
# outside of the affdected areas(Blue)
"""
# Defining the path to the County Boundary Shapefile
county_shapefile_path = "/workspaces/Hurricane-Idalia-DES-Data/FL_Counties/tl_2023_12_cousub.shp"

# Creating a function that plots the simulation and affected areas
def simulation_and_affected_areas(env, evacuation_locations, sites, shelters, vehicles, county_shapefile_path):
    # Defining the Basemap parameters for Florida
    bm = Basemap(projection='merc', llcrnrlat=24.0, urcrnrlat=31.5,
                  llcrnrlon=-87.5, urcrnrlon=-79.5, resolution='i')    
    
    # Drawing Coastlines, Countries, and Other Map Features
    fig, ax = plt.subplots(figsize=(10, 8))
    bm.drawcoastlines()
    bm.drawcountries()
    bm.drawstates()
    bm.fillcontinents(color='gainsboro', lake_color='paleturquoise')
    bm.drawmapboundary(fill_color='powderblue')

    # Defining the Basemap parameters for Florida
    longitude_min, longitude_max = -83.8, -83.4
    latitude_min, latitude_max = 29.8, 30.1
    
    # Creating a Lists to hold text objects called objects
    objects = []

    # Plotting evacuation locations
    for loc in evacuation_locations:
        if longitude_min <= loc['longitude'] <= longitude_max and latitude_min <= loc['latitude'] <= latitude_max:
            x, y = bm(loc['longitude'], loc['latitude'])
            bm.scatter(x, y, color='red', label='Evacuation Point', marker='*', s=40)

    # Plotting the sites
    site_label = False
    for site in sites:
        if longitude_min <= site['longitude'] <= longitude_max and latitude_min <= site['latitude'] <= latitude_max:
            x, y = bm(site['longitude'], site['latitude'])
            if not site_label:
                bm.scatter(x, y, color='green', label='Site', marker='o', s=10)
                label = ax.annotate(site['name'], xy=(x, y), xytext=(x + 2000, y + 2000), fontsize=4, color='darkgreen', fontweight='bold')
                ax.plot([x, x + 2000], [y, y + 2000], color='forestgreen', linewidth=.5)
                objects.append(label)
                site_label = True
            else:
                bm.scatter(x, y, color='green', marker='o', s=10)
                label = ax.annotate(site['name'], xy=(x, y), xytext=(x + 2000, y + 2000), fontsize=4, color='darkgreen', fontweight='bold')
                ax.plot([x, x + 2000], [y, y + 2000], color='forestgreen', linewidth=.5)
                objects.append(label)

    # Plotting the shelters
    shelter_label = False    
    for shelter in shelters:
        x, y = bm(shelter['longitude'], shelter['latitude'])
        if not shelter_label:
            bm.scatter(x, y, color='blue', label='Shelter', marker='o', s=10)
            label = ax.annotate(shelter['name'], xy=(x, y), xytext=(x + 2000, y + 2000), fontsize=4, color='royalblue', fontweight='bold')
            ax.plot([x, x + 2000], [y, y + 2000], color='royalblue', linewidth=.5)
            objects.append(label)
            shelter_label = True
        else:
            bm.scatter(x, y, color='blue', marker='o', s=10)
            label = ax.annotate(shelter['name'], xy=(x, y), xytext=(x + 2000, y + 2000), fontsize=4, color='royalblue', fontweight='bold')
            ax.plot([x, x + 2000], [y, y + 2000], color='royalblue', linewidth=.5)
            objects.append(label)
            
    # Adjusting label positions to avoid overlap
    adjust_text(objects, ax=ax, force_points=0.5, expand_text=1.0)

    # Creating an Animation function for vehicle paths
    def animation_plot(frame):
        ax.clear()
        bm.drawcoastlines()
        bm.drawcountries()
        bm.drawstates()
        bm.fillcontinents(color='gainsboro', lake_color='paleturquoise')
        bm.drawmapboundary(fill_color='powderblue')

        # Plotting evacuation locations for legend
        for loc in evacuation_locations:
            if longitude_min <= loc['longitude'] <= longitude_max and latitude_min <= loc['latitude'] <= latitude_max:
                x, y = bm(loc['longitude'], loc['latitude'])
                bm.scatter(x, y, color='red', label='Evacuation Point', marker='*', s=40)

        # Plotting sites for legend
        for site in sites:
            if longitude_min <= site['longitude'] <= longitude_max and latitude_min <= site['latitude'] <= latitude_max:
                x, y = bm(site['longitude'], site['latitude'])
                bm.scatter(x, y, color='green', marker='o', s=10)
                label = ax.annotate(site['name'], xy=(x, y), xytext=(x + 2000, y + 2000), fontsize=4, color='darkgreen', fontweight='bold')  # Add label with bold text
                ax.plot([x, x + 2000], [y, y + 2000], color='forestgreen', linewidth=.5)  

        # Plotting shelters for legend
        for shelter in shelters:
            x, y = bm(shelter['longitude'], shelter['latitude'])
            bm.scatter(x, y, color='blue', marker='o', s=10)
            label = ax.annotate(shelter['name'], xy=(x, y), xytext=(x + 2000, y + 2000), fontsize=4, color='royalblue', fontweight='bold')  # Add label with bold text
            ax.plot([x, x + 2000], [y, y + 2000], color='royalblue', linewidth=.5)  
            
        # Adjusting the label positions to avoid overlap
        adjust_text(objects, ax=ax, force_points=0.5, expand_text=1.0)

        # Plotting the vehicle's travel paths (capabilities) and making the paths multiple colors
        vehicle_label = set()
        color_palette = plt.cm.get_cmap('tab20', len(vehicles))
        for idx, vehicle in enumerate(vehicles):
            if len(vehicle.path_latitudes) > frame:
                start_lat = vehicle.path_latitudes[frame]
                start_lon = vehicle.path_longitudes[frame]
                end_lat = vehicle.path_latitudes[frame + 1] if frame + 1 < len(vehicle.path_latitudes) else start_lat
                end_lon = vehicle.path_longitudes[frame + 1] if frame + 1 < len(vehicle.path_longitudes) else start_lon
                start_x, start_y = bm(start_lon, start_lat)
                end_x, end_y = bm(end_lon, end_lat)
                vehicle_name = vehicle.name
                
                # Ensures the vehicle lables are depicted once
                label = vehicle_name if vehicle_name not in vehicle_label else ""
                ax.plot([start_x, end_x], [start_y, end_y], label=label, color=color_palette(idx))
                vehicle_label.add(vehicle_name)
                ax.plot([start_x, end_x], [start_y, end_y], color=color_palette(idx))
                
        # Opening the shapefile from the county_shapefile_path
        sf = shapefile.Reader(county_shapefile_path)

        # Accessing the field names from the shapefile
        fields = sf.fields[1:]
        field_names = [field[0] for field in fields]

        # Creating a clause that checks if 'NAME' is in the field_names dataframe (by using an if else statement)
        if 'NAME' not in field_names:
            print("'NAME' column is not present in the shapefile.")
        else:
            # Accessing the records (rows of data) from the shapefile
            records = sf.records()

            # Creating a list of locations to filter from the fields_names dataframe
            affected_locations = [
                "Gainesville", "Hawthorne", "High Springs-Alachua", "Newberry-Archer", "Waldo",
                "Inverness", "Crystal River", "Hernando Beach", "Cross City North", "Cross City South",
                "Apalachicola", "Carrabelle", "Eastpoint", "Trenton", "Bell", "Port St. Joe", "Wewahitchka",
                "Brooksville", "Hernando Beach", "Ridge Manor", "Spring Hill", "Mayo", "Day",
                "Tampa", "Brandon", "Keystone Heights", "Keystone-Citrus Park", "Plant City", "Ruskin",
                "Tallahassee Central", "Tallahassee East", "Tallahassee Northeast", "Tallahassee Northwest", 
                "Tallahassee South", "Tallahassee Southwest","Cedar Key-Yankeetown", "Chiefland", "Williston-Bronson", 
                "Bradenton", "Longboat Key", "Palmetto", "Whiting Field", "Dade City", "Lacoochee", "New Port Richey", 
                "Port Richey", "Zephyrhills", "Clearwater", "Tarpon Springs", "Sarasota", "Longboat Key", 
                "North Port", "Osprey-Laurel-Nokomis", "Plantation", "Venice", "St. Augustine" , "Fruit Cove", 
                "Hastings", "Bushnell-Center Hill", "Live Oak", "Branford", "Perry North", "Perry South",
                "Jasper", "Jennings", "White Springs"
            ]

            # Filtering the records based on the 'NAME' column
            affected_areas = []
            for i, record in enumerate(records):
                if record[field_names.index('NAME')] in affected_locations:
                    affected_areas.append(sf.shape(i))

            # Plotting the affected areas
            affected_areas_parameters = None
            for shape in affected_areas:
                points = shape.points
                x_coords = [point[0] for point in points]
                y_coords = [point[1] for point in points]          

                # Transforming coordinates to the map's projection
                x_map, y_map = bm(x_coords, y_coords)
                ax.fill(x_map, y_map, color='red', alpha=0.2, edgecolor='none')  # Red with no outline
                
                # Creating a patch for the affected areas in the legend
                if affected_areas_parameters is None:
                    affected_areas_parameters = Patch(color='red', alpha=0.2, label='Affected Areas')

            # Plotting the County Boundary 
            for shape in sf.shapes():
                # Getting the coordinates of the shape
                coords = shape.points
                # Separating x and y coordinates
                x_coords, y_coords = zip(*coords)
                
                # Writing a clause for polygons (County Boundaries)
                if shape.shapeType == shapefile.POLYGON or shape.shapeType == shapefile.POLYLINE:
                    x_coords = list(x_coords)
                    y_coords = list(y_coords)
                    if shape.shapeType == shapefile.POLYGON:
                        x_coords.append(x_coords[0])
                        y_coords.append(y_coords[0])
                        
                    # Plotting the County Boundary and specifing parameters for visulization
                    x_map, y_map = bm(x_coords, y_coords)
                    ax.plot(x_map, y_map, color='black', linewidth=0.5, linestyle='-', marker=None)

            # Creating a custom legend list (legend_parameters) with Sites, Shelters, and Affected Areas
            legend_parameters = []

            # Specifying parameters for (Sites, Shelters, Evacuation Points)
            site_parameters = mlines.Line2D([], [], color='green', label='Site', marker='o', linestyle='None', markersize=5)
            shelter_parameters = mlines.Line2D([], [], color='blue', label='Shelter', marker='o', linestyle='None', markersize=5)
            evacuation_locations_parameters = mlines.Line2D([], [], color='red', label='Evacuation Point', marker='*', linestyle='None', markersize=5)

            # Creating a list called (vehicle_parameters) to specify parameters for the different vehicles in the legend
            vehicle_parameters = []
            color_palette = plt.cm.get_cmap('tab20', len(vehicles))
            for idx, vehicle in enumerate(vehicles):
                vehicle_parameter = mlines.Line2D([], [], color=color_palette(idx), label=vehicle.name)
                vehicle_parameters.append(vehicle_parameter)

            # Setting the legend parameters for (Affected Areas)
            affected_areas_parameters = Patch(color='red', alpha=0.2, label='Affected Areas')

            # Specifying legend parameters ordered by (Points, Lines, Polygons)
            legend_parameters.extend([site_parameters, shelter_parameters, evacuation_locations_parameters] + vehicle_parameters + [affected_areas_parameters])

            # Incoperating the legend
            plt.legend(handles=legend_parameters, loc='lower left', fontsize=12)

        plt.tight_layout()

        # Defining the Plot Labels and Title
        plt.title("Hurricane Idalia Evacuation and Affected Areas Simulation Map")
       
    # Calculating maximum path length from all vehicles
    max_path_length = max([len(vehicle.path_latitudes) for vehicle in vehicles if vehicle.path_latitudes], default=1)

    # Creating the animation and assigning it to a variable
    anim = FuncAnimation(fig, animation_plot, frames=max_path_length, interval=900)

    # Saveing the animation
    anim.save('Hurricane_Idalia_Evacuation_Animation.mp4', writer='ffmpeg')

    # Ensuring the animation is displayed after being saved
    plt.show()

#%%
#################### Defining the Environment, Capabilities, Evacuation Locations, Shelters######################
"""
# Defining the Simulation Parameters According to Hurricane Idalia Proposed DES Model Diagram (Figure 3.4)
# Affected Area / Evacuation Location (Kingston Beach), Planned Network Locations (sites),
# Mobile Logistic Nodes (capabilities) that defines what Capabilities are available, 
# and list the Speed, Throughput, & Capacity of the (capabilities)
# Lastly, Logistic nodes, which are the shelters that are not in the affected areas (path of) Hurricane Idalia 
"""
env = simpy.Environment()

# Creating dataframes for shelters and sites to bring back only the latitudes and longitudes
# then turing the latitudes and longitudes into a list
shelters_lat = shelters_df['rsig_Latitude'].tolist()
shelters_long = shelters_df['rsig_Longitude'].tolist()

sites_lat = sites_df['fs_Latitude'].tolist()
sites_long = sites_df['fs_Longitude'].tolist()
  
# Planned Network Path Locations (Sites)
# sites_df.fs_Longitude, sites_df.fs_Latitude 
sites = [
    {'name': 'Site1', 'evacuation_location': 5, 'latitude': sites_lat[0], 'longitude': sites_long[0], 'people': 50},
    {'name': 'Site2', 'evacuation_location': 15, 'latitude': sites_lat[1], 'longitude': sites_long[1], 'people': 30},
    {'name': 'Site3', 'evacuation_location': 25, 'latitude': sites_lat[2], 'longitude': sites_long[2], 'people': 20},
    {'name': 'Site4', 'evacuation_location': 35, 'latitude': sites_lat[3], 'longitude': sites_long[3], 'people': 40},
    {'name': 'Site5', 'evacuation_location': 45, 'latitude': sites_lat[4], 'longitude': sites_long[4], 'people': 10},
    {'name': 'Site6', 'evacuation_location': 12, 'latitude': sites_lat[5], 'longitude': sites_long[5], 'people': 12},
    {'name': 'Site7', 'evacuation_location': 8, 'latitude': sites_lat[6], 'longitude': sites_long[6], 'people': 20},
    {'name': 'Site8', 'evacuation_location': 2, 'latitude': sites_lat[7], 'longitude': sites_long[7], 'people': 8},
    {'name': 'Site9', 'evacuation_location': 10, 'latitude': sites_lat[8], 'longitude': sites_long[8], 'people': 4}
]

# Mobile Logistic Nodes (Defined by Capabilities, Speed, Throughput, & Capacity)
# Using the same data in em_df not using latitudes and longitudes because it uses
# the same coordinates as the evacuation location
capabilities = [
    {'name': 'Truck', 'Speed': 70, 'Throughput': 40, 'Capacity': 4},
    {'name': 'SUV', 'Speed': 70, 'Throughput': 30, 'Capacity': 5},
    {'name': 'Car', 'Speed': 70, 'Throughput': 30, 'Capacity': 5},
    {'name': 'Bus', 'Speed': 50, 'Throughput': 480, 'Capacity': 48},
    {'name': 'Helicopter', 'Speed': 150, 'Throughput': 320, 'Capacity': 6},
    {'name': 'Plane', 'Speed': 530, 'Throughput': 530, 'Capacity': 150},
    {'name': 'Boat', 'Speed': 40, 'Throughput': 300, 'Capacity': 10}
]

# Logistic Nodes (Defined by Shelters)
# (shelters_df.rsig_Latitude, shelters_df.rsig_Longitude)
shelters = [
    {'name': 'S1', 'location': 3, 'latitude': shelters_lat[0], 'longitude': shelters_long[0], 'Risk_Capacity_Space': 'Full to Capacity'},
    {'name': 'S2', 'location': 18, 'latitude': shelters_lat[1], 'longitude': shelters_long[1], 'Risk_Capacity_Space': 'Empty'},
    {'name': 'S3', 'location': 30, 'latitude': shelters_lat[2], 'longitude': shelters_long[2], 'Risk_Capacity_Space': 'Moderately Full'},
    {'name': 'S4', 'location': 50, 'latitude': shelters_lat[3], 'longitude': shelters_long[3], 'Risk_Capacity_Space': 'Full to Capacity'},
    {'name': 'S5', 'location': 60, 'latitude': shelters_lat[4], 'longitude': shelters_long[4], 'Risk_Capacity_Space': 'Empty'},
    {'name': 'S6', 'location': 40, 'latitude': shelters_lat[5], 'longitude': shelters_long[5], 'Risk_Capacity_Space': 'Moderately Full'}
]

# Initializing and running the simulation
env = simpy.Environment()

# Running the simulation until all people are evacuated
simulation = HurricaneIdaliaSimulation(env, capabilities, evacuation_locations, sites, shelters)

env.run()

# Plotting the Simulation results
simulation_and_affected_areas(env, evacuation_locations, sites, shelters, simulation.capabilities, county_shapefile_path)

# Collecting evacuation data and mobile logistic (capabilities) data from the simulation to
# conduct the validation processes further down in the script for 
# (T-Test, Chi-Squared, Q-Qplot, and Cross-Validation K-Fold)
evacuated_people = [vehicle.evacuated_people for vehicle in simulation.capabilities]
speeds = simulation.vehicle_speeds
throughputs = simulation.vehicle_throughputs
capacities = simulation.vehicle_capacities

# Creating a dataframe to store simulation results for speeds, throughputs, capacities, and evacuated people
mobilel_df = pd.DataFrame({
    'vehicle_speeds': speeds,
    'vehicle_throughputs': throughputs,
    'vehicle_capacities': capacities,
    'evacuated_people': evacuated_people
})

# Creating categorical variables for network paths (Path_A, Path_B) and incoperating them into mobilel_df 
# to observe how the two network paths (A & B) influence (speeds, throughputs, and capacities) when conducting 
# Two-Sample T-Test futher down in the script (Model Validation and Hyperparameter Tuning).
mobilel_df['network_paths'] = ['Path_A', 'Path_B', 'Path_A', 'Path_B', 'Path_A', 'Path_B', 'Path_A']

#%%
#############################Hurricane Idalia Shelters and Affected Areas Map########################################
# This portion of the script is used to visualize Hurricane Idalia Shelters and Affected Areas 

# Defining the path to the County Boundary Shapefiles
county_shapefile_path = "/workspaces/Hurricane-Idalia-DES-Data/FL_Counties/tl_2023_12_cousub.shp"

# Opening the shapefile from the county_shapefile_path
sf = shapefile.Reader(county_shapefile_path)

# Accessing the field names from the shapefile
fields = sf.fields[1:]
field_names = [field[0] for field in fields]

# Creating a clause that checks if 'NAME' is in the field_names dataframe (by using an if else statement)
if 'NAME' not in field_names:
    print("'NAME' column is not present in the shapefile.")
else:
    # Accessing the records (rows of data) from the shapefile
    records = sf.records()

    # Creating a list of locations to filter from the fields_names dataframe
    affected_locations = [
        "Gainesville", "Hawthorne", "High Springs-Alachua", "Newberry-Archer", "Waldo",
        "Inverness", "Crystal River", "Hernando Beach", "Cross City North", "Cross City South",
        "Apalachicola", "Carrabelle", "Eastpoint", "Trenton", "Bell", "Port St. Joe", "Wewahitchka",
        "Brooksville", "Hernando Beach", "Ridge Manor", "Spring Hill", "Mayo", "Day",
        "Tampa", "Brandon", "Keystone Heights", "Keystone-Citrus Park", "Plant City", "Ruskin",
        "Tallahassee Central", "Tallahassee East", "Tallahassee Northeast", "Tallahassee Northwest", 
        "Tallahassee South", "Tallahassee Southwest","Cedar Key-Yankeetown", "Chiefland", "Williston-Bronson", 
        "Bradenton", "Longboat Key", "Palmetto", "Whiting Field", "Dade City", "Lacoochee", "New Port Richey", 
        "Port Richey", "Zephyrhills", "Clearwater", "Tarpon Springs", "Sarasota", "Longboat Key", 
        "North Port", "Osprey-Laurel-Nokomis", "Plantation", "Venice", "St. Augustine" , "Fruit Cove", 
        "Hastings", "Bushnell-Center Hill", "Live Oak", "Branford", "Perry North", "Perry South",
        "Jasper", "Jennings", "White Springs"
    ]

    # Filtering the records based on the 'NAME' column
    affected_areas = []
    for i, record in enumerate(records):
        if record[field_names.index('NAME')] in affected_locations:
            affected_areas.append(sf.shape(i))

    # Creating shelters_df
    shelters_df = pd.DataFrame(shelters_df)

    # Creating a function to plot the County shapefile, shelters_df, and affected_areas
    def county_shelters_affected_areas(county_shapefile_path, shelters_df, affected_areas):
        # Creating a Figure and Axes
        fig, ax = plt.subplots(figsize=(10, 8))

        # Creating a Basemap instance for the Florida Map
        bm = Basemap(projection='merc', llcrnrlat=24.0, urcrnrlat=31.5, 
                      llcrnrlon=-87.5, urcrnrlon=-79.5, resolution='i')

        # Drawing Coastlines, Countries, and Other Map Features
        bm.drawcoastlines()
        bm.drawcountries()
        bm.drawstates()
        bm.fillcontinents(color='gainsboro', lake_color='paleturquoise')
        bm.drawmapboundary(fill_color='powderblue')

        # Plotting shelters from shelters_df (latitudes and longitudes)
        latitudes = shelters_df['rsig_Latitude']
        longitudes = shelters_df['rsig_Longitude']
        x, y = bm(longitudes, latitudes)
        
        # Creating a Line2D for shelters in the legend
        shelters_plot = Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=6, label='Shelters')
        bm.plot(x, y, 'bo', markersize=3)

        # Plotting the affected areas
        affected_area_patch = None
        for shape in affected_areas:
            points = shape.points
            x_coords = [point[0] for point in points]
            y_coords = [point[1] for point in points]          

            # Transforming coordinates to the map's projection
            x_map, y_map = bm(x_coords, y_coords)
            ax.fill(x_map, y_map, color='red', alpha=0.3, edgecolor='none')  # Red with no outline
            
            # Creating a patch for the affected areas in the legend
            if affected_area_patch is None:
                affected_area_patch = Patch(color='red', alpha=0.3, label='Affected Areas')

        # Plotting the County Boundary 
        for shape in sf.shapes():
            # Getting the coordinates of the shape
            coords = shape.points
            # Separating x and y coordinates
            x_coords, y_coords = zip(*coords)
            
            # Writing a clause for polygons (County Boundaries)
            if shape.shapeType == shapefile.POLYGON or shape.shapeType == shapefile.POLYLINE:
                x_coords = list(x_coords)
                y_coords = list(y_coords)
                if shape.shapeType == shapefile.POLYGON:
                    x_coords.append(x_coords[0])
                    y_coords.append(y_coords[0])
                    
                # Plotting the County Boundary and specifing parameters for visulization
                x_map, y_map = bm(x_coords, y_coords)
                ax.plot(x_map, y_map, color='black', linewidth=0.5, linestyle='-', marker=None)
                
        # Creating a custom legend with Shelters and Affected Areas
        plt.legend(handles=[shelters_plot, affected_area_patch], loc='lower left', fontsize=12)

        # Defining the Plot Labels and Title
        ax.set_title("Hurricane Idalia Shelters and Affected Areas Map", fontsize=14)
        
        # Depicting the plot
        plt.show()

    # Plotting the Hurricane Idalia Shelters and Affected Areas
    county_shelters_affected_areas(county_shapefile_path, shelters_df, affected_areas)
    
#%%
###########################################Model Validation and Hyperparameter Tuning################################

"""
""""Hyperparameter Tuning Portion"""
"""
# Hyperparameter Tuning was conducted during the deveopement of the model by adjusting the number of sites 
# and shelters because all componets are set to be sampled randomly. 
# Due to observing and investigating changes in the evacuation times from sites to shelters based on capabilities,
# speed, capacity, and throughput.
# The model componets for sits and shelters were also adjusted due to the number of people 
# being evacuated and transported from various sites and arriving to various shelters outside affected areas
# Hyperparameter Tuning was also conducted when identifying the number of frames need to animate the simulation results
"""

#%%
"""
"""" Model Validation"""
"""
# In this portion of the script a Two-Sample T-Test(independent samples) is being conducted
# to see if there is a statistically significant difference between
# the independent and dependent variables for each research question.
# Within the script if the p_value < 0.05 for each research question then the null hypothesis
# will be rejected, however if p-value is greater than or equal to 0.05 then fail to reject the null hypothesis
# will be printed (this means the alternative hypothesis will be accepted) (This also applies for the Chi-Squared Test)
"""

# RQ1: How would planned network path locations influence speed for mobile logistics
# post-Hurricane Idalia hitting Keaston Beach, Florida?
t_stat, p_val = stats.ttest_ind(mobilel_df[mobilel_df['network_paths'] == 'Path_A']['vehicle_speeds'], 
                                mobilel_df[mobilel_df['network_paths'] == 'Path_B']['vehicle_speeds'])
if p_val < 0.05:
    print(f"RQ1: Reject Null Hypothesis - Planned network path locations do not influence speed \
for mobile logistics post-Hurricane Idalia hitting Keaston Beach, Florida. (p-value: {p_val})")
else:
    print(f"RQ1: Fail to reject Null Hypothesis - Planned network path locations influence speed \
for mobile logistics post-Hurricane Idalia hitting Keaston Beach, Florida.  (p-value: {p_val})")
   

# RQ2: How would planned network path locations influence throughput for mobile logistics
# post-Hurricane Idalia hitting Keaston Beach, Florida?
t_stat2, p_val2 = stats.ttest_ind(mobilel_df[mobilel_df['network_paths'] == 'Path_A']['vehicle_throughputs'], 
                                   mobilel_df[mobilel_df['network_paths'] == 'Path_B']['vehicle_throughputs'])
if p_val2 < 0.05:
    print(f"RQ2: Reject Null Hypothesis - Planned network path locations do not influence throughput for \
mobile logistics post-Hurricane Idalia hitting Keaston Beach, Florida. (p-value: {p_val2})")
else:
    print(f"RQ2: Fail to reject Null Hypothesis - Planned network path locations influence throughput for \
mobile logistics post-Hurricane Idalia hitting Keaston Beach, Florida. (p-value: {p_val2})")


# RQ3: How would capacities influence speed for mobile logistics post-Hurricane 
# Idalia hitting Keaston Beach, Florida
t_stat3, p_val3 = stats.ttest_ind(mobilel_df['vehicle_capacities'], mobilel_df['vehicle_speeds'])
if p_val3 < 0.05:
    print(f"RQ3: Reject Null Hypothesis - Capacities do not influence speed for mobile logistics\
post-Hurricane Idalia hitting Keaston Beach, Florida. (p-value: {p_val3})")
else:
    print(f"RQ3: Fail to reject Null Hypothesis - Capacities influence speed for mobile logistics\
post-Hurricane Idalia hitting Keaston Beach, Florida. (p-value: {p_val3})")


# RQ4: How would capacities influence throughput for mobile logistics post-Hurricane
# Idalia hitting Keaston Beach, Florida?
t_stat4, p_val4 = stats.ttest_ind(mobilel_df['vehicle_capacities'], mobilel_df['vehicle_throughputs'])
if p_val4 < 0.05:
    print(f"RQ4: Reject Null Hypothesis - Capacities do not influence throughput for mobile logistics\
post-Hurricane Idalia hitting Keaston Beach, Florida. (p-value: {p_val4})")
else:
    print(f"RQ4: Fail to reject Null Hypothesis - Capacities influence throughput for mobile logistics\
post-Hurricane Idalia hitting Keaston Beach, Florida. (p-value: {p_val4})")


# RQ5: How would logistics node locations, capabilities, and capacities influence 
# speed for mobile logistics post-Hurricane Idalia hitting Keaston Beach, Florida?
t_stat5, p_val5 = stats.ttest_ind(mobilel_df[mobilel_df['network_paths'] == 'Path_A']['vehicle_speeds'], 
                                  mobilel_df[mobilel_df['network_paths'] == 'Path_B']['vehicle_speeds'])
if p_val5 < 0.05:
    print(f"RQ5: Reject Null Hypothesis - Logistics node locations, capabilities, and capacities\
do not influence the speed of mobile logistics post-Hurricane Idalia hitting Keaston Beach, Florida. (p-value: {p_val5})")
else:
    print(f"RQ5: Fail to reject Null Hypothesis - Logistics node locations, capabilities, and capacities\
influence speed for mobile logistics post-Hurricane Idalia hitting Keaston Beach, Florida. (p-value: {p_val5})")


# RQ6: How would logistics node locations, capabilities, and capacities 
# influence throughput for mobile logistics post-Hurricane Idalia hitting Keaston Beach, Florida?
t_stat6, p_val6 = stats.ttest_ind(mobilel_df[mobilel_df['network_paths'] == 'Path_A']['vehicle_throughputs'], 
                                  mobilel_df[mobilel_df['network_paths'] == 'Path_B']['vehicle_throughputs'])
if p_val6 < 0.05:
    print(f"RQ6: Reject Null Hypothesis -  Logistics node locations, capabilities, and capacities do not influence\
throughput for mobile logistics post-Hurricane Idalia hitting Keaston Beach, Florida. (p-value: {p_val6})")
else:
    print(f"RQ6: Fail to reject Null Hypothesis -  Logistics node locations, capabilities, and capacities influence\
throughput for mobile logistics post-Hurricane Idalia hitting Keaston Beach, Florida. (p-value: {p_val6})")


# RQ7: How would evacuation mean, capabilities, and capacities influence speed
# post-Hurricane Idalia hitting Keaston Beach, Florida?
t_stat7, p_val7 = stats.ttest_ind(mobilel_df[mobilel_df['vehicle_capacities'] > 7]['vehicle_speeds'], 
                                  mobilel_df[mobilel_df['vehicle_capacities'] <= 7]['vehicle_speeds'])
if p_val7 < 0.05:
    print(f"RQ7: Reject Null Hypothesis - Evacuation means, capabilities, and capacities influence\
speed post-Hurricane Idalia hitting Keaston Beach, Florida. (p-value: {p_val7})")
else:
    print(f"RQ7: Fail to reject Null Hypothesis - Evacuation means, capabilities, and capacities influence\
speed post-Hurricane Idalia hitting Keaston Beach, Florida. (p-value: {p_val7})")


# RQ8: How would evacuation mean, capabilities, and capacities influence throughput 
# post-Hurricane Idalia hitting Keaston Beach, Florida?	
t_stat8, p_val8 = stats.ttest_ind(mobilel_df[mobilel_df['vehicle_capacities'] > 7]['vehicle_throughputs'], 
                                  mobilel_df[mobilel_df['vehicle_capacities'] <= 7]['vehicle_throughputs'])
if p_val8 < 0.05:
    print(f"RQ8: Reject Null Hypothesis - Evacuation means, capabilities, and capacities do not influence\
throughput post-Hurricane Idalia hitting Keaston Beach, Florida. (p-value: {p_val8})")
else:
    print(f"RQ8: Fail to reject Null Hypothesis - Evacuation means, capabilities, and capacities do not influence\
throughput post-Hurricane Idalia hitting Keaston Beach, Florida. (p-value: {p_val8})")
 
    
#%%
"""
Conducting Chi-Squared Test for network paths, speeds, and throughput 
(This portion of the script is only testing research questions 1 & 2 
because we want to see if planned network paths (network_paths) influence speed
(speed_categories) and throughput (throughput_categories)"""

# Performing Chi-Square Test and creating categorical data (speed categories) derived from 
# capabilities speeds values (this falls under Mobile Logistics) that are 
# categorized as 'Low', 'Medium', 'High'
mobilel_df['speed_categories'] = pd.cut(mobilel_df['vehicle_speeds'], bins=[40, 70, 150, 530], labels=['Low', 'Medium', 'High'])

# Generating a Contingency table for Chi-Square test to cross validate (cross-tabulation) between
# network_path and speed_categories (both categorical variables)
# to depict the number of occurrences for each combination among (network_paths and speed_categories)
# to see if a relationship among the two variables exist
contingency_table = pd.crosstab(mobilel_df['network_paths'], mobilel_df['speed_categories'])
chi2_stat, p_val_chi2, dof, expected = stats.chi2_contingency(contingency_table)

"""# If the p_value < 0.05 then the null hypothesis
# will be rejected, however if p-value is greater than or equal to 0.05 then fail to reject the null hypothesis
# will be printed (this means the alternative hypothesis will be accepted) for RQ1"""
# RQ1: How would planned network path locations influence speed for mobile logistics
# post-Hurricane Idalia hitting Keaston Beach, Florida?

if p_val_chi2 < 0.05:
    print(f"Chi-Square Test RQ1: Reject Null Hypothesis - Planned network path locations do not influence speed\
for mobile logistics post-Hurricane Idalia hitting Keaston Beach, Florida. (p-value: {p_val_chi2})")
else:
    print(f"Chi-Square Test RQ1: Fail to reject Null Hypothesis - Planned network path locations influence speed\
for mobile logistics post-Hurricane Idalia hitting Keaston Beach, Florida.   (p-value: {p_val_chi2})")

print(contingency_table)


# Performing Chi-Square Test and creating categorical data (throughput_categories) derived from 
# capabilities throughput values (this falls under Mobile Logistics) that are 
# categorized as 'Low', 'Medium', 'High'
mobilel_df['throughput_categories'] = pd.cut(mobilel_df['vehicle_throughputs'], bins=[30, 300, 480, 530], labels=['Low', 'Medium', 'High'])

# Generating a Contingency table for Chi-Square test to cross validate (cross-tabulation) between
# network_paths and throughput_categories to depict the number of occurrences for each combination 
# to see if a relationship among the two variables exist
contingency_table2 = pd.crosstab(mobilel_df['network_paths'], mobilel_df['throughput_categories'])
chi2_stat, p_val_chi2, dof, expected = stats.chi2_contingency(contingency_table2)

"""# If the p_value < 0.05 then the null hypothesis
# will be rejected, however if p-value is greater than or equal to 0.05 then fail to reject the null hypothesis
# will be printed (this means the alternative hypothesis will be accepted) for RQ2"""
# RQ2: How would planned network path locations influence throughput for mobile logistics
# post-Hurricane Idalia hitting Keaston Beach, Florida?

if p_val_chi2 < 0.05:
    print(f"Chi-Square Test RQ2: Reject Null Hypothesis - Planned network path locations do not influence\
throughput for mobile logistics post-Hurricane Idalia hitting Keaston Beach, Florida.(p-value: {p_val_chi2})")
else:
    print(f"Chi-Square Test RQ2: Fail to reject Null Hypothesis - Planned network path locations influence\
throughput for mobile logistics post-Hurricane Idalia hitting Keaston Beach, Florida. (p-value: {p_val_chi2})")
    
print(contingency_table2)

#%%
""" Generating Quantile-Quantile (Q-Q) Plots and Z-Scores for speeds, throughput, capacities, and evacuated_people """

# Q-Q Plot: Checking normality for speeds by visually comparing vehical_speeds to normal distribution
plt.figure(figsize=(6, 6))
stats.probplot(mobilel_df['vehicle_speeds'], dist="norm", plot=plt)
plt.title('Vehicle Speeds Q-Q Plot')
plt.show()

# Calculating the Mean and Standard Standard Deviation for vehicle_speeds
mean_speeds = mobilel_df['vehicle_speeds'].mean()  
std_speeds = mobilel_df['vehicle_speeds'].std()  

# Calculating the Z-Scores for Each Data Point
mobilel_df['z_score_speed'] = (mobilel_df['vehicle_speeds'] - mean_speeds) / std_speeds
# Printing z-scores for vehicle_speeds
print(mobilel_df.z_score_speed)

# Z-Score Plot: ] Visulizing Z-scores for vehicle_speeds in a histogram to see the distribute
plt.hist(mobilel_df['z_score'], bins=30, edgecolor='black')
plt.title('Vehicle Speeds Z-Scores Distribution')
plt.xlabel('Z-Score')
plt.ylabel('Frequency')
plt.show()


# Q-Q Plot: Checking normality for throughouput by visually comparing vehical_throughput to normal distribution
plt.figure(figsize=(6, 6))
stats.probplot(mobilel_df['vehicle_throughputs'], dist="norm", plot=plt)
plt.title('Vehicle Throughputs Q-Q Plot')
plt.show()

# Calculating the Mean and Standard Standard Deviation for vehicle_throughputs
mean_throughputs = mobilel_df['vehicle_throughputs'].mean()  
std_throughputs = mobilel_df['vehicle_throughputs'].std()  

# Calculating the Z-Scores for Each Data Point
mobilel_df['z_score_throughput'] = (mobilel_df['vehicle_throughputs'] - mean_throughputs) / std_throughputs
# Printing z-scores for vehicle_throughputs
print(mobilel_df.z_score_throughput)

# Z-Score Plot: ] Visulizing Z-scores for vehicle_throughputs in a histogram to see the distribute
plt.hist(mobilel_df['z_score'], bins=30, edgecolor='black')
plt.title('Vehicle Throughputs Z-Scores Distribution')
plt.xlabel('Z-Score')
plt.ylabel('Frequency')
plt.show()


# Q-Q Plot: Checking normality for capacity by visually comparing vehicle_capacities to normal distribution
plt.figure(figsize=(6, 6))
stats.probplot(mobilel_df['vehicle_capacities'], dist="norm", plot=plt)
plt.title('Vehicle Capacities Q-Q Plot')
plt.show()

# Calculating the Mean and Standard Standard Deviation for vehicle_capacities
mean_capacities= mobilel_df['vehicle_capacities'].mean()  
std_capacities = mobilel_df['vehicle_capacities'].std()  

# Calculating the Z-Scores for Each Data Point
mobilel_df['z_score_capacities'] = (mobilel_df['vehicle_capacities'] - mean_capacities) / std_capacities
# Printing z-scores for vehicle_capacities
print(mobilel_df.z_score_capacities)

# Z-Score Plot: ] Visulizing Z-scores for vehicle_capacities in a histogram to see the distribute
plt.hist(mobilel_df['z_score'], bins=30, edgecolor='black')
plt.title('Vehicle Capacities Z-Scores Distribution')
plt.xlabel('Z-Score')
plt.ylabel('Frequency')
plt.show()


# Q-Q Plot for Checking normality for evacuated people by visually comparing evacuated_people to normal distribution
plt.figure(figsize=(6, 6))
stats.probplot(mobilel_df['evacuated_people'], dist="norm", plot=plt)
plt.title('Evacuated People Q-Q Plot')
plt.show()

# Calculating the Mean and Standard Standard Deviation for evacuated_people
mean_evacuated_people = mobilel_df['evacuated_people'].mean()  
std_evacuated_people = mobilel_df['evacuated_people'].std()  

# Calculating the Z-Scores for Each Data Point
mobilel_df['z_score'] = (mobilel_df['evacuated_people'] - mean_evacuated_people ) / std_evacuated_people 
# Printing z-scores for evacuated_people
print(mobilel_df.z_score)

# Z-Score Plot: ] Visulizing Z-scores for evacuated_people in a histogram to see the distribute
plt.hist(mobilel_df['z_score'], bins=30, edgecolor='black')
plt.title('Evacuated People Z-Scores Distribution')
plt.xlabel('Z-Score')
plt.ylabel('Frequency')
plt.show()

#%%
""" Performing K-Fold Cross Validation for speeds, throughput, capacities, and evacuated people """

# Evaluating the performance of the Hurrican Idalia Simulation Model
# Setting predictors (speeds, throughputs, and capacities) and target variable (evacuated people) for the model 
# so the model can provide an estimate of evacuated people needed to better fit the model
X = mobilel_df[['vehicle_speeds', 'vehicle_throughputs', 'vehicle_capacities']]
y = mobilel_df['evacuated_people']

# Standardizing the data so that it has a mean of 0 and a standard deviation of 1
# to assit with better performance and coverage for each feature to be scaled
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Conduting K-Fold Cross Validation using Linear Regression by generating
# a crosss validation object that will be split into 5 (subsets) to
# train and test the five splits of data that will be randomly shuffled to prevent bias
# The number 42 is just a random number chosen (seeds)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Identifying the best linear between speeds, throughputs, capacities, and evacuated people
model = LinearRegression()

# Performing K-Fold Cross-Validation
cv_results = cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')

print(f"K-Fold Cross Validation Results: {cv_results}")
print(f"Mean Squared Error (MSE) across folds: {-cv_results.mean()}")

#%%
""""Generating Area Under the Curve (AUC) of the Receiver Operating Characteristic (ROC) Curve """""

# Creating a Receiver Operating Characteristic (ROC)Curve to identify the level of Prediction for the DES Model 
# and AUC Based on Evacuated People that features vehicle_speeds, vehicle_throughputs, and vehicle_capacities
# and targets evacuation_high

# Converting regression targets to binary classification (evacuated_people)
threshold = mobilel_df['evacuated_people'].median()
mobilel_df['evacuation_high'] = (mobilel_df['evacuated_people'] > threshold).astype(int)

# Specifing Features and targets vehicle_speeds, vehicle_throughputs, evacuation_high, and vehicle_capacities
X = mobilel_df[['vehicle_speeds', 'vehicle_throughputs', 'vehicle_capacities']]
y = mobilel_df['evacuation_high']

# Standardizing features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying Logistic Regression
clf = LogisticRegression()
clf.fit(X_scaled, y)

# Predicting probabilities
y_prob = clf.predict_proba(X_scaled)[:, 1]

# Computing ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

# Ploting ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for High Evacuation Prediction')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
