import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from netCDF4 import Dataset
from PIL import Image
import ipywidgets as widgets
from IPython.display import display, clear_output
from geopy.geocoders import Nominatim

# Load the netCDF file
file = 'RF25_ind2022_rfp25.nc'
data = Dataset(file, mode='r')

# Extract latitude, longitude, time, and rainfall data
lats = data.variables['LATITUDE'][:]
longs = data.variables['LONGITUDE'][:]
time = data.variables['TIME'][:]
tave = data.variables['RAINFALL'][:]

# Create a directory to save individual images
import os
if not os.path.exists('output_images'):
    os.makedirs('output_images')

# Generate images for each day
days = np.arange(0, 365)
for i in days:
    plt.figure(figsize=(6, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([68.42, 99.98, 7.45, 37.78], crs=ccrs.PlateCarree())
    c_scheme = plt.pcolormesh(longs, lats, np.squeeze(tave[i, :, :]), cmap='jet', transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle='--')
    ax.set_title('Average Temperature for Day ' + str(i + 1) + ' of year 2022')
    cbar = plt.colorbar(c_scheme, orientation='vertical')
    cbar.set_label('Temperature')
    plt.savefig('output_images/' + str(i) + '.jpg')
    plt.close()

# Load the saved images for creating an animated GIF
image_frames = []
for k in days:
    new_frame = Image.open('output_images/' + str(k) + '.jpg')
    image_frames.append(new_frame)

# Create an interactive dashboard
day_selector = widgets.IntSlider(min=1, max=365, step=1, description='Select Day:')

# Use geopy to obtain coordinates for Indian cities
geolocator = Nominatim(user_agent="city_coordinates")
indian_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata"]

city_coordinates = {}
for city in indian_cities:
    location = geolocator.geocode(city + ", India")
    if location:
        city_coordinates[city] = (location.latitude, location.longitude)

location_selector = widgets.Dropdown(options=city_coordinates.keys(), description='Select City:')

output_image = widgets.Output()
display(day_selector, location_selector, output_image)

def update_image(change):
    day = day_selector.value - 1
    selected_city = location_selector.value

    with output_image:
        clear_output(wait=True)
        display(image_frames[day])
        lat, lon = city_coordinates[selected_city]

        # Find the indices corresponding to the selected city's location
        location_lat_range = np.where((lats >= lat - 0.25) & (lats <= lat + 0.25))[0]
        location_lon_range = np.where((longs >= lon - 0.25) & (longs <= lon + 0.25))[0]

        # Use the first index in the range (you might want to adjust this based on your data)
        location_lat_index = location_lat_range[0]
        location_lon_index = location_lon_range[0]

        # Extract rainfall values for the selected city
        rainfall_values = tave[day, location_lat_index, :][:, location_lon_index]

        avg_rainfall = np.mean(rainfall_values)
        print(f"On Day {day + 1} in {selected_city} (Latitude {lat}, Longitude {lon}):")
        print(f"Average Rainfall: {avg_rainfall:.2f} mm")

day_selector.observe(update_image, 'value')
location_selector.observe(update_image, 'value')

# Initialize the dashboard with the first day and city
update_image(None)
