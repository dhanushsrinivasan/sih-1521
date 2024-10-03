from flask import Flask, render_template, request, send_from_directory
import numpy as np
from netCDF4 import Dataset
from geopy.geocoders import Nominatim
import os
from datetime import datetime

app = Flask(__name__)

file = 'RF25_ind2022_rfp25.nc'
data = Dataset(file, mode='r')

lats = data.variables['LATITUDE'][:]
longs = data.variables['LONGITUDE'][:]
time = data.variables['TIME'][:]
tave = data.variables['RAINFALL'][:]

geolocator = Nominatim(user_agent="city_coordinates")
indian_cities = ["Pune", "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata"]

city_coordinates = {}
for city in indian_cities:
    location = geolocator.geocode(city + ", India")
    if location:
        city_coordinates[city] = (location.latitude, location.longitude)

# Define the directory containing existing images
image_directory = 'output_images'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_day = request.form['day']
        selected_city = request.form['city']
    else:
        selected_day = "2022-01-01"  # Default to the start date
        selected_city = "Chennai"

    # Convert the selected_day to day of the year
    selected_day_datetime = datetime.strptime(selected_day, '%Y-%m-%d')
    day_of_year = selected_day_datetime.timetuple().tm_yday

    # Calculate the image URL based on day_of_year and selected_city
    image_filename = f'{day_of_year - 1}.jpg'
    image_url = f"/{image_directory}/{image_filename}"

    lat, lon = city_coordinates[selected_city]
    location_lat_range = np.where((lats >= lat - 0.25) & (lats <= lat + 0.25))[0]
    location_lon_range = np.where((longs >= lon - 0.25) & (longs <= lon + 0.25))[0]

    # Calculate the average rainfall for the selected day and city
    rainfall_values = tave[day_of_year - 1, location_lat_range, :][:, location_lon_range]
    avg_rainfall = np.mean(rainfall_values)

    return render_template('index.html', city_coordinates=city_coordinates,
                       selected_day=selected_day, selected_city=selected_city,
                       image_url=image_url, avg_rainfall=avg_rainfall)


# Serve existing images from the output_images directory
@app.route(f'/{image_directory}/<filename>')
def serve_image(filename):
    return send_from_directory(image_directory, filename)


if __name__ == '__main__':
    app.run(debug=True)
