#Importing necessary python libraries to complete mini project
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openmeteo_requests
import requests_cache
from retry_requests import retry

#This is the code Open-Meteo provided after selecting the data needed for the project

#Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

#Make sure all required weather variables are listed here
#The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
params = {
	"latitude": 36.137415, #cooridinates for Red Rock Canyon
	"longitude": -115.43129,
	"hourly": ["temperature_2m", "relative_humidity_2m", "precipitation_probability", "precipitation", "wind_speed_10m"],
	"timezone": "America/Los_Angeles",
	"wind_speed_unit": "mph",
	"temperature_unit": "fahrenheit",
	"precipitation_unit": "inch",
	"start_date": "2026-01-19",
	"end_date": "2026-04-24",
}
responses = openmeteo.weather_api(url, params = params)

#Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

#Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_precipitation_probability = hourly.Variables(2).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(3).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(4).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time() + response.UtcOffsetSeconds(), unit = "s", utc = True),
	end =  pd.to_datetime(hourly.TimeEnd() + response.UtcOffsetSeconds(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["precipitation_probability"] = hourly_precipitation_probability
hourly_data["precipitation"] = hourly_precipitation
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m

hourly_dataframe = pd.DataFrame(data = hourly_data)
hourly_dataframe

#Checking the df information
hourly_dataframe.info()

#Checking description of df
hourly_dataframe.describe()

#Looking at the rows with missing data
hourly_dataframe[hourly_dataframe['temperature_2m'].isnull()]

#Looking at the data for the first row that has data for all columns
hourly_dataframe.iloc[728]

#Dropping all rows with any missing values
weather_df = hourly_dataframe.dropna()

#Checking df information and confirming all rows with missing values have been removed
weather_df.info()

#Checking df summary and finding that everything looks normal
weather_df.describe()
weather_df

#Plotting line graph of temperature
#Defining x and y points
x_points = weather_df['date']
y_points = weather_df['temperature_2m']

#Graphing x and y in line graph
plt.plot(x_points, y_points)
plt.title('Hourly Temperature')
plt.xlabel('Date and Hour')
plt.ylabel('Temperature (°F)')
plt.show()

#Plotting line graph of humidity

#Defining x and y points
x_points = weather_df['date']
y_points = weather_df['relative_humidity_2m']

#Graphing x and y in line graph
plt.plot(x_points, y_points)
plt.title('Hourly Humidity')
plt.xlabel('Date and Hour')
plt.ylabel('Humidity (%)')
plt.show()

#Plotting line graph of precipitation

#Defining x and y points
x_points = weather_df['date']
y_points = weather_df['precipitation']

#Graphing x and y in line graph
plt.plot(x_points, y_points)
plt.title('Hourly Precipitation')
plt.xlabel('Date and Hour')
plt.ylabel('Precipitation (inches)')
plt.show()

#Plotting line graph of wind speed

#Defining x and y points
x_points = weather_df['date']
y_points = weather_df['wind_speed_10m']

#Graphing x and y in line graph
plt.plot(x_points, y_points)
plt.title('Hourly Wind Speed')
plt.xlabel('Date and Hour')
plt.ylabel('Wind Speed (MPH)')
plt.show()

#I wanted to create new column in the dataframe called Ideal Climbing Conditions
#My ideal climbing conditions:
#Ideally, I would want to climb in the hours between 10:00 to 19:00. Since, I am currently unemployed and have the freedom to climb during the day every day. Ideal date would date >=10:00 & <=19:00.
#In Red Rock Canyon, the rock type is sandstone and the local ethics is to not climb on wet rock. Sandstone is a fragile and pourous stone that needs to be dry when climbed. Sandstone usually 24 to 48 hours to dry. I will need to create a new column for a rolling sum of the precipitation for the last 48 hours. I will also need to create another new column to indicate if the rock is dry and definite dry rock as if rolling sum of precipation is equal to 0.
#My ideal temputare to climb in is from °45 to °65 (temperature_2m >=45 & <=65)]
#My ideal humidity is less than 50%
#My ideal wind speed is less than 25 mph (wind_speed_10m <=25)


          
