from functools import reduce

from extract_features import *
from scipy.stats import *
from sklearn import tree, ensemble, svm, linear_model, neural_network
from concurrent.futures import *
import datetime



d = datetime.datetime.fromtimestamp(1476784800)
str1 = d.strftime("%Y-%m-%d %H:%M:%S.%f")
print(str1)
exit()

output_file = "o.txt"
path_to_file = "./training/weather (table 7)_training.csv"


if not os.path.exists(output_file):
    fr = open(path_to_file, "r")
    fr.readline()
    weather_data = fr.readlines()
    fr.close()

    with open(output_file, "w") as f:
        for line in weather_data:
            line = line.replace('"', '').split(',')
            weather_day = line[0]
            weather_day = datetime.strptime(weather_day, "%Y-%m-%d")
            weather_hour = int(line[1])
            weather_time = datetime(weather_day.year, weather_day.month, weather_day.day, weather_hour, 0, 0).timestamp()
            pressure, sea_pressure, wind_direction, wind_speed, temperature, rel_humidity, precipitation = tuple(
                int(eval(item) * 100) for item in line[2:])
            print("%d %d %d %d %d %d %d %d" % (
                weather_time, pressure, sea_pressure, wind_direction, wind_speed, temperature, rel_humidity,
                precipitation), file=f)