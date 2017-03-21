# the naive model
# Do keep everything local as possible as you can.
from extract_features import *


def main():
    prepare_data("naive")


def prepare_data(method, clean=False):
    if clean:
        os.remove("weather.data")
        os.remove("volume.data")
        os.remove("travel_time.data")

    weather_data, volume_data, travel_time_data = None, None, None
    if method is "naive":
        # Note that the sort is required
        # Note that the index of features is hard coded, be careful
        weather_data = extract_weather("./training/weather (table 7)_training.csv", "weather.data")
        weather_data = weather_data[weather_data[:, 0].argsort()]
        volume_data = extract_volume_naive("./training/volume(table 6)_training.csv", "volume.data", weather_data)
        volume_data = volume_data[volume_data[:, -1].argsort()]
        travel_time_data = extract_travel_time_naive("./training/trajectories(table 5)_training.csv",
                                                     "travel_time.data", volume_data)
    return weather_data, volume_data, travel_time_data


if __name__ == "__main__":
    main()
