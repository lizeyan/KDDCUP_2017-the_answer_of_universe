# Extract features and labels from raw data and save them into files with ndarray format if it's needed
# In other words, print each number (all are in same type, integer or float) row by row and separate them by spaces
# Please note the meaning of the output ndarray here
import numpy as np
from utility import *
import os


def extract_volume_naive(path_to_file, output_file, weather_data):
    """
    each column of the output ndarray is:
        0   tollgate id,
        1   direction,
        2   average volume in last timewindow,
        3   average volume,
        4   pressure * 100
        5   sea_pressure * 100
        6   wind_direction * 100
        7   wind_speed * 100
        8   temperature * 100
        9   rel_humidity * 100
        10   precipitation * 100
        11  the start time of timewindow with unix timestamp format (the length of timewindow is fixed to 20 minutes)
    """
    if not os.path.exists(output_file):
        # Step 1: Load volume data
        fr = open(path_to_file, 'r')
        fr.readline()  # skip the header
        vol_data = fr.readlines()
        fr.close()

        # Step 2: Create a dictionary to calculate and store volume per time window
        volumes = {}  # key: time window value: dictionary
        for line in vol_data:
            each_pass = line.replace('"', '').split(',')
            tollgate_id = each_pass[1]
            direction = each_pass[2]

            pass_time = each_pass[0]
            pass_time = datetime.strptime(pass_time, "%Y-%m-%d %H:%M:%S")
            time_window_minute = int(np.math.floor(pass_time.minute / 20) * 20)
            # print pass_time
            start_time_window = datetime(pass_time.year, pass_time.month, pass_time.day,
                                         pass_time.hour, time_window_minute, 0)
            # change start time windows into unix timestamp format
            start_time_window = start_time_window.timestamp()

            if tollgate_id not in volumes:
                volumes[tollgate_id] = {}
            if direction not in volumes[tollgate_id]:
                volumes[tollgate_id][direction] = {}
            if start_time_window not in volumes[tollgate_id][direction]:
                volumes[tollgate_id][direction][start_time_window] = 1
            else:
                volumes[tollgate_id][direction][start_time_window] += 1

        weather_times = weather_data[:, 0]
        # Step 3: format output for tollgate and direction per time window
        with open(output_file, "w") as f:
            for tollgate_id, each_tollgate in volumes.items():
                for direction, each_tollgate_direction in each_tollgate.items():
                    time_windows = sorted(list(each_tollgate_direction.keys()))
                    for time_window in time_windows:
                        last_time_window = last_timewindow(time_window, 20 * 60)
                        if last_time_window not in each_tollgate_direction:
                            continue
                        weather_idx = np.searchsorted(weather_times, last_time_window)
                        weather_time = weather_times[weather_idx]
                        if weather_idx >= np.size(weather_times) or weather_time - last_time_window > 3 * 60 * 60:
                            # if the time gap is larger than 3 hours
                            continue
                        ps, sps, wd, ws, tp, rh, pp = weather_data[weather_idx, 1:]
                        print("%s %s %d %d %d %d %d %d %d %d %d %d" % (
                            tollgate_id, direction, each_tollgate_direction[last_time_window],
                            each_tollgate_direction[time_window], ps, sps, wd, ws, tp, rh, pp, time_window),
                              file=f)
        log("save volume data to %s" % output_file)
    log("load volume data from %s" % output_file)
    return np.loadtxt(output_file)


def extract_travel_time_naive(path_to_file, output_file, volume_data):
    """
    each column of the output ndarray is:
        tollgate id,
        intersection id,
        average travel time in last timewindow * 100,
        average travel time * 100,
        the start time of timewindow with unix timestamp format (the length of timewindow is fixed to 20 minutes)
    note that average travel time is timed by 100
    """
    # Step 1: Load trajectories
    if not os.path.exists(output_file):
        fr = open(path_to_file, 'r')
        fr.readline()  # skip the header
        traj_data = fr.readlines()
        fr.close()

        # Step 2: Create a dictionary to store travel time for each route per time window
        travel_times = {}
        for line in traj_data:
            each_traj = line.replace('"', '').split(',')
            intersection_id = each_traj[0]
            tollgate_id = each_traj[1]

            route_id = intersection_id + '-' + tollgate_id
            if route_id not in travel_times.keys():
                travel_times[route_id] = {}

            trace_start_time = each_traj[3]
            trace_start_time = datetime.strptime(trace_start_time, "%Y-%m-%d %H:%M:%S")
            time_window_minute = np.math.floor(trace_start_time.minute / 20) * 20
            start_time_window = datetime(trace_start_time.year, trace_start_time.month, trace_start_time.day,
                                         trace_start_time.hour, time_window_minute, 0)
            start_time_window = start_time_window.timestamp()
            tt = float(each_traj[-1])  # travel time

            if start_time_window not in travel_times[route_id].keys():
                travel_times[route_id][start_time_window] = [tt]
            else:
                travel_times[route_id][start_time_window].append(tt)

        # Step 3: Calculate average travel time for each route per time window
        with open(output_file, "w") as f:
            volume_time_windows = volume_data[:, -1]
            for route, each_route in travel_times.items():
                route_time_windows = sorted(list(each_route.keys()))
                for time_window_start in route_time_windows:
                    last_time_window = last_timewindow(time_window_start, 20 * 60)
                    if last_time_window not in route_time_windows:
                        continue
                    volume_idx = np.searchsorted(volume_time_windows, last_time_window)
                    if volume_idx >= np.size(volume_time_windows) or volume_time_windows[volume_idx] != last_time_window:
                        continue
                    volume, ps, sps, wd, ws, tp, rh, pp = volume_data[volume_idx, 3:11]
                    tt_set = np.asarray(each_route[time_window_start]).astype(float)
                    avg_tt = int(np.mean(tt_set) * 100)
                    tt_set_last = np.asarray(each_route[last_time_window]).astype(float)
                    avg_tt_last = int(np.mean(tt_set_last) * 100)
                    intersection_id, tollgate_id = route.split('-')
                    intersection_id = ord(intersection_id)
                    print("%s %s %d %d %d %d %d %d %d %d %d %d %d" % (
                        tollgate_id, intersection_id, avg_tt_last, avg_tt, volume, ps, sps, wd, ws, tp, rh, pp,
                        time_window_start), file=f)
            log("save travel time data to %s" % output_file)
    log("load travel time data from %s" % output_file)
    return np.loadtxt(output_file)


def extract_weather(path_to_file, output_file) -> np.ndarray:
    """
    output data format:
        0   unix timestamp
        1   pressure * 100
        2   sea_pressure * 100
        3   wind_direction * 100
        4   wind_speed * 100
        5   temperature * 100
        6   rel_humidity * 100
        7   precipitation * 100
    """
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
                weather_time = datetime(weather_day.year, weather_day.month, weather_day.day, weather_hour).timestamp()
                pressure, sea_pressure, wind_direction, wind_speed, temperature, rel_humidity, precipitation = tuple(
                    int(eval(item) * 100) for item in line[2:])
                print("%d %d %d %d %d %d %d %d" % (
                    weather_time, pressure, sea_pressure, wind_direction, wind_speed, temperature, rel_humidity,
                    precipitation), file=f)
        log("save weather data to %s" % output_file)
    log("load weather data from %s" % output_file)
    return np.loadtxt(output_file)
