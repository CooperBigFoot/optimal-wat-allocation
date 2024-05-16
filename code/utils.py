import scipy.io
import numpy as np
import pandas as pd
import os
from jdcal import jd2gcal


def read_q_mat(path: str) -> pd.DataFrame:
    """
    Read the q_[station].mat file and return the data as pandas DataFrame.

    Parameters:
    - path: str, path to the .mat file

    Returns:
    - df: pd.DataFrame
    """

    filename = os.path.basename(path)
    station_name = os.path.splitext(filename)[0]

    mat = scipy.io.loadmat(path)
    q = mat[station_name]
    signal = q[0][0]["signal"]
    time = signal[0][0]["time"]
    value = signal[0][0]["value"]
    time = np.mean(time, axis=1)
    value = value.flatten()
    df = pd.DataFrame({"time": time, "value": value})
    df["time"] = df["time"].apply(lambda x: jd2gcal(x, 0))
    df["time"] = pd.to_datetime(
        df["time"].apply(lambda x: f"{int(x[0])}-{int(x[1])}-{int(x[2])}")
    )
    df.set_index("time", inplace=True)

    df.rename(columns={"value": "streamflow_[m3/s]"}, inplace=True)

    return df


def read_a_mat(path: str) -> pd.DataFrame:
    """
    Function to load the a_hoabinh_K.mat file into a DataFrame.

    Parameters:
    - path: str, the path to the MATLAB file.

    Returns:
    - df: pd.DataFrame, the DataFrame containing the data.
    """

    mat = scipy.io.loadmat(path)

    kali_tmp = mat["Kali_tmp"]

    signal_content = kali_tmp[0, 0]["signal"][0, 0]

    time_array = signal_content["time"]
    value_array = signal_content["value"]

    time_values = np.ravel(time_array[:, 0])
    value_values = np.ravel(value_array)

    df = pd.DataFrame({"Time": time_values, "Value": value_values})

    start_date = "1956-01-01"
    dates = pd.date_range(start=start_date, periods=len(df), freq="D")

    df["Time"] = dates
    df.set_index("Time", inplace=True)

    df.rename(columns={"Value": "Streamflow_[m3/s]"}, inplace=True)

    return df

