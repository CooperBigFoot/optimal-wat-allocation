import numpy as np
from scipy.interpolate import interp1d
from utils import read_a_mat, read_q_mat
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
from typing import Callable


sys_param = {}


def storage_to_level(s: float, sys_param: dict) -> float:
    """
    Function to convert storage to water level.

    Parameters:
    - s: float, the storage value.
    - sys_param: dict, the system parameters.

    Returns:
    - h: float, the water level."""

    lsv = sys_param["lsv"]
    h = interp_lin_scalar(lsv[:, 2], lsv[:, 0], s)
    return h


def level_to_storage(h: float, sys_param: dict) -> float:
    """
    Function to convert water level to storage.

    Parameters:
    - h: float, the water level.
    - sys_param: dict, the system parameters.

    Returns:
    - s: float, the storage value."""

    lsv = sys_param["lsv"]
    s = interp_lin_scalar(lsv[:, 0], lsv[:, 2], h)
    return s


def sim_ann(M: np.ndarray, n: int, theta: np.ndarray, n_type: str) -> np.ndarray:
    """
    Function to simulate an ANN.

    Parameters:
    - M: np.ndarray, the input data.
    - n: int, the number of neurons.
    - theta: np.ndarray, the weights.
    - n_type: str, the activation function.

    Returns:
    - output: np.ndarray, the output data."""

    model = Sequential()
    model.add(
        Dense(
            n, input_dim=M.shape[0], activation="tanh" if n_type == "tansig" else n_type
        )
    )
    model.add(Dense(1, activation="linear"))

    # Reshape theta to match the structure of the model's weights
    weights = [
        theta[: n * M.shape[0]].reshape(M.shape[0], n),
        theta[n * M.shape[0] : n * (M.shape[0] + 1)],
        theta[n * (M.shape[0] + 1) : -1].reshape(n, 1),
        theta[-1:].reshape(
            1,
        ),
    ]
    model.set_weights(weights)

    # Assuming M is your input data
    output = model.predict(M.T)  # Keras models expect input as (N, r)
    return output.flatten()


def sim_hb(qq: dict, h_in: float, policy: tuple, sys_param: dict) -> tuple[np.ndarray]:
    """
    Function to simulate the Hoa Binh reservoir.

    Parameters:
    - qq: dict, a dictionary containing the streamflow data.
    - h_in: float, the initial water level.
    - policy: tuple, the operating policy.
    - sys_param: dict, the system parameters.

    Returns:
    - h: np.ndarray, the water level.
    - u: np.ndarray, the release.
    - r: np.ndarray, the actual release.
    - ht_HN: np.ndarray, the estimated water level in Hanoi."""

    # Simulation settings
    q_sim = np.append(np.nan, qq["q_Da"])
    H = len(q_sim) - 1

    # Initialization
    h = np.full(H+1, np.nan)
    s = np.full(H+1, np.nan)
    r = np.full(H+1, np.nan)
    u = np.full(H+1, np.nan)

    # Start simulation
    h[0] = h_in
    s[0] = level_to_storage(h[0], sys_param)

    for t in range(H):
        u[t] = std_operating_policy(h[t], policy)
        s[t + 1], r[t + 1] = mass_balance(s[t], u[t], q_sim[t + 1])
        h[t + 1] = storage_to_level(s[t + 1], sys_param)

    # ANN based routing to estimate water level in Hanoi
    q_YB_sim = np.append(np.nan, qq["q_YB"])
    q_VQ_sim = np.append(np.nan, qq["q_VQ"])
    M = np.vstack((r, q_YB_sim, q_VQ_sim)).T
    N = M.shape[0]

    # Normalize inputs
    M = (M - np.array(sys_param["sh"]["min_input"])) / (
        np.array(sys_param["sh"]["max_input"]) - np.array(sys_param["sh"]["min_input"])
    )

    # ANN model to estimate water level
    ht_HN = sim_ann(
        M.T,
        sys_param["hanoi"]["neuron"],
        sys_param["hanoi"]["theta"],
        sys_param["hanoi"]["neuron_type"],
    ).T
    # Scale outputs
    ht_HN = (
        ht_HN * (sys_param["sh"]["max_output"][0] - sys_param["sh"]["min_output"][0])
        + sys_param["sh"]["min_output"][0]
    )

    return h, u, r, ht_HN


def model_setup(
    ini_date: str,
    fin_date: str,
    path_a_hoabinh_K: str,
    path_q_yenbai: str,
    path_q_vuquang: str,
) -> tuple[dict, dict]:
    """
    Function to setup the model for simulation.

    Parameters:
    - ini_date: str, the initial date for simulation.
    - fin_date: str, the final date for simulation.
    - path_a_hoabinh_K: str, the path to the a_hoabinh_K.mat file.
    - path_q_yenbai: str, the path to the q_yenbai.mat file.
    - path_q_vuquang: str, the path to the q_vuquang.mat file.

    Returns:
    - qq: dict, a dictionary containing the streamflow data.
    - sys_param: dict, the system parameters."""

    ini_date_m = pd.to_datetime(ini_date)
    fin_date_m = pd.to_datetime(fin_date)

    # Load data
    a_hoabinh_K: pd.DataFrame = read_a_mat(path_a_hoabinh_K)
    q_yenbai: pd.DataFrame = read_q_mat(path_q_yenbai)
    q_vuquang: pd.DataFrame = read_q_mat(path_q_vuquang)

    # Extract relevant data based on dates
    a = a_hoabinh_K[ini_date_m:fin_date_m]
    q_YB = q_yenbai[ini_date_m:fin_date_m]
    q_VQ = q_vuquang[ini_date_m:fin_date_m]

    qq = {"q_Da": a, "q_YB": q_YB, "q_VQ": q_VQ}

    # Load reservoir and downstream model parameters
    sys_param["lsv"] = np.loadtxt(
        "/Users/cooper/Desktop/optimal-wat-allocation/Script/MAT_data/lsv_rel_HoaBinh.txt"
    ).T
    sys_param["maxRel"] = np.loadtxt(
        "/Users/cooper/Desktop/optimal-wat-allocation/Script/MAT_data/max_release_HoaBinh.txt"
    ).T
    sys_param["sh"] = {
        "min_input": [0, 0, 0],
        "max_input": [58315, 31164, 36917],
        "min_output": [0, 0],
        "max_output": [15, 25900],
    }
    sys_param["hanoi"] = {
        "theta": np.loadtxt(
            "/Users/cooper/Desktop/optimal-wat-allocation/Script/MAT_data/HN_theta_n8_m0.txt"
        ),
        "neuron": 8,
        "neuron_type": "tansig",
    }

    sys_param["warmup"] = 62  # first day to compute objective functions

    return qq, sys_param


def interp_lin_scalar(x: np.ndarray, y: np.ndarray, new_x: float) -> float:
    """
    Performs linear interpolation for a scalar.

    Parameters:
    - x: np.ndarray, array of x-coordinates (independent variables).
    - y: np.ndarray, array of y-coordinates (dependent variables).
    - new_x: float, the x-value to interpolate.

    Returns:
    - new_y: float, the interpolated y-value.
    """
    # Ensure that x and y are numpy arrays for interp1d compatibility
    x = np.array(x)
    y = np.array(y)

    # Create interpolation function
    interpolation_function = interp1d(x, y, kind="linear", fill_value="extrapolate")

    # Use the interpolation function to find the interpolated value
    new_y = interpolation_function(new_x)

    # Return the interpolated value as a scalar
    return new_y.item()


def min_release(s: float, sys_param: dict) -> float:
    """
    Function to compute the minimum release.

    Parameters:
    - s: float, the storage value.
    - sys_param: dict, the system parameters.

    Returns:
    - q: float, the minimum release."""

    h = storage_to_level(s, sys_param)

    if h > 117.3:
        q = max_release(s, sys_param)
    elif h >= 117.0:
        q = interp_lin_scalar(
            [117.0, 117.3],
            [0, max_release(storage_to_level(117.3, sys_param), sys_param)],
            h,
        )
    else:
        q = 0.0

    return q


def max_release(s: float, sys_param: dict) -> float:
    """
    Function to compute the maximum release.

    Parameters:
    - s: float, the storage value.
    - sys_param: dict, the system parameters.
    - storage_to_level: Callable, the function to convert storage to water level.

    Returns:
    - V: float, the maximum release."""

    MR = sys_param["maxRel"]
    h = storage_to_level(s, sys_param)

    V = interp_lin_scalar(MR[:, 0], MR[:, 1], h)
    return V


def mass_balance(
    s: float,
    u: float,
    q: float,
) -> tuple[float, float]:
    """
    Function to compute the mass balance.

    Parameters:
    - s: float, the storage value.
    - u: float, the release.
    - q: float, the inflow.

    Returns:
    - s1: float, the new storage value.
    - r1: float, the actual release."""

    HH = 24
    delta = 3600
    s_ = np.full(HH + 1, np.nan)
    r_ = np.full(HH + 1, np.nan)

    s_[0] = s
    for i in range(HH):
        qm = min_release(s_[i], sys_param)
        qM = max_release(s_[i], sys_param)
        r_[i + 1] = min(qM, max(qm, u))
        s_[i + 1] = s_[i] + delta * (q - r_[i + 1])

    s1 = s_[HH]
    r1 = np.nanmean(r_[1:])
    return s1, r1


def std_operating_policy(h: float, policy: tuple, debug: bool = False) -> float:
    """
    Function to implement the standard operating policy.

    Parameters:
    - h: float, the water level.
    - policy: tuple, the operating policy.
    - debug: bool, whether to print the policy.

    Returns:
    - u: float, the release."""

    h1, h2, m1, m2, w = policy

    if debug:
        print(h1, h2, m1, m2, w)

    m1 = np.tan(m1)
    m2 = np.tan(m2)

    if h >= h1 and h < h2:
        u = w
    elif h >= h2:
        u = w + m2 * (h - h2)
    else:
        u = max(0, m1 * (h - h1))

    return u
