import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from metrics import OF_flood, OF_hydro
from sim import (
    model_setup,
    sim_hb,
    std_operating_policy,
    max_release,
    min_release,
    level_to_storage,
)
import matplotlib.pyplot as plt
import seaborn as sns


def plot_policy(
    policies: list[tuple[float, float, float, float, float]],
    sys_params: dict,
    output_destination: str = None,
    names: list[str] = None,
) -> None:
    """
    Function to plot the policies.

    Parameters:
    - policies list of tuples, the policy parameters
    - sys_params: dict, the system parameters. Output from model_setup.
    - output_destination: str, the output destination for the plot.
    - names: list of str, the names of the policies

    Returns:
    - None
    """

    h_levels = np.arange(75, 125, 0.5)
    s_levels = [level_to_storage(h, sys_params) for h in h_levels]
    max_releases = [max_release(s, sys_params) for s in s_levels]
    min_releases = [min_release(s, sys_params) for s in s_levels]

    for i, policy in enumerate(policies):
        policy_release = [
            std_operating_policy(h, policy, debug=False) for h in h_levels
        ]

        df = pd.DataFrame(
            {
                "h": h_levels,
                "s": s_levels,
                "policy_release": policy_release,
                "max_release": max_releases,
                "min_release": min_releases,
            }
        )

        df["actual_release"] = np.where(
            (df["policy_release"] < df["max_release"])
            & (df["policy_release"] >= df["min_release"]),
            df["policy_release"],
            df["max_release"],
        )

        plt.plot(df["h"], df["actual_release"], label=names[i])

    plt.plot(
        df["h"],
        df["max_release"],
        label="Max Release",
        linestyle="-.",
        alpha=0.5,
        color="black",
    )
    plt.plot(
        df["h"],
        df["min_release"],
        label="Min Release",
        linestyle="--",
        alpha=0.5,
        color="black",
    )
    plt.xlabel("Reservoir Level (m)")
    plt.ylabel("Release (m$^3$/s)")
    sns.despine()
    plt.grid(alpha=0.2, linestyle="--", color="#0000FF")
    plt.xticks(np.arange(75, 125, 5))
    plt.yticks(np.arange(0, 50000, 5000))
    plt.legend()

    if output_destination:
        plt.savefig(output_destination, dpi=300, bbox_inches="tight")
    plt.show()


def plot_policy_ts(
    policies: list[pd.DataFrame], names: list[str], output_destination: str
):
    """
    Function to plot the policy time series.

    Parameters:
    - policies: list of pd.DataFrame, the output of the simHB function in matlab
    - names: list of str, the names of the policies
    - output_destination: str, the output destination for the plot

    Returns:
    - None
    """

    layout = (2, 2)
    ax_ReservoirLevel = plt.subplot2grid(layout, (0, 0), colspan=2)
    ax_Release = plt.subplot2grid(layout, (1, 0), colspan=1)
    ax_h = plt.subplot2grid(layout, (1, 1), colspan=1)

    for policy, name in zip(policies, names):
        ax_ReservoirLevel.plot(
            policy.index, policy["ReservoirLevel"], label=f"Reservoir Level: {name}"
        )
        ax_Release.plot(
            policy.index, policy["Release"], label=f"Release: {name}", alpha=0.7
        )
        ax_h.plot(
            policy.index,
            policy["WaterLevelHanoi"],
            label=f"Hanoi Water Level: {name}",
            alpha=0.7,
        )

    ax_ReservoirLevel.set_xlabel("")
    ax_ReservoirLevel.set_ylabel("Monthly Reservoir Level (m)")
    ax_ReservoirLevel.grid(alpha=0.2, linestyle="--", color="#0000FF")
    ax_ReservoirLevel.legend(loc="upper right")

    ax_Release.set_xlabel("")
    ax_Release.set_ylabel("Monthly Release (m$^3$/s)")
    ax_Release.grid(alpha=0.2, linestyle="--", color="#0000FF")
    ax_Release.legend(loc="upper right")

    ax_h.axhline(y=9.5, color="black", linestyle="--", label="Flooding threshold")
    ax_h.set_xlabel("")
    ax_h.set_ylabel("Water level in Hanoi (cm)")
    ax_h.grid(alpha=0.2, linestyle="--", color="#0000FF")
    ax_h.legend(loc="upper right")

    sns.despine()

    if output_destination:
        plt.savefig(output_destination, dpi=300, bbox_inches="tight")

    plt.show()


def process_policy(path_to_policy: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Function to process policy data.

    Parameters:
    - path_to_policy: str, path to the policy_XX.csv file
    - start_date: str, start date for the time series
    - end_date: str, end date for the time series

    Returns:
    - policy: pd.DataFrame, the processed policy data
    """

    HEADER = ["ReservoirLevel", "ReleaseDecision", "Release", "WaterLevelHanoi"]
    policy = pd.read_csv(path_to_policy, header=None, names=HEADER)
    policy["Date"] = pd.date_range(start=start_date, end=end_date, freq="D")
    return policy.set_index("Date")


def find_best_policy(
    criteria: str, start_date: str, end_date: str, qq: dict, folder_name: str
) -> dict:
    """
    Function to find the best policy.

    Parameters:
    - criteria: str, the criteria for selecting the best policy. Choose from 'compromise', 'flood', 'hydro'.
    - start_date: str, start date for the time series
    - end_date: str, end date for the time series
    - qq: dict, the dictionary containing the streamflow data. Output from model_setup.
    - folder_name: str, the folder containing the policy data.

    Returns:
    - result: dict, the result containing the best policy and objective function values
    """

    if criteria not in ["compromise", "flood", "hydro"]:
        raise ValueError(
            "Invalid criteria. Choose from 'compromise', 'flood', 'hydro'."
        )

    load_dotenv()
    workspace = os.getenv("WORKSPACE")
    data_folder = os.path.join(workspace, folder_name)

    best_j_flood = best_j_hydro = -np.inf
    best_policy = None

    for file in os.listdir(data_folder):
        if file == "policy_summary.csv":
            continue

        path_to_policy = os.path.join(data_folder, file)
        policy = process_policy(path_to_policy, start_date, end_date)

        j_flood = -OF_flood(policy["WaterLevelHanoi"] * 100)
        res_inflow = qq["q_Da"]["1995-01-01":"2006-12-31"]
        res_release = policy["Release"]["1995-01-01":"2006-12-31"]
        res_lvl = policy["ReservoirLevel"]["1995-01-01":"2006-12-31"]
        j_hydro = OF_hydro(res_inflow, res_release, res_lvl)

        policy_number = file.replace("policy_", "").replace(".csv", "")

        if (
            criteria == "compromise"
            and j_flood > best_j_flood
            and j_hydro > best_j_hydro
        ):
            best_j_flood = j_flood
            best_j_hydro = j_hydro
            best_policy = policy_number
        elif criteria == "flood" and j_flood > best_j_flood:
            best_j_flood = j_flood
            best_j_hydro = j_hydro
            best_policy = policy_number
        elif criteria == "hydro" and j_hydro > best_j_hydro:
            best_j_flood = j_flood
            best_j_hydro = j_hydro
            best_policy = policy_number

    return {
        "policy": int(best_policy),
        "OF_flood": -best_j_flood,
        "OF_hydro": best_j_hydro,
        "criteria": criteria,
    }
