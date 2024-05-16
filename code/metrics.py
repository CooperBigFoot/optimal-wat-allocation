import numpy as np
import pandas as pd

# Constants and parameters
ETA_NOM = 0.85  # Nominal efficiency
KAPPA = 86400 / (3.6 * 10**6) * 1000  # Conversion factor for power calculation
Q_MAX_T = 2360  # Maximum turbine capacity in m^3/s
Q_MIN_T = 38  # Minimum flow for turbine operation in m^3/s
H_F = 950  # Flood threshold in cm


def compute_zt(rt: pd.Series) -> pd.Series:
    """
    Function to compute the zeta.

    Parameters:
    - rt: array of reservoir release

    Returns:
    - zt: array of zeta.
    """

    return (
        0.000000000003663570691434010 * rt**3
        - 0.000000136377708978325000000 * rt**2
        + 0.002087770897833130000000000 * rt
        + 11.660165118679700000000000000
    )


def OF_hydro(
    inflow: pd.Series, release: pd.Series, reservoir_level: pd.Series
) -> float:
    """
    Objective function for hydropower production.

    Parameters:
    - inflow: array of inflow to the reservoir
    - release: array of reservoir release
    - WaterLevelHanoi: array of reservoir level

    Returns:
    - J_HP: float, the objective function value. [GWh/day]
    """

    H = len(inflow)
    zt = compute_zt(release)

    Pt = np.zeros_like(inflow)

    for i in range(len(inflow)):

        q_T = (
            min(Q_MAX_T, max(Q_MIN_T, release.iloc[i]))
            if release.iloc[i] >= Q_MIN_T
            else 0
        )
        delta_ht = reservoir_level.iloc[i] - zt[i]

        eta_t = ETA_NOM * (
            -0.000747602341932219 * (delta_ht**2)
            + 0.137069286795915 * delta_ht
            + 3.02854144738924
        )

        Pt[i] = KAPPA * eta_t * q_T * delta_ht

    J_HP = np.sum(Pt) / H

    return round(J_HP / 1_000_000, 2)  # Convert to GWh/day


def OF_flood(WaterLevelHanoi: pd.Series) -> float:
    """
    Objective function for flood control.

    Parameters:
    - WaterLevelHanoi: array of water level in Hanoi

    Returns:
    - J_flo: float, the objective function value.
    """

    H = len(WaterLevelHanoi)

    Ft = np.where(WaterLevelHanoi > H_F, (WaterLevelHanoi - H_F) ** 2, 0)
    J_flo = np.sum(Ft) / H

    return round(J_flo, 2)
