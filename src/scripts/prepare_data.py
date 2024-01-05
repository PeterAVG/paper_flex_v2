#%%
import numpy as np
import pandas as pd

from src.prepare_problem import BUS, NB_BUS, OD_COLS, TMP_COLS

df = pd.read_csv("../../data/chunk2.csv")
setpoint = -18.5

tc_meas = df[TMP_COLS].values[:, BUS].reshape(-1, 1)
od_meas = (df[OD_COLS].values / 100)[:, BUS].reshape(-1, 1)
eta = 3

R_eta_steady_state = (df.room_temp.values - setpoint) / (
    df.Po_ss.values * np.mean(od_meas) / NB_BUS * eta
)
day_filter = ((df.Hour.values >= 6) & (df.Hour.values <= 22)).astype(int)
night_filter = 1 - day_filter

ta = df.room_temp.values
t_air = tc_meas.reshape(-1)
temperature_filter = (np.diff(t_air, prepend=0) > 4).astype(int)

# print(R_eta_steady_state * (day_filter * 1.246 + night_filter * 1.5))

pd.DataFrame(
    {
        "t": np.array(list(range(1, tc_meas.shape[0] + 1, 1))) / 4,
        "Tc": tc_meas.reshape(-1),
        "OD": od_meas.reshape(-1),
        "Pt": df.Pt.values.reshape(-1) / NB_BUS,
        "day_filter": day_filter,
        "night_filter": night_filter,
        "Rt": R_eta_steady_state,
        "defrost_filter": temperature_filter,
        "Ta": ta,
    }
).to_csv("../../data/sde.csv", index=False)
