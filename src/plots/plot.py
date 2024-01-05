#%% # noqa
####### PLOT CHUNK DATA #######
from typing import Any, Tuple, cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.common.utils import _set_font_size
from src.prepare_problem import NB_BUS

sns.set_theme()
sns.set(font_scale=1.5)


df = pd.read_csv("data/chunk2.csv")
df["t"] = df["t"] / 4  # converted to hour
df.columns

tmp = "tmp_32.0"
od = "od_32.0"

# create three subplots sharing the x-axis (hours) of "tmp", "od", and Pt
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 10))
ax1.step(df["t"], df[tmp], "b-", label="Air temperature")
ax1.set_ylabel("Temperature [°C]")
ax2.step(df["t"], df[od], "r-", label="Opening degree")
ax2.set_ylabel("OD [%]")
ax3.step(df["t"], df["Pt"] / NB_BUS, "g-", label="Power")
ax3.set_ylabel("Power [kW]")
ax3.set_xlabel("Time [h]")
ax3.set_xlim((0, 24))
ax1.get_yaxis().set_label_coords(-0.1, 0.5)
ax2.get_yaxis().set_label_coords(-0.1, 0.5)
ax3.get_yaxis().set_label_coords(-0.1, 0.5)
ax1.legend()
ax2.legend()
ax3.legend()
# save figure to this folder
_set_font_size([ax1, ax2, ax3], legend=20)
plt.tight_layout()
plt.savefig("tex/figures/tmp_od_Pt.png", dpi=300)
# plt.show()


#%% # noqa
######### PLOT CHUNK SIMULATION OF 2nd ORDER TCL MODEL #########
from typing import Tuple, cast  # noqa

import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import pandas as pd  # noqa
import seaborn as sns  # noqa

from src.prepare_problem import build_uncertainty_set_v2  # noqa
from src.prepare_problem import get_chunk_instance  # noqa

df_scenarios = pd.read_csv("data/scenarios_v2.csv")
scenarios = build_uncertainty_set_v2(df_scenarios, nb=1)

instance = get_chunk_instance(scenarios)

dt = 0.25
_t = np.arange(0.25, 24.25, dt)

cc = instance.c_c
cf = instance.c_f
r_ci = instance.r_ci
epsilon = instance.epsilon
r_cf = instance.r_cf
eta = cast(float, instance.eta)

tc_true = instance.t_c_data
od = instance.od  # type:ignore
assert isinstance(od, np.ndarray)
p_base = instance.p_base
ti = instance.t_i
_filter = instance.temperature_filter


def simulate(rebound: bool, tf_base: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    if rebound:
        assert len(tf_base) == 96

    tf = np.empty(96)
    tc = np.empty(96)

    tf[0] = tc_true[0]
    tc[0] = tc_true[0]

    for i in range(1, 96):
        h = i // 4
        if rebound and h >= 10 and h < 12:
            p = 0.0
        elif rebound and tf[i - 1] > tf_base[i - 1] + 0.1 and h >= 12:
            p = cast(float, instance.p_nom)
        else:
            p = p_base[h]
        # simulate differential equation:
        tf[i] = tf[i - 1] + dt * 1 / cf * (tc[i - 1] - tf[i - 1])
        delta = (
            1
            / cc
            * (
                1 / r_ci[i] * (ti[i] - tc[i - 1])
                + 1 / r_cf * (tf[i - 1] - tc[i - 1])
                - od[i] * eta * p
                + _filter[i] * epsilon
            )
        )
        tc[i] = tc[i - 1] + dt * delta

    return tf, tc


sns.set_theme()
sns.set(font_scale=1.5)

tf1, tc1 = simulate(rebound=False, tf_base=np.empty(1))
fig, ax1 = plt.subplots(1, 2, figsize=(14, 10), sharey=True)
ax1[0].plot(_t, tf1, "b-", label="Sim. food temperature")
ax1[0].plot(_t, tc1, "r-", label="Sim. air temperature")
tf2, tc2 = simulate(rebound=True, tf_base=tf1.copy())
ax1[1].plot(_t, tf2, "b-", label="Sim. food temperature")
ax1[1].plot(_t, tc2, "r-", label="Sim. air temperature")
ax1[0].plot(
    _t,
    tc_true,
    color="black",
    linestyle="--",
    label="Meas. air temperature",
    alpha=0.5,
)
ax1[0].set_ylabel("Temperature [°C]")
ax1[0].set_xlabel("Time [h]")
ax1[1].set_xlabel("Time [h]")
ax1[0].set_xlim((0, 24))
ax1[1].set_xlim((0, 24))
ax1[0].legend(loc="upper right")
ax1[1].legend()
# save figure to this folder
_set_font_size(ax1, legend=20)
plt.tight_layout()
plt.savefig("tex/figures/2ndFreezerModelSimulation.png", dpi=300)
# plt.show()

#%% # noqa

import matplotlib.pyplot as plt  # noqa
import pandas as pd  # noqa
import seaborn as sns  # noqa
from matplotlib.ticker import MaxNLocator  # noqa

######### PLOT CHUNK SCENARIOS #########
from src.prepare_problem import build_uncertainty_set_v2  # noqa

df_scenarios = pd.read_csv("data/scenarios_v2.csv")
scenarios = build_uncertainty_set_v2(df_scenarios, nb=1)

sns.set_theme()
sns.set(font_scale=1.5)

nb_scenarios = scenarios.lambda_rp.shape[0]
t = np.arange(1, 25, 1)
# t2 = np.repeat(t, nb_scenarios, axis=0)
fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 12))
# plot all lambda_rp prices in same plot
for omega in range(nb_scenarios):
    if omega < nb_scenarios - 1:
        ax1.step(t, scenarios.lambda_rp[omega, :], "b-", alpha=0.5, label="_nolegend_")
    else:
        ax1.step(
            t, scenarios.lambda_rp[omega, :], "b-", alpha=0.5, label=r"$\lambda^{b}$"
        )

ax1.step(t, scenarios.lambda_spot[0, :], "r-", linewidth=3, label=r"$\lambda^{s}$")
ax1.step(t, scenarios.lambda_mfrr[0, :], "g-", linewidth=3, label=r"$\lambda^{r}$")
ax1.set_ylabel("Price [DKK/kWh]")
ax1.set_xlabel("Time [h]")
ax1.legend()

x = scenarios.up_regulation_event.sum(axis=1).astype(int)
ix = np.argsort(x)
y = scenarios.prob
ax2.plot(x[ix], y[ix], "o-", linewidth=3, label=r"$\pi_{\omega}$")
ax2.set_xlabel(r"Up-regulation hours in scenario $\omega$")
ax2.set_ylabel("Probability")
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.legend()

# plt.savefig("tex/figures/scenarios.png", dpi=300)
plt.show()
