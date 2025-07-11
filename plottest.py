import cdflib
import numpy as np
import datetime as dt
import spiceypy as spice
import matplotlib.pyplot as plt
from Ef_utilities import *

spice.furnsh("SPICE/JUICE/kernels/mk/juice_ops.tm")
spice.furnsh("SPICE/gsm_frame.tf")

plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.labelcolor'] = 'linecolor'
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['font.family'] = 'Serif'

R_E = 6371.2  # Earth radius in km

def convert_1970(epochs):
    datetimes = []
    for epoch in epochs:
        datetimes.append(dt.datetime(1970, 1, 1) + dt.timedelta(days=epoch))
    return datetimes

def get_JUICE_distance(epochs):
    dist_list = []
    for epoch in epochs:
        # Get the distance from JUICE to Earth in Earth radii
        et = spice.datetime2et(epoch)
        pos, _ = spice.spkpos('JUICE', et, 'GSM', 'NONE', 'EARTH')
        pos = np.array(pos)
        pos_norm = np.linalg.norm(pos)
        pos_norm_earth_radii = pos_norm / R_E
        dist_list.append(float(pos_norm_earth_radii))
    return dist_list

cdf = cdflib.CDF('DATA/JUICE_L1a_RPWI-LP-SID1_RICH_DE763_SNAP_20240820T205007_V03.cdf')

lp_epochs = cdf.varget('Epoch')
lp_epochs = cdflib.cdfepoch.to_datetime(lp_epochs)
lp_epochs = np.array(lp_epochs, dtype='datetime64[ms]').astype('O')

U12 = cdf.varget('LP_DATA')[:,0]
U23 = cdf.varget('LP_DATA')[:,1]
U34 = cdf.varget('LP_DATA')[:,2]
U40 = cdf.varget('LP_DATA')[:,3]

U12 = U12 * TM2diff[0]
U23 = U23 * TM2diff[1]
U34 = U34 * TM2diff[2]
U40 = U40 * TM2diff[3]

U12 = filter_out_start_config_noise(U12, lp_epochs)
U23 = filter_out_start_config_noise(U23, lp_epochs)
U34 = filter_out_start_config_noise(U34, lp_epochs)
U40 = filter_out_start_config_noise(U40, lp_epochs)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(lp_epochs, U12, label='U12')
ax.plot(lp_epochs, U23, label='U23')
ax.plot(lp_epochs, U34, label='U34')
ax.plot(lp_epochs, U40, label='U40')
ax.set_title(f"Caracteristic time: seconds, downsample factor:")
ax.set_xlabel('Epoch \n Distance (R$_E$)')
ax.xaxis.set_label_coords(-0.04, -0.02)
ticks = ax.get_xticks()
ticks = convert_1970(ticks)
distances = get_JUICE_distance(ticks)
for tick, dist in zip(ticks, distances):
    ax.annotate(f"{dist:.1f}", xy=(tick, ax.get_ylim()[0]), xycoords=('data', 'data'),
                xytext=(0, -20), textcoords='offset points',
                ha='center', va='top', fontsize=10, rotation=0)
ax.set_ylabel('Voltage difference (V)')
ax.legend()
ax.grid()

plt.show()