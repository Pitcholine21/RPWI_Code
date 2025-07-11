import cdflib
import numpy as np
import datetime as dt
import spiceypy as spice
import matplotlib.pyplot as plt
from Ef_utilities import *
from utilities import print_entire, print_info

R_E = 6371  # Earth radius

# Get cdfs and JUICE state vectors

erg = cdflib.CDF("DATA/erg_orbs_l2_20240820000000_20240821000000_cdaweb.cdf")

erg_pos = erg.varget('pos_gsm')
erg_epoch = erg.varget('epoch')
erg_epoch = cdflib.cdfepoch.to_datetime(erg_epoch)
erg_epoch = np.array(erg_epoch, dtype='datetime64[ms]').astype('O')

cdfs = []

cdfs.append(cdflib.CDF('DATA/JUICE_L1a_RPWI-LP-SID1_RICH_DE763_SNAP_20240820T180737_V03.cdf'))
cdfs.append(cdflib.CDF('DATA/JUICE_L1a_RPWI-LP-SID1_RICH_DE763_SNAP_20240820T184752_V03.cdf'))
cdfs.append(cdflib.CDF('DATA/JUICE_L1a_RPWI-LP-SID1_RICH_DE763_SNAP_20240820T192916_V03.cdf'))
cdfs.append(cdflib.CDF('DATA/JUICE_L1a_RPWI-LP-SID1_RICH_DE763_SNAP_20240820T200931_V03.cdf'))
cdfs.append(cdflib.CDF('DATA/JUICE_L1a_RPWI-LP-SID1_RICH_DE763_SNAP_20240820T205007_V03.cdf'))
cdfs.append(cdflib.CDF('DATA/JUICE_L1a_RPWI-LP-SID1_RICH_DE763_SNAP_20240820T212819_V03.cdf'))
cdfs.append(cdflib.CDF('DATA/JUICE_L1a_RPWI-LP-SID1_RICH_DE763_SNAP_20240820T220245_V03.cdf'))
cdfs.append(cdflib.CDF('DATA/JUICE_L1a_RPWI-LP-SID1_RICH_DE763_SNAP_20240820T223650_V03.cdf'))
cdfs.append(cdflib.CDF('DATA/JUICE_L1a_RPWI-LP-SID1_RICH_DE763_SNAP_20240820T231519_V03.cdf'))
cdfs.append(cdflib.CDF('DATA/JUICE_L1a_RPWI-LP-SID1_RICH_DE763_SNAP_20240820T235531_V03.cdf'))

# Getting JUICE state vectors

spice.furnsh("SPICE/JUICE/kernels/mk/juice_ops.tm")
spice.furnsh("SPICE/gsm_frame.tf")

downsample_factor = 50  # Downsample factor for epochs

lp_epochs = np.array([])

for cdf in cdfs:
    lp_epoch = cdf.varget('Epoch')
    lp_epoch = cdflib.cdfepoch.to_datetime(lp_epoch)
    lp_epoch = np.array(lp_epoch, dtype='datetime64[ms]').astype('O')
    lp_epochs = np.concatenate((lp_epochs, lp_epoch))

states = np.zeros((len(lp_epochs[::downsample_factor]), 6))

for i, epoch in enumerate(lp_epochs[::downsample_factor]):
    print(f"Processing epoch {i+1}/{len(lp_epochs[::downsample_factor])}: {epoch.isoformat()}")
    et = spice.str2et(epoch.isoformat())
    state, _ = spice.spkezr('JUICE', et, 'GSM', 'NONE', 'EARTH')
    states[i] = state

spice.kclear()

# Look at JUICE position
x = states[:, 0]
y = states[:, 1]
z = states[:, 2]

tha = cdflib.CDF('DATA/tha_l1s_state_20240820000000_20240821000000_cdaweb.cdf')
thb = cdflib.CDF('DATA/thb_l1s_state_20240820000000_20240821000000_cdaweb.cdf')
thc = cdflib.CDF('DATA/thc_l1s_state_20240820000000_20240821000000_cdaweb.cdf')

tha_pos = tha.varget('tha_pos_gsm')
thb_pos = thb.varget('thb_pos_gsm')
thc_pos = thc.varget('thc_pos_gsm')

tha_epoch = tha.varget('tha_state_time')

tha_epoch = [dt.datetime(1970, 1, 1, 0, 0, 0) + dt.timedelta(seconds=float(epoch)) for epoch in tha_epoch]

xa =  tha_pos[:, 0]
ya =  tha_pos[:, 1]
za =  tha_pos[:, 2]

xb =  thb_pos[:, 0]
yb =  thb_pos[:, 1]
zb =  thb_pos[:, 2]

xc =  thc_pos[:, 0]
yc =  thc_pos[:, 1]
zc =  thc_pos[:, 2]

# Define the time window
start_time = dt.datetime(2024, 8, 20, 19, 0, 0)
end_time = dt.datetime(2024, 8, 21, 0, 0, 0)

# Filter tha_epoch for times within the window
selected_indices = [i for i, t in enumerate(tha_epoch) if start_time <= t < end_time]
selected_times = [tha_epoch[i] for i in selected_indices]

# Prepare arrays for THEMIS and JUICE positions at selected times
themis_positions = np.array([[xa[i], ya[i], za[i]] for i in selected_indices])
themis_b_positions = np.array([[xb[i], yb[i], zb[i]] for i in selected_indices])
themis_c_positions = np.array([[xc[i], yc[i], zc[i]] for i in selected_indices])

# For each selected time, find the closest lp_epoch and get JUICE position
juice_positions = []
for t in selected_times:
    # Find index of closest lp_epoch
    idx = np.argmin([abs((t - lp_t).total_seconds()) for lp_t in lp_epochs[::downsample_factor]])
    juice_positions.append([x[idx], y[idx], z[idx]])
juice_positions = np.array(juice_positions)

# Compute distances (in Earth radii)
dist_tha_juice = np.linalg.norm(themis_positions - juice_positions, axis=1) / R_E
dist_thb_juice = np.linalg.norm(themis_b_positions - juice_positions, axis=1) / R_E
dist_thc_juice = np.linalg.norm(themis_c_positions - juice_positions, axis=1) / R_E

plt.figure(figsize=(12, 6))
plt.plot(selected_times, dist_tha_juice, label='THA-JUICE Distance', marker='o', markersize=2)
plt.plot(selected_times, dist_thb_juice, label='THB-JUICE Distance', marker='o', markersize=2)
plt.plot(selected_times, dist_thc_juice, label='THC-JUICE Distance', marker='o', markersize=2)
plt.title('Distance from THEMIS Probes to JUICE')
plt.xlabel('Time')
plt.ylabel('Distance (Earth radii)')
plt.xticks(rotation=45)
plt.legend()
plt.grid()

plt.show()
"""
distances = np.sqrt(x**2 + y**2 + z**2) / R_E  # in Earth radii

plt.figure(figsize=(10, 5))
plt.scatter(lp_epochs[::downsample_factor], distances, s=0.1)
plt.title('Distance from JUICE to Earth')
plt.xlabel('Epoch')
plt.ylabel('Distance (Earth radii)')
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()


# Draw Earth as a circle with radius 1 R_E
earth_circle = plt.Circle((0, 0), 1, color='blue', alpha=0.3, label='Earth')
plt.figure(figsize=(8, 8))
plt.gca().add_patch(earth_circle)
plt.scatter(x/R_E, y/R_E, s=0.1)
plt.xlabel('X (Earth radii)')
plt.ylabel('Y (Earth radii)')
plt.title('JUICE Position in GSM Coordinates')
plt.grid()
plt.axis('equal')
plt.legend()
"""

"""
# Define the time window
start_time = dt.datetime(2024, 8, 20, 19, 0, 0)
end_time = dt.datetime(2024, 8, 21, 0, 0, 0)

# Filter erg times within the window
erg_indices = [i for i, t in enumerate(erg_epoch) if start_time <= t < end_time]
erg_times = [erg_epoch[i] for i in erg_indices]
erg_positions = erg_pos[erg_indices]

# For each erg time, find the closest lp_epoch and get JUICE position
juice_positions = []
for t in erg_times:
    idx = np.argmin([abs((t - lp_t).total_seconds()) for lp_t in lp_epochs[::downsample_factor]])
    juice_positions.append(states[idx, :3])
    print(f"Getting JUICE position for time {t.isoformat()} at index {idx}")
juice_positions = np.array(juice_positions)

# Compute distances (in Earth radii)
dist_erg_juice = np.linalg.norm(erg_positions - juice_positions, axis=1) / R_E

# Optional: plot the distances
plt.figure(figsize=(10, 5))
plt.plot(erg_times, dist_erg_juice, marker='o', markersize=2)
plt.title('Distance from ERG to JUICE (19:00-00:00, Aug 20)')
plt.xlabel('Time')
plt.ylabel('Distance (Earth radii)')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()

plt.show()
"""