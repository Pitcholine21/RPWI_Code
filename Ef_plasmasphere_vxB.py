import cdflib
import numpy as np
import datetime as dt
import spiceypy as spice
import matplotlib.pyplot as plt
from geopack import geopack
from Ef_utilities import *

# Get current time in UTC
start = dt.datetime.now()

spice.furnsh("SPICE/JUICE/kernels/mk/juice_ops.tm")
spice.furnsh("SPICE/gsm_frame.tf")

R_E = 6371  # Earth radius

downsample_factor = 100

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

lp_epochs = np.array([])

for cdf in cdfs:
    lp_epoch = cdf.varget('Epoch')
    lp_epoch = cdflib.cdfepoch.to_datetime(lp_epoch)
    lp_epoch = np.array(lp_epoch, dtype='datetime64[ms]').astype('O')
    lp_epochs = np.concatenate((lp_epochs, lp_epoch))

states = np.zeros((len(lp_epochs[::downsample_factor]), 6))

for i, epoch in enumerate(lp_epochs[::downsample_factor]):
    print(f"Gettting JUICE state for epoch {i+1}/{len(lp_epochs[::downsample_factor])}: {epoch.isoformat()}")
    et = spice.str2et(epoch.isoformat())
    state, _ = spice.spkezr('JUICE', et, 'GSM', 'NONE', 'EARTH')
    states[i] = state

# Getting B-field

By, Bx, Bz = [], [], []

for i, epoch in enumerate(lp_epochs[::downsample_factor]):
    # Get time in seconds since 1970-01-01T00:00:00
    print(f"Calculating B for epoch {i+1}/{len(lp_epochs[::downsample_factor])}: {lp_epochs[::downsample_factor][i].isoformat()}")
    unix = (epoch - dt.datetime(1970, 1, 1)).total_seconds()
    geopack.recalc(unix)
    x = states[i,0]
    y = states[i,1]
    z = states[i,2]
    bx, by, bz = geopack.igrf_gsm(x/R_E, y/R_E, z/R_E)
    Bx.append(bx)
    By.append(by)
    Bz.append(bz)

# Get corotation speed in GSM frame
omega_earth = 7.2921159e-5  # Earth's rotation rate in rad/s

corotation_v = np.zeros_like(states[:, 3:6])
for i in range(len(states)):
    x, y, z = states[i, 0:3]
    lat = np.arctan2(z, np.sqrt(x**2 + y**2))  # in radians
    r = np.sqrt(x**2 + y**2)
    # Unit vector in e_theta direction (z out of plane)
    e_theta = np.array([-y, x, 0]) / r
    v_coro = np.cos(lat) * omega_earth * np.exp(-r/(4*R_E)) * r * e_theta  # km/s
    corotation_v[i] = v_coro

# Subtract corotation speed from state vector velocities
plasma_rel_speed = states[:, 3:6] - corotation_v

# Calculating - v x B
v_x_B = np.zeros((len(lp_epochs[::downsample_factor]), 3))
for i in range(len(lp_epochs[::downsample_factor])):
    print(f"Calculating -v x B for epoch {i+1}/{len(lp_epochs[::downsample_factor])}: {lp_epochs[::downsample_factor][i].isoformat()}")
    v_x_B[i] = -np.cross(plasma_rel_speed[i], [Bx[i], By[i], Bz[i]])

# Write data to a txt file with relevant info in the filename
output_filename = f"vxB_downsample{downsample_factor}.txt"

print(f"Writing vxB data to {output_filename}")

header = (
    "Epoch\t"
    "Bx\tBy\tBz\t"
    "vxSC\tvySC\tvzSC\t"
    "vxCorot\tvyCorot\t"
    "EvxBx\tEvxBy\tEvxBz\n"
)

# Set output directory (change this as needed)
output_dir = "Olivier_RPWI/Plasmasphere_data_files/vxB/"
output_path = output_dir + output_filename

with open(output_path, "w") as f:
    f.write(header)
    for i in range(len(lp_epochs[::downsample_factor])):
        line = (
            f"{lp_epochs[::downsample_factor][i]}"
            f"\t{Bx[i]:.6f}\t{By[i]:.6f}\t{Bz[i]:.6f}"
            f"\t{states[i,3]:.6f}\t{states[i,4]:.6f}\t{states[i,5]:.6f}"
            f"\t{-corotation_v[i,0]:.6f}\t{-corotation_v[i,1]:.6f}"
            f"\t{v_x_B[i,0]*1e-3:.6f}\t{v_x_B[i,1]*1e-3:.6f}\t{v_x_B[i,2]*1e-3:.6f}\n"
        )
        f.write(line)

print("Wrote into data file")

print("Finished")
print(f"Running time: {dt.datetime.now() - start}")

spice.kclear()

plt.show()