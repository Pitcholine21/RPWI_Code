import cdflib
import numpy as np
import datetime as dt
import spiceypy as spice
import matplotlib.pyplot as plt
from geopack import geopack
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d
from Ef_utilities import *
from utilities import print_entire

R_E = 6371  # Earth radius

# Get current time in UTC
start = dt.datetime.now()

# Get cdfs and JUICE state vectors

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

downsample_factor = 300  # Downsample factor for epochs

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
    et = spice.str2et(epoch.isoformat())
    x = states[i,0]
    y = states[i,1]
    z = states[i,2]
    bx, by, bz = geopack.igrf_gsm(x/R_E, y/R_E, z/R_E)
    Bx.append(bx)
    By.append(by)
    Bz.append(bz)

Bmag = np.sqrt(np.array(Bx)**2 + np.array(By)**2 + np.array(Bz)**2)

plt.figure(figsize=(10, 5))
plt.plot(lp_epochs[::downsample_factor], Bx, label='Bx')
plt.plot(lp_epochs[::downsample_factor], By, label='By')
plt.plot(lp_epochs[::downsample_factor], Bz, label='Bz')
plt.plot(lp_epochs[::downsample_factor], Bmag, label='|B|')
plt.xlabel('Time')
plt.ylabel('Magnetic Field (nT)')
plt.title('Magnetic Field Components at JUICE')
plt.legend()
plt.grid()


# Calculating - v x B

v_x_B = np.zeros((len(lp_epochs[::downsample_factor]), 3))
for i in range(len(lp_epochs[::downsample_factor])):
    print(f"Calculating -v x B for epoch {i+1}/{len(lp_epochs[::downsample_factor])}: {lp_epochs[::downsample_factor][i].isoformat()}")
    v_x_B[i] = -np.cross(states[i, 3:6], [Bx[i], By[i], Bz[i]])
    et = spice.str2et(lp_epochs[::downsample_factor][i].isoformat())
    v_x_B[i] = np.dot(spice.pxform('GSM', 'JUICE_SPACECRAFT', et), v_x_B[i])

plt.figure(figsize=(10, 5))
plt.plot(lp_epochs[::downsample_factor], states[:, 3], label='vx')
plt.plot(lp_epochs[::downsample_factor], states[:, 4], label='vy')
plt.plot(lp_epochs[::downsample_factor], states[:, 5], label='vz')
plt.xlabel('Time')
plt.ylabel('Velocity (km/s)')
plt.title('JUICE Spacecraft Velocity Components in GSM frame')
plt.legend()
plt.grid()

plt.figure(figsize=(10, 5))
plt.plot(lp_epochs[::downsample_factor], v_x_B[:, 0]*1e-3, label='-vxB_x')
plt.plot(lp_epochs[::downsample_factor], v_x_B[:, 1]*1e-3, label='-vxB_y')
plt.plot(lp_epochs[::downsample_factor], v_x_B[:, 2]*1e-3, label='-vxB_z')
plt.xlabel('Time')
plt.ylabel('Electric Field (mV/m)')
plt.title('Electric Field from -v x B at JUICE in JUICE spacecraft frame')
plt.legend()
plt.grid()
plt.show()

print(f"Running time: {dt.datetime.now() - start}")

spice.kclear()