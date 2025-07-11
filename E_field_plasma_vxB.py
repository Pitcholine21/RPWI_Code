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

plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.labelcolor'] = 'linecolor'
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['font.family'] = 'Serif'

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
    x = states[i,0]
    y = states[i,1]
    z = states[i,2]
    bx, by, bz = geopack.igrf_gsm(x/R_E, y/R_E, z/R_E)
    Bx.append(bx)
    By.append(by)
    Bz.append(bz)

Bmag = np.sqrt(np.array(Bx)**2 + np.array(By)**2 + np.array(Bz)**2)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(lp_epochs[::downsample_factor], Bx, label='Bx')
ax.plot(lp_epochs[::downsample_factor], By, label='By')
ax.plot(lp_epochs[::downsample_factor], Bz, label='Bz')
ax.plot(lp_epochs[::downsample_factor], Bmag, label='|B|')
ax.set_xlabel('Epoch \n Distance (R$_E$)')
ax.xaxis.set_label_coords(-0.06, -0.025)
ticks = ax.get_xticks()
ticks = convert_1970(ticks)
distances = get_JUICE_distance(ticks)
for tick, dist in zip(ticks, distances):
    ax.annotate(f"{dist:.1f}", xy=(tick, ax.get_ylim()[0]), xycoords=('data', 'data'),
                xytext=(0, -20), textcoords='offset points',
                ha='center', va='top', fontsize=10, rotation=0)
ax.set_ylabel('Magnetic Field (nT)')
ax.set_title('Magnetic Field Components at JUICE in GSM frame')
ax.legend()
ax.grid()

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

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(lp_epochs[::downsample_factor], states[:, 3], label='vx from spacecraft', linestyle=':', color='tab:blue')
ax.plot(lp_epochs[::downsample_factor], states[:, 4], label='vy from spacecraft', linestyle=':', color='tab:orange')
ax.plot(lp_epochs[::downsample_factor], -corotation_v[:, 0], label='vx from corotation', linestyle='--', color='tab:blue')
ax.plot(lp_epochs[::downsample_factor], -corotation_v[:, 1], label='vy from corotation', linestyle='--', color='tab:orange')
ax.plot(lp_epochs[::downsample_factor], plasma_rel_speed[:, 0], label='vx', color='tab:blue')
ax.plot(lp_epochs[::downsample_factor], plasma_rel_speed[:, 1], label='vy', color='tab:orange')
ax.plot(lp_epochs[::downsample_factor], plasma_rel_speed[:, 2], label='vz', color='tab:green')
ax.set_xlabel('Epoch \n Distance (R$_E$)')
ax.xaxis.set_label_coords(-0.06, -0.025)
ticks = ax.get_xticks()
ticks = convert_1970(ticks)
distances = get_JUICE_distance(ticks)
for tick, dist in zip(ticks, distances):
    ax.annotate(f"{dist:.1f}", xy=(tick, ax.get_ylim()[0]), xycoords=('data', 'data'),
                xytext=(0, -20), textcoords='offset points',
                ha='center', va='top', fontsize=10, rotation=0)
ax.set_ylabel('Velocity (km/s)')
ax.set_title('JUICE Spacecraft Velocity Components in GSM frame')
ax.legend(loc = 'upper left')
ax.grid()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(lp_epochs[::downsample_factor], v_x_B[:, 0]*1e-3, label='-vxB_x')
ax.plot(lp_epochs[::downsample_factor], v_x_B[:, 1]*1e-3, label='-vxB_y')
ax.plot(lp_epochs[::downsample_factor], v_x_B[:, 2]*1e-3, label='-vxB_z')
ax.set_xlabel('Epoch \n Distance (R$_E$)')
ax.xaxis.set_label_coords(-0.06, -0.025)
ticks = ax.get_xticks()
ticks = convert_1970(ticks)
distances = get_JUICE_distance(ticks)
for tick, dist in zip(ticks, distances):
    ax.annotate(f"{dist:.1f}", xy=(tick, ax.get_ylim()[0]), xycoords=('data', 'data'),
                xytext=(0, -20), textcoords='offset points',
                ha='center', va='top', fontsize=10, rotation=0)
ax.set_ylabel('Electric Field (mV/m)')
ax.set_title('Electric Field from -v x B at JUICE in GSM frame')
ax.legend()
ax.grid()

for i in range(len(lp_epochs[::downsample_factor])):
    et = spice.str2et(lp_epochs[::downsample_factor][i].isoformat())
    v_x_B[i] = np.dot(spice.pxform('GSM', 'JUICE_SPACECRAFT', et), v_x_B[i])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(lp_epochs[::downsample_factor], v_x_B[:, 0]*1e-3, label='-vxB_x')
ax.plot(lp_epochs[::downsample_factor], v_x_B[:, 1]*1e-3, label='-vxB_y')
ax.plot(lp_epochs[::downsample_factor], v_x_B[:, 2]*1e-3, label='-vxB_z')
ax.set_xlabel('Epoch \n Distance (R$_E$)')
ax.xaxis.set_label_coords(-0.06, -0.025)
ticks = ax.get_xticks()
ticks = convert_1970(ticks)
distances = get_JUICE_distance(ticks)
for tick, dist in zip(ticks, distances):
    ax.annotate(f"{dist:.1f}", xy=(tick, ax.get_ylim()[0]), xycoords=('data', 'data'),
                xytext=(0, -20), textcoords='offset points',
                ha='center', va='top', fontsize=10, rotation=0)
ax.set_ylabel('Electric Field (mV/m)')
ax.set_title('Electric Field from -v x B at JUICE in JUICE spacecraft frame')
ax.legend()
ax.grid()

print(f"Running time: {dt.datetime.now() - start}")

spice.kclear()

plt.show()