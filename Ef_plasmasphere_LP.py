import cdflib
import numpy as np
import datetime as dt
import spiceypy as spice
import Ef_utilities as Ef
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from multiprocessing import Pool

def exponential(t, carac_time):
    return np.exp(-(t / carac_time)**2)

def box(t, carac_time):
    return 1

def correct_for_spacecraft_charge(U123, U4, epochs, carac_time, weight_function=exponential):
    # carac_time is the characteristic time of the exponential in seconds
    # Calculate the average weighed by the exponential function given

    nb_epochs = len(epochs)

    U123_corrected = np.zeros(nb_epochs)

    start_time = dt.datetime.now()

    for i in range(nb_epochs):
        curr_time = dt.datetime.now()
        epoch = epochs[i]
        mean_numerator = 0
        mean_denominator = 0
        seconds = (curr_time - start_time).total_seconds()
        remaining = seconds*(nb_epochs-i-1)/(i+1)
        print(f"Correcting for time {epoch} ({i+1}/{nb_epochs}), estimated time remaining at {remaining:.2f} seconds")
        for j in range(i,nb_epochs):
            time_delta = (epochs[j]-epoch).total_seconds()
            if time_delta >= 5*carac_time:
                #print(f"Breaking at {epochs[j]} in incremental loop")
                break
            if np.isfinite(U4[j]) and np.isfinite(U123[j]):
                weight = weight_function(time_delta, carac_time)
                mean_numerator += (U4[j] - U123[j]) * weight   
                mean_denominator += weight
        for j in range(i-1, -1, -1):
            time_delta = (epochs[j]-epoch).total_seconds()
            if time_delta <= -5*carac_time:
                #print(f"Breaking at {epochs[j]} in decremental loop")
                break
            if np.isfinite(U4[j]) and np.isfinite(U123[j]):
                weight = weight_function(time_delta, carac_time)
                mean_numerator += (U4[j] - U123[j]) * weight   
                mean_denominator += weight
        if mean_denominator != 0:
            U123_corrected[i] = U123[i] + mean_numerator / mean_denominator
        else:
            U123_corrected[i] = U123[i]

    """
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, remainings, label='Estimated time remaining')
    plt.xlabel('Timestamp')
    plt.ylabel('Estimated time remaining (seconds)')
    plt.title('Estimated Time Remaining for Correction')
    plt.legend()
    plt.grid()

    num_bins = 10

    timestamps_float = np.array([(ts - timestamps[0]).total_seconds() for ts in timestamps])
    bin_edges = np.linspace(timestamps_float.min(), timestamps_float.max(), num_bins + 1)
    bin_indices = np.digitize(timestamps_float, bin_edges) - 1  # bins: 0 to num_bins-1

    colors = plt.cm.viridis(np.linspace(0, 1, num_bins))

    # Create a stacked bar plot (histogram) with multicolored "bins
    plt.figure(figsize=(10, 5))
    num_bins_hist = 30
    # For each time bin, get the durations that fall into it
    stacked_counts = []
    bins = np.linspace(np.min(durations), np.max(durations), num_bins_hist + 1)
    for b in range(num_bins):
        mask = (bin_indices == b)
        counts, _ = np.histogram(np.array(durations)[mask], bins=bins)
        stacked_counts.append(counts)
    stacked_counts = np.array(stacked_counts)

    # Plot stacked bars
    bottom = np.zeros(num_bins_hist)
    for b in range(num_bins):
        plt.bar(
            0.5 * (bins[:-1] + bins[1:]),
            stacked_counts[b],
            width=(bins[1] - bins[0]),
            bottom=bottom,
            color=colors[b],
            edgecolor='black',
            label=f'Time bin {b+1}'
        )
        bottom += stacked_counts[b]

    plt.xlabel('Duration per iteration (seconds)')
    plt.ylabel('Count')
    plt.title('Stacked Histogram of Correction Durations per Iteration (by time bin)')
    plt.legend()
    plt.grid()
    """

    return U123_corrected

def worker(U123, U4, epochs, carac_time, weight_function, i):
    if i%100 == 0:
        print(f"Worker {i}/{len(epochs)} started")
    epoch = epochs[i]
    mean_numerator = 0
    mean_denominator = 0
    for j in range(i, len(epochs)):
        time_delta = (epochs[j] - epoch).total_seconds()
        if time_delta >= 5 * carac_time:
            break
        if np.isfinite(U4[j]) and np.isfinite(U123[j]):
            weight = weight_function(time_delta, carac_time)
            mean_numerator += (U4[j] - U123[j]) * weight   
            mean_denominator += weight
    for j in range(i-1, -1, -1):
        time_delta = (epochs[j] - epoch).total_seconds()
        if time_delta <= -5 * carac_time:
            break
        if np.isfinite(U4[j]) and np.isfinite(U123[j]):
            weight = weight_function(time_delta, carac_time)
            mean_numerator += (U4[j] - U123[j]) * weight   
            mean_denominator += weight
    return i, U123[i] + mean_numerator / mean_denominator if mean_denominator != 0 else U123[i]

def correct_for_spacecraft_charge_parallel(U123, U4, epochs, carac_time, weight_function=exponential):
    # carac_time is the characteristic time of the exponential in seconds
    # Calculate the average weighed by the exponential function given

    args = [(U123, U4, epochs, carac_time, weight_function, i) for i in range(len(epochs))]

    with Pool(processes=4) as pool:
        results = pool.starmap(worker, args)
    # Unzip the results
    indices, corrected_values = zip(*results)
    # Create the corrected array
    U123_corrected = np.zeros(len(epochs))
    U123_corrected[list(indices)] = corrected_values

    return U123_corrected

def main():

    print("Running E-field plasma LP correction script")

    spice.furnsh("SPICE/JUICE/kernels/mk/juice_ops.tm")
    spice.furnsh("SPICE/gsm_frame.tf")

    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.labelcolor'] = 'linecolor'
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['font.family'] = 'Serif'
    plt.ion()

    R_E = 6371  # Earth radius

    carac_time = 300
    downsample_factor = 200
    weight_function_used = "Exponential"
    w_func = exponential

    start_time = dt.datetime(2024, 8, 20, 12, 0, 0)
    end_time = dt.datetime(2024, 8, 21, 12, 15, 0)

    start = dt.datetime.now()

    # Get cdfs and JUICE state vectors

    cdfs = []

    print("Loading CDF files...")

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

    print(f"Got {len(cdfs)} CDF files")

    # Get E-field data

    U12 = np.array([])
    U23 = np.array([])
    U34 = np.array([])
    U40 = np.array([])
    lp_epochs = np.array([])

    for idx, cdf in enumerate(cdfs):
        lp_epoch = cdf.varget('Epoch')
        lp_epoch = cdflib.cdfepoch.to_datetime(lp_epoch)
        lp_epoch = np.array(lp_epoch, dtype='datetime64[ms]').astype('O')
        lp_epochs = np.concatenate((lp_epochs, lp_epoch))

        u12 = cdf.varget('LP_DATA')[:,0]
        u23 = cdf.varget('LP_DATA')[:,1]
        u34 = cdf.varget('LP_DATA')[:,2]
        u40 = cdf.varget('LP_DATA')[:,3]

        U12 = np.concatenate((U12, u12))
        U23 = np.concatenate((U23, u23))
        U34 = np.concatenate((U34, u34))
        U40 = np.concatenate((U40, u40))

    print("Got LP data")

    U12 = U12 * Ef.TM2diff[0]
    U23 = U23 * Ef.TM2diff[1]
    U34 = U34 * Ef.TM2diff[2]
    U40 = U40 * Ef.TM2diff[3]

    U12 = Ef.filter_out_start_config_noise(U12, lp_epochs)
    U23 = Ef.filter_out_start_config_noise(U23, lp_epochs)
    U34 = Ef.filter_out_start_config_noise(U34, lp_epochs)
    U40 = Ef.filter_out_start_config_noise(U40, lp_epochs)

    print("Got filtered LP data")

    # Get the indices for the epochs within the specified time range
    start_idx = np.searchsorted(lp_epochs, start_time)
    end_idx = np.searchsorted(lp_epochs, end_time)

    U12 = U12[start_idx:end_idx:downsample_factor]
    U23 = U23[start_idx:end_idx:downsample_factor]
    U34 = U34[start_idx:end_idx:downsample_factor]
    U40 = U40[start_idx:end_idx:downsample_factor]
    lp_epochs = lp_epochs[start_idx:end_idx:downsample_factor]

    print("Downsampled and truncated LP data")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(lp_epochs, U12, label='U12 (uncalibrated)')
    ax.plot(lp_epochs, U23, label='U23 (uncalibrated)')
    ax.plot(lp_epochs, U34, label='U34 (uncalibrated)')
    ax.plot(lp_epochs, U40, label='U40 (uncalibrated)')
    ax.set_title(f"Downsample factor: {downsample_factor}")
    ax.set_xlabel('Epoch \n Distance (R$_E$)')
    ax.xaxis.set_label_coords(-0.04, -0.02)
    ticks = ax.get_xticks()
    ticks = Ef.convert_1970(ticks)
    distances = Ef.get_JUICE_distance(ticks)
    for tick, dist in zip(ticks, distances):
        ax.annotate(f"{dist:.1f}", xy=(tick, ax.get_ylim()[0]), xycoords=('data', 'data'),
                    xytext=(0, -20), textcoords='offset points',
                    ha='center', va='top', fontsize=10, rotation=0)
    ax.set_ylabel('Voltage difference (V)')
    ax.legend()
    ax.grid()

    U1, U2, U3, U4 = Ef.multiply_lists_by_44matrix(U12, U23, U34, U40, Ef.diff2volt)

    """
    # Plot U1 and U4 over the entire time period
    center_time = dt.datetime(2024, 8, 20, 22, 0, 0)
    # Prepare exponential weights centered at center_time
    exp_weights = exponential(np.array([(dt.datetime.combine(lp_epochs[i].date(), lp_epochs[i].time()) - center_time).total_seconds() for i in range(len(lp_epochs))]), carac_time)
    plt.figure(figsize=(10, 5))

    # Plot U1*exp and U4*exp
    plt.plot(lp_epochs, U1 * exp_weights, label='U1 * weight function(t)', color='tab:blue')
    plt.plot(lp_epochs, U4 * exp_weights, label='U4 * weight function(t)', color='tab:orange')
    plt.plot(lp_epochs, U1, label='U1', alpha=0.5,color='tab:blue')
    plt.plot(lp_epochs, U4, label='U4', alpha=0.5,color='tab:orange')
    plt.plot(lp_epochs, exp_weights, label='Weight function(t)', linestyle='--', color='red')
    plt.title('U1 and U4 weighted')
    plt.xlabel('Epoch')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.grid()

    plt.tight_layout()

    # Calculate and print average of (U4 - U1) weighted by exponential(t)
    weighted_diff = (U4 - U1) * exp_weights
    avg_weighted_diff = np.nansum(weighted_diff) / np.nansum(exp_weights)
    axs[1].text(0.01, 0.90, f'Weighted average of U4-U1: {avg_weighted_diff:.4f} V', transform=axs[1].transAxes, fontsize=12, verticalalignment='top')

    print(f"Weighted average of U4-U1: {avg_weighted_diff:.4f} V")

    plt.show()
    """

    print("Got single probe potentials")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(lp_epochs, U1, label='U1 (uncalibrated)')
    ax.plot(lp_epochs, U2, label='U2 (uncalibrated)')
    ax.plot(lp_epochs, U3, label='U3 (uncalibrated)')
    ax.plot(lp_epochs, U4, label='U4 (uncalibrated)')
    ax.set_title(f"Caracteristic time: {carac_time} seconds, downsample factor: {downsample_factor}")
    ax.set_xlabel('Epoch \n Distance (R$_E$)')
    ax.xaxis.set_label_coords(-0.04, -0.02)
    ticks = ax.get_xticks()
    ticks = Ef.convert_1970(ticks)
    distances = Ef.get_JUICE_distance(ticks)
    for tick, dist in zip(ticks, distances):
        ax.annotate(f"{dist:.1f}", xy=(tick, ax.get_ylim()[0]), xycoords=('data', 'data'),
                    xytext=(0, -20), textcoords='offset points',
                    ha='center', va='top', fontsize=10, rotation=0)
    ax.set_ylabel('Voltage (V)')
    ax.legend()
    ax.grid()

    mean_diff_U1 = np.nanmean(np.array(U4) - np.array(U1))
    mean_diff_U2 = np.nanmean(np.array(U4) - np.array(U2))
    mean_diff_U3 = np.nanmean(np.array(U4) - np.array(U3))

    U1reg = np.array([float(u) + float(mean_diff_U1) for u in U1])
    U2reg = np.array([float(u) + float(mean_diff_U2) for u in U2])
    U3reg = np.array([float(u) + float(mean_diff_U3) for u in U3])


    U1 = correct_for_spacecraft_charge_parallel(U1, U4, lp_epochs, carac_time, weight_function=w_func)
    U2 = correct_for_spacecraft_charge_parallel(U2, U4, lp_epochs, carac_time, weight_function=w_func)
    U3 = correct_for_spacecraft_charge_parallel(U3, U4, lp_epochs, carac_time, weight_function=w_func)

    print("Corrected for spacecraft charge")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(lp_epochs, U1, label='U1 (sliding cal)')
    ax.plot(lp_epochs, U2, label='U2 (sliding cal)')
    ax.plot(lp_epochs, U3, label='U3 (sliding cal)')
    ax.plot(lp_epochs, U4, label='U4 (sliding cal)')
    ax.set_title(f"Caracteristic time: {carac_time} seconds, downsample factor: {downsample_factor}, weight function: {weight_function_used}")
    ax.set_xlabel('Epoch \n Distance (R$_E$)')
    ax.xaxis.set_label_coords(-0.04, -0.02)
    ticks = ax.get_xticks()
    ticks = Ef.convert_1970(ticks)
    distances = Ef.get_JUICE_distance(ticks)
    for tick, dist in zip(ticks, distances):
        ax.annotate(f"{dist:.1f}", xy=(tick, ax.get_ylim()[0]), xycoords=('data', 'data'),
                    xytext=(0, -20), textcoords='offset points',
                    ha='center', va='top', fontsize=10, rotation=0)
    ax.set_ylabel('Voltage (V)')
    ax.legend()
    ax.grid()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(lp_epochs, U1reg, label='U1 (regular cal)')
    ax.plot(lp_epochs, U2reg, label='U2 (regular cal)')
    ax.plot(lp_epochs, U3reg, label='U3 (regular cal)')
    ax.plot(lp_epochs, U4, label='U4 (regular cal)')
    ax.set_title(f"Downsample factor: {downsample_factor}")
    ax.set_xlabel('Epoch \n Distance (R$_E$)')
    ax.xaxis.set_label_coords(-0.04, -0.02)
    ticks = ax.get_xticks()
    ticks = Ef.convert_1970(ticks)
    distances = Ef.get_JUICE_distance(ticks)
    for tick, dist in zip(ticks, distances):
        ax.annotate(f"{dist:.1f}", xy=(tick, ax.get_ylim()[0]), xycoords=('data', 'data'),
                    xytext=(0, -20), textcoords='offset points',
                    ha='center', va='top', fontsize=10, rotation=0)
    ax.set_ylabel('Voltage (V)')
    ax.legend()
    ax.grid()

    U12, U23, U34, U40 = Ef.multiply_lists_by_44matrix(U1, U2, U3, U4, Ef.volt2diff)
    U12reg, U23reg, U34reg, U40reg = Ef.multiply_lists_by_44matrix(U1reg, U2reg, U3reg, U4, Ef.volt2diff)

    print("Went back to differentials")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(lp_epochs, U12, label='U12 (sliding cal)')
    ax.plot(lp_epochs, U23, label='U23 (sliding cal)')
    ax.plot(lp_epochs, U34, label='U34 (sliding cal)')
    ax.plot(lp_epochs, U40, label='U40 (sliding cal)')
    ax.set_title(f"Caracteristic time: {carac_time} seconds, downsample factor: {downsample_factor}, weight function: {weight_function_used}")
    ax.set_xlabel('Epoch \n Distance (R$_E$)')
    ax.xaxis.set_label_coords(-0.04, -0.02)
    ticks = ax.get_xticks()
    ticks = Ef.convert_1970(ticks)
    distances = Ef.get_JUICE_distance(ticks)
    for tick, dist in zip(ticks, distances):
        ax.annotate(f"{dist:.1f}", xy=(tick, ax.get_ylim()[0]), xycoords=('data', 'data'),
                    xytext=(0, -20), textcoords='offset points',
                    ha='center', va='top', fontsize=10, rotation=0)
    ax.set_ylabel('Voltage difference (V)')
    ax.legend()
    ax.grid()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(lp_epochs, U12reg, label='U12 (regular cal)')
    ax.plot(lp_epochs, U23reg, label='U23 (regular cal)')
    ax.plot(lp_epochs, U34reg, label='U34 (regular cal)')
    ax.plot(lp_epochs, U40reg, label='U40 (regular cal)')
    ax.set_title(f"Downsample factor: {downsample_factor}")
    ax.set_xlabel('Epoch \n Distance (R$_E$)')
    ax.xaxis.set_label_coords(-0.04, -0.02)
    ticks = ax.get_xticks()
    ticks = Ef.convert_1970(ticks)
    distances = Ef.get_JUICE_distance(ticks)
    for tick, dist in zip(ticks, distances):
        ax.annotate(f"{dist:.1f}", xy=(tick, ax.get_ylim()[0]), xycoords=('data', 'data'),
                    xytext=(0, -20), textcoords='offset points',
                    ha='center', va='top', fontsize=10, rotation=0)
    ax.set_ylabel('Voltage difference (V)')
    ax.legend()
    ax.grid()

    Ex, Ey, Ez = Ef.multiply_lists_by_33matrix(U12, U23, U34, Ef.volt2E)
    Emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    Exreg, Eyreg, Ezreg = Ef.multiply_lists_by_33matrix(U12reg, U23reg, U34reg, Ef.volt2E)
    Emagreg = np.sqrt(Exreg**2 + Eyreg**2 + Ezreg**2)

    print("Got E field")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(lp_epochs, Ex*1e3, label='Ex (sliding cal)', lw=1.1)
    ax.plot(lp_epochs, Ey*1e3, label='Ey (sliding cal)', lw=1.1)
    ax.plot(lp_epochs, Ez*1e3, label='Ez (sliding cal)', lw=1.1)
    ax.plot(lp_epochs, Emag*1e3, label='|E| (sliding cal)', color = 'red', lw=1.1)
    ax.set_title(f"Caracteristic time: {carac_time} seconds, downsample factor: {downsample_factor}, weight function: {weight_function_used}")
    ax.set_xlabel('Epoch \n Distance (R$_E$)')
    ax.xaxis.set_label_coords(-0.04, -0.02)
    ticks = ax.get_xticks()
    ticks = Ef.convert_1970(ticks)
    distances = Ef.get_JUICE_distance(ticks)
    for tick, dist in zip(ticks, distances):
        ax.annotate(f"{dist:.1f}", xy=(tick, ax.get_ylim()[0]), xycoords=('data', 'data'),
                    xytext=(0, -20), textcoords='offset points',
                    ha='center', va='top', fontsize=10, rotation=0)
    ax.set_ylabel('Electric Field (mV/m)')
    ax.legend()
    ax.grid()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(lp_epochs, Exreg*1e3, label='Ex (regular cal)', lw=1.1)
    ax.plot(lp_epochs, Eyreg*1e3, label='Ey (regular cal)', lw=1.1)
    ax.plot(lp_epochs, Ezreg*1e3, label='Ez (regular cal)', lw=1.1)
    ax.plot(lp_epochs, Emagreg*1e3, label='|E| (regular cal)', color = 'red', lw=1.1)
    ax.set_title(f"Downsample factor: {downsample_factor}")
    ax.set_xlabel('Epoch \n Distance (R$_E$)')
    ax.xaxis.set_label_coords(-0.04, -0.02)
    ticks = ax.get_xticks()
    ticks = Ef.convert_1970(ticks)
    distances = Ef.get_JUICE_distance(ticks)
    for tick, dist in zip(ticks, distances):
        ax.annotate(f"{dist:.1f}", xy=(tick, ax.get_ylim()[0]), xycoords=('data', 'data'),
                    xytext=(0, -20), textcoords='offset points',
                    ha='center', va='top', fontsize=10, rotation=0)
    ax.set_ylabel('Electric Field (mV/m)')
    ax.legend()
    ax.grid()

    # Write E-field data to a txt file with relevant info in the filename
    output_filename = (
        f"Efield_{start_time.strftime('%Y%m%dT%H%M%S')}_"
        f"{end_time.strftime('%Y%m%dT%H%M%S')}_"
        f"downsample{downsample_factor}_"
        f"caractime{carac_time}_"
        f"{weight_function_used}.txt"
    )

    print(f"Writing E-field data to {output_filename}")

    header = (
        "Epoch\t"
        "Ex(sliding)\tEy(sliding)\tEz(sliding)\tEmag(sliding)\t"
        "Ex(regular)\tEy(regular)\tEz(regular)\tEmag(regular)\n"
    )

    with open(output_filename, "w") as f:
        f.write(header)
        for i in range(len(lp_epochs)):
            line = (
                f"{lp_epochs[i]}\t"
                f"{Ex[i]:.6e}\t{Ey[i]:.6e}\t{Ez[i]:.6e}\t{Emag[i]:.6e}\t"
                f"{Exreg[i]:.6e}\t{Eyreg[i]:.6e}\t{Ezreg[i]:.6e}\t{Emagreg[i]:.6e}\n"
            )
            f.write(line)

    spice.kclear()

    print("Finished")
    print(f"Running time : {(dt.datetime.now()-start).total_seconds()} seconds")
    plt.show(block = False)  # Show the plot without blocking the script
    input("Press Enter to exit...")  # Keep the plot window open until user closes it
    plt.close('all')  # Close all plot windows after user input

if __name__ == "__main__":
    main()