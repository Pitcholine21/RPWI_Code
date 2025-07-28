import cdflib
import numpy as np
import datetime as dt
import spiceypy as spice
import Ef_utilities as Ef
from multiprocessing import Pool

def exponential(t, carac_time):
    return np.exp(-(t / carac_time)**2)

def correct_for_spacecraft_charge(U123, U4, epochs, carac_time, weight_function=exponential):
    # carac_time is the characteristic time of the exponential in seconds
    # Calculate the average weighed by the exponential function given

    nb_epochs = len(epochs)

    correction_list = np.zeros(nb_epochs)

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
            correction_list[i] = mean_numerator / mean_denominator
        else:
            correction_list[i] = 0

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

    return correction_list

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
    return i, mean_numerator / mean_denominator if mean_denominator != 0 else 0

def correct_for_spacecraft_charge_parallel(U123, U4, epochs, carac_time, weight_function=exponential):
    # carac_time is the characteristic time of the exponential in seconds
    # Calculate the average weighed by the exponential function given

    args = [(U123, U4, epochs, carac_time, weight_function, i) for i in range(len(epochs))]

    with Pool(processes=4) as pool:
        results = pool.starmap(worker, args)
    # Unzip the results
    indices, corrections = zip(*results)
    # Create the corrected array
    correction_list = np.zeros(len(epochs))
    correction_list[list(indices)] = corrections

    return correction_list

def main():

    print("Running E-field plasma LP correction script")
    start = dt.datetime.now()

    spice.furnsh("SPICE/JUICE/kernels/mk/juice_ops.tm")
    spice.furnsh("SPICE/gsm_frame.tf")

    carac_time = 300
    downsample_factor = 50
    w_func = exponential

    start_time = dt.datetime(2024, 8, 20, 12, 0, 0)
    end_time = dt.datetime(2024, 8, 21, 12, 0, 0)

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

    U12uncal = U12[start_idx:end_idx:downsample_factor]
    U23uncal = U23[start_idx:end_idx:downsample_factor]
    U34uncal = U34[start_idx:end_idx:downsample_factor]
    U40 = U40[start_idx:end_idx:downsample_factor]
    lp_epochs = lp_epochs[start_idx:end_idx:downsample_factor]

    print("Downsampled and truncated LP data")

    U1uncal, U2uncal, U3uncal, U4 = Ef.multiply_lists_by_44matrix(U12uncal, U23uncal, U34uncal, U40, Ef.diff2volt)

    """
    # Uncomment this section to plot the uncalibrated potentials and the weight function
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

    mean_diff_U1 = np.nanmean(np.array(U4) - np.array(U1uncal))
    mean_diff_U2 = np.nanmean(np.array(U4) - np.array(U2uncal))
    mean_diff_U3 = np.nanmean(np.array(U4) - np.array(U3uncal))

    U1regcal = np.array([float(u) + float(mean_diff_U1) for u in U1uncal])
    U2regcal = np.array([float(u) + float(mean_diff_U2) for u in U2uncal])
    U3regcal = np.array([float(u) + float(mean_diff_U3) for u in U3uncal])

    print("Did regular calibration")

    corrections1 = correct_for_spacecraft_charge_parallel(U1uncal, U4, lp_epochs, carac_time, weight_function=w_func)
    corrections2 = correct_for_spacecraft_charge_parallel(U2uncal, U4, lp_epochs, carac_time, weight_function=w_func)
    corrections3 = correct_for_spacecraft_charge_parallel(U3uncal, U4, lp_epochs, carac_time, weight_function=w_func)

    U1cal = U1uncal + corrections1
    U2cal = U2uncal + corrections2
    U3cal = U3uncal + corrections3

    print("Did new calibration")

    U12cal, U23cal, U34cal, U40 = Ef.multiply_lists_by_44matrix(U1cal, U2cal, U3cal, U4, Ef.volt2diff)
    U12regcal, U23regcal, U34regcal, U40reg = Ef.multiply_lists_by_44matrix(U1regcal, U2regcal, U3regcal, U4, Ef.volt2diff)

    print("Went back to differentials")

    Ex, Ey, Ez = Ef.multiply_lists_by_33matrix(U12cal, U23cal, U34cal, Ef.volt2E)
    Exreg, Eyreg, Ezreg = Ef.multiply_lists_by_33matrix(U12regcal, U23regcal, U34regcal, Ef.volt2E)

    print("Got E field")

    # Write data to a txt file with relevant info in the filename
    output_filename = (
        f"Efield_{start_time.strftime('%Y%m%dT%H%M%S')}_"
        f"{end_time.strftime('%Y%m%dT%H%M%S')}_"
        f"downsample{downsample_factor}_"
        f"caractime{carac_time}.txt"
    )

    print(f"Writing E-field data to {output_filename}")

    header = (
        "Epoch\t"
        "U40\t"
        "U12uncal\tU23uncal\tU34uncal\t"
        "U1uncal\tU2uncal\tU3uncal\t"
        "Corrections1\tCorrections2\tCorrections3\t"
        "U1regcal\tU2regcal\tU3regcal\t"
        "U1cal\tU2cal\tU3cal\t"
        "U12regcal\tU23regcal\tU34regcal\t"
        "U12cal\tU23cal\tU34cal\t"
        "Ex\tEy\tEz\t"
        "Exreg\tEyreg\tEzreg\n"
    )

    # Set output directory (change this as needed)
    output_dir = "Olivier_RPWI/Plasmasphere_data_files/LP/"
    output_path = output_dir + output_filename

    with open(output_path, "w") as f:
        f.write(header)
        for i in range(len(lp_epochs)):
            line = (
                f"{lp_epochs[i]}"
                f"\t{U40[i]:.6f}"
                f"\t{U12uncal[i]:.6f}\t{U23uncal[i]:.6f}\t{U34uncal[i]:.6f}"
                f"\t{U1uncal[i]:.6f}\t{U2uncal[i]:.6f}\t{U3uncal[i]:.6f}"
                f"\t{corrections1[i]:.6f}\t{corrections2[i]:.6f}\t{corrections3[i]:.6f}"
                f"\t{U1regcal[i]:.6f}\t{U2regcal[i]:.6f}\t{U3regcal[i]:.6f}"
                f"\t{U1cal[i]:.6f}\t{U2cal[i]:.6f}\t{U3cal[i]:.6f}"
                f"\t{U12regcal[i]:.6f}\t{U23regcal[i]:.6f}\t{U34regcal[i]:.6f}"
                f"\t{U12cal[i]:.6f}\t{U23cal[i]:.6f}\t{U34cal[i]:.6f}"
                f"\t{Ex[i]:.6f}\t{Ey[i]:.6f}\t{Ez[i]:.6f}"
                f"\t{Exreg[i]:.6f}\t{Eyreg[i]:.6f}\t{Ezreg[i]:.6f}\n"
            )
            f.write(line)

    print("Wrote into data file")

    spice.kclear()

    print("Finished")
    print(f"Running time : {(dt.datetime.now()-start).total_seconds()} seconds")

if __name__ == "__main__":
    main()