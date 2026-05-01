import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CALB L148N58A Real Experimental Data Preprocessor
# Paper 2 -- Real dataset validation
#
# Input:  .mat files from Mendeley dataset
#         battery_data/Temperature_10C/DV_WLTP/*.mat  etc.
# Output: calb_data/calb_train.csv
#         calb_data/calb_val.csv
#         calb_data/calb_test.csv
#         calb_data/surface_validation.png  (key figure for paper)
# ============================================================

OUTPUT_DIR = "calb_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CALB L148N58A cell parameters
CAPACITY_AH = 58.0
V_MAX       = 4.2
V_MIN       = 2.5

# Two-node thermal model parameters
# These are the same parameters used in Paper 1
# Validated by comparing predicted T_surface vs real T_surface
MCP_CORE = 0.65 * 1000.0   # J/K  core thermal mass
MCP_SURF = 0.28 * 1000.0   # J/K  surface thermal mass
K_COND   = 1.2              # W/K  core to surface conductance
SURF_AREA = 0.052           # m2   cell surface area
H_CONV   = 12.0             # W/m2.K  convection coefficient
HA       = H_CONV * SURF_AREA  # W/K

# Hotspot threshold -- from CALB datasheet max operating temperature
HOTSPOT_THRESH = 44.0       # C
DANGER_THRESH  = 55.0       # C

# Folder structure
DATA_ROOT = r"C:\Users\yuvav\battery_data"

TEMP_FOLDERS = {
    "10C":  os.path.join(DATA_ROOT, "Temperature_10C"),
    "25C":  os.path.join(DATA_ROOT, "Temperature_25C"),
    "40C":  os.path.join(DATA_ROOT, "Temperature_40C"),
}

PROFILE_FOLDERS = {
    "DV_WLTP":      "WLTP",
    "DV_UDDS":      "UDDS",
    "DV_US06":      "US06",
    "C20_charge":   "C20_charge",
    "C20_discharge":"C20_discharge",
}


# ============================================================
# LOAD ONE MAT FILE
# ============================================================

def load_mat_file(filepath):
    """
    Load one .mat file and return a clean DataFrame.
    Handles the nested MATLAB structure format.
    """
    mat  = sio.loadmat(filepath)
    data = mat["Data"][0, 0]

    # Extract each column from the nested structure
    times    = data["Times"].flatten().astype(np.float64)
    voltage  = data["VoltageV"].flatten().astype(np.float64)
    current  = data["CurrentA"].flatten().astype(np.float64)
    t_surf   = data["TempC"].flatten().astype(np.float64)
    t_amb    = data["Temp_Amb"].flatten().astype(np.float64)

    df = pd.DataFrame({
        "time_s"      : times,
        "voltage_V"   : voltage,
        "current_A"   : current,
        "T_surface_real" : t_surf,   # real thermocouple -- for validation
        "T_ambient_C" : t_amb,
    })

    # Remove duplicate timestamps and sort
    df = df.drop_duplicates(subset="time_s").sort_values("time_s").reset_index(drop=True)

    return df


# ============================================================
# RESAMPLE TO 1-SECOND INTERVALS
# ============================================================

def resample_to_1s(df):
    """
    The .mat files have irregular sampling (roughly 0.1s).
    Resample to clean 1-second intervals for the LSTM.
    """
    t_orig = df["time_s"].values
    t_new  = np.arange(0, t_orig[-1], 1.0, dtype=np.float64)

    resampled = {}
    for col in ["voltage_V", "current_A", "T_surface_real", "T_ambient_C"]:
        f = interp1d(t_orig, df[col].values,
                     kind="linear", fill_value="extrapolate")
        resampled[col] = f(t_new)

    resampled["time_s"] = t_new
    return pd.DataFrame(resampled)


# ============================================================
# COMPUTE SOC VIA COULOMB COUNTING
# ============================================================

def compute_soc(current_A, dt=1.0, soc_init=None):
    """
    Coulomb counting from real current measurements.
    Positive current = discharge (SOC decreases).
    Negative current = charge (SOC increases).
    soc_init: if None, estimate from starting voltage.
    """
    if soc_init is None:
        soc_init = 0.80   # reasonable default

    cap_C = CAPACITY_AH * 3600.0
    soc   = np.zeros(len(current_A), dtype=np.float64)
    soc[0] = float(np.clip(soc_init, 0.0, 1.0))

    for k in range(len(current_A) - 1):
        dsoc   = -float(current_A[k]) * dt / cap_C
        soc[k+1] = float(np.clip(soc[k] + dsoc, 0.0, 1.0))

    return soc


# ============================================================
# TWO-NODE THERMAL MODEL
# Computes T_internal from real current and real T_surface.
# This is the key step -- using measured data to derive T_internal.
# ============================================================

def compute_t_internal(current_A, T_surface_real, T_ambient_C, SOC,
                        dt=1.0):
    """
    Derives T_internal using the inverse two-node thermal model.

    Standard approach in battery thermal literature.
    Reference: Forgez et al. (2010), Journal of Power Sources.

    The two-node equations are:
      mCp_core * dTcore/dt = Q_gen - k*(Tcore - Tsurf)
      mCp_surf * dTsurf/dt = k*(Tcore - Tsurf) - hA*(Tsurf - Tamb)

    We KNOW:
      - Q_gen  (from real current and estimated resistance)
      - T_surf (from real thermocouple)
      - T_amb  (from real ambient sensor)

    We COMPUTE:
      - T_core by integrating the core ODE forward in time

    Then we validate by checking our predicted T_surf
    against the real measured T_surf.
    """
    n      = len(current_A)
    T_core = np.zeros(n, dtype=np.float64)
    Q_gen  = np.zeros(n, dtype=np.float64)

    # Start core temperature at same as surface
    T_core[0] = float(T_surface_real[0])

    for k in range(n - 1):
        I   = float(current_A[k])
        S   = float(np.clip(SOC[k], 0.05, 1.0))
        Ta  = float(T_ambient_C[k])
        Ts  = float(T_surface_real[k])   # real measured surface
        Tc  = T_core[k]

        # Internal resistance -- SOC dependent
        R = (0.0008 +
             0.0005 * (1.0 - S) +
             0.0003 * np.exp(-10.0 * (S - 0.05)))

        # Heat generation from real current
        Q_gen[k] = I * I * R

        # Core temperature ODE (forward Euler)
        dTc = (Q_gen[k] - K_COND * (Tc - Ts)) / MCP_CORE
        T_core[k+1] = Tc + dTc * dt

    # Fill last step
    S_last    = float(np.clip(SOC[-1], 0.05, 1.0))
    R_last    = (0.0008 + 0.0005*(1.0-S_last) +
                 0.0003*np.exp(-10.0*(S_last-0.05)))
    Q_gen[-1] = float(current_A[-1])**2 * R_last

    return T_core, Q_gen


# ============================================================
# PREDICT T_SURFACE FROM T_CORE (for validation)
# ============================================================

def predict_t_surface(T_core, T_ambient_C, dt=1.0):
    """
    Given T_core and T_ambient, predict what T_surface should be.
    We then compare this against the real measured T_surface.
    If they match -- our model is validated.
    """
    n      = len(T_core)
    T_surf = np.zeros(n, dtype=np.float64)
    T_surf[0] = float(T_core[0])

    for k in range(n - 1):
        Ta  = float(T_ambient_C[k])
        Tc  = float(T_core[k])
        Ts  = T_surf[k]
        dTs = (K_COND * (Tc - Ts) - HA * (Ts - Ta)) / MCP_SURF
        T_surf[k+1] = Ts + dTs * dt

    return T_surf


# ============================================================
# ENGINEER FEATURES
# ============================================================

def engineer_features(df, run_id, profile, T_amb_label):
    """
    Create all 12 LSTM input features from the real measurements.
    Same features as Paper 1 so the same model architecture works.
    """
    t       = df["time_s"].values
    I       = df["current_A"].values
    V       = df["voltage_V"].values
    T_surf  = df["T_surface_real"].values
    T_amb   = df["T_ambient_C"].values
    SOC     = df["SOC"].values
    Q_gen   = df["Q_gen_W"].values
    T_core  = df["T_internal_C"].values

    # Internal resistance estimate
    R_int = np.clip(
        np.abs(3.7 + 0.5*SOC - V) / np.clip(np.abs(I), 0.5, 999.0),
        0.0001, 0.05
    )

    # Rate of change features
    dI_dt = np.gradient(I, t)
    dV_dt = np.gradient(V, t)

    # Delta temperatures
    delta_T_surf = T_surf - T_amb
    delta_T_core = T_core - T_amb
    gap          = T_core - T_surf

    # Rolling window features (60-second window)
    I_ser     = pd.Series(I)
    I_mean_60 = I_ser.rolling(60, min_periods=1).mean().values
    I_std_60  = I_ser.rolling(60, min_periods=1).std().fillna(0).values

    # Physics residuals for PINN loss
    dTc_dt = np.gradient(T_core, t)
    dTs_dt = np.gradient(T_surf, t)
    res1   = MCP_CORE*dTc_dt - Q_gen + K_COND*(T_core - T_surf)
    res2   = MCP_SURF*dTs_dt - K_COND*(T_core - T_surf) + HA*(T_surf - T_amb)

    out = pd.DataFrame({
        "time_s"             : t,
        "current_A"          : I,
        "voltage_V"          : V,
        "SOC"                : SOC,
        "T_ambient_C"        : T_amb,
        "T_surface_C"        : T_surf,
        "T_internal_C"       : T_core,       # derived ground truth
        "T_surface_real"     : T_surf,        # real thermocouple (kept for validation)
        "T_surface_predicted": df["T_surface_predicted"].values,  # model predicted surface
        "Q_gen_W"            : Q_gen,
        "R_internal_Ohm"     : R_int,
        "dI_dt"              : dI_dt,
        "dV_dt"              : dV_dt,
        "delta_T_surf"       : delta_T_surf,
        "delta_T_core"       : delta_T_core,
        "T_core_surf_gap"    : gap,
        "I_mean_60s"         : I_mean_60,
        "I_std_60s"          : I_std_60,
        "phys_res_1"         : res1,
        "phys_res_2"         : res2,
        "hotspot_flag"       : (T_core >= HOTSPOT_THRESH).astype(int),
        "danger_flag"        : (T_core >= DANGER_THRESH).astype(int),
        "profile"            : profile,
        "T_amb_label"        : T_amb_label,
        "run_id"             : run_id,
    })

    return out


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(filepath, run_id, profile, T_amb_label):
    """
    Load, resample, compute SOC, apply two-node model, engineer features.
    Returns processed DataFrame or None if file is unusable.
    """
    try:
        # Load raw data
        df_raw = load_mat_file(filepath)

        # Skip very short files
        duration = df_raw["time_s"].iloc[-1] - df_raw["time_s"].iloc[0]
        if duration < 300:
            print("    Skipping -- too short (" + str(round(duration)) + "s)")
            return None

        # Resample to 1-second intervals
        df = resample_to_1s(df_raw)
        n  = len(df)

        # Estimate initial SOC from starting voltage
        v_start  = float(df["voltage_V"].iloc[0])
        soc_init = float(np.clip((v_start - 3.2) / (4.2 - 3.2), 0.05, 0.99))

        # Compute SOC via Coulomb counting
        df["SOC"] = compute_soc(df["current_A"].values, soc_init=soc_init)

        # Apply two-node model to derive T_internal
        T_core, Q_gen = compute_t_internal(
            df["current_A"].values,
            df["T_surface_real"].values,
            df["T_ambient_C"].values,
            df["SOC"].values,
        )
        df["T_internal_C"] = T_core
        df["Q_gen_W"]      = Q_gen

        # Predict T_surface from T_core (for validation)
        T_surf_pred = predict_t_surface(T_core, df["T_ambient_C"].values)
        df["T_surface_predicted"] = T_surf_pred

        # Compute validation RMSE for this file
        surf_rmse = float(np.sqrt(np.mean(
            (T_surf_pred - df["T_surface_real"].values)**2
        )))

        # Engineer all features
        df_out = engineer_features(df, run_id, profile, T_amb_label)

        T_max = round(float(T_core.max()), 1)
        hs    = int((T_core >= HOTSPOT_THRESH).sum())
        print("    OK  rows=" + str(n) +
              "  T_max=" + str(T_max) + "C" +
              "  surf_RMSE=" + str(round(surf_rmse, 3)) + "C" +
              "  hs=" + str(hs))

        return df_out, surf_rmse

    except Exception as e:
        print("    FAILED: " + str(e))
        return None


# ============================================================
# STRATIFIED SPLIT
# ============================================================

def stratified_split(dataset, seed=42):
    np.random.seed(seed)
    dataset["split"] = "train"
    info = dataset.groupby("run_id")["profile"].first().reset_index()

    def base(p):
        p = str(p).upper()
        if "US06"    in p: return "US06"
        if "UDDS"    in p: return "UDDS"
        if "WLTP"    in p: return "WLTP"
        if "CHARGE"  in p: return "C20_charge"
        if "DISCHAR" in p: return "C20_discharge"
        return "OTHER"

    info["base"] = info["profile"].apply(base)
    print("\n  Stratified split:")
    for b in sorted(info["base"].unique()):
        runs = info[info["base"] == b]["run_id"].values
        np.random.shuffle(runs)
        n    = len(runs)
        if n <= 1:
            continue
        n_t = max(1, round(0.15 * n))
        n_v = max(1, round(0.15 * n))
        dataset.loc[dataset["run_id"].isin(runs[-n_t:]),           "split"] = "test"
        dataset.loc[dataset["run_id"].isin(runs[-(n_t+n_v):-n_t]), "split"] = "val"
        print("    " + b.ljust(14) + ": " +
              str(n-n_t-n_v) + " train  " +
              str(n_v) + " val  " + str(n_t) + " test")
    return dataset


# ============================================================
# VALIDATION PLOT -- KEY FIGURE FOR PAPER
# ============================================================

def save_validation_plot(all_rmses, sample_df):
    """
    Two-panel figure showing:
    1. Predicted vs real surface temperature for one run
    2. Distribution of surface RMSE across all runs
    This is the proof that our thermal model is valid.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Two-Node Thermal Model Validation Against Real CALB Measurements",
        fontsize=13, fontweight="bold"
    )

    # Panel 1 -- predicted vs real T_surface for one run
    t_min = sample_df["time_s"].values / 60.0
    axes[0].plot(t_min, sample_df["T_surface_real"].values,
                 color="#2563eb", lw=1.5, label="Real T_surface (thermocouple)")
    axes[0].plot(t_min, sample_df["T_surface_predicted"].values,
                 color="#dc2626", lw=1.0, ls="--",
                 label="Predicted T_surface (two-node model)")
    axes[0].plot(t_min, sample_df["T_internal_C"].values,
                 color="#7c3aed", lw=1.2, ls=":",
                 label="Derived T_internal")
    rmse_val = float(np.sqrt(np.mean(
        (sample_df["T_surface_predicted"].values -
         sample_df["T_surface_real"].values)**2
    )))
    axes[0].set_xlabel("Time (minutes)")
    axes[0].set_ylabel("Temperature (C)")
    axes[0].set_title("Surface temperature validation (RMSE = " +
                       str(round(rmse_val, 3)) + "C)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Panel 2 -- histogram of RMSE across all files
    axes[1].hist(all_rmses, bins=20, color="#2563eb",
                 alpha=0.7, edgecolor="white")
    axes[1].axvline(np.mean(all_rmses), color="#dc2626",
                    lw=2, label="Mean RMSE = " +
                    str(round(float(np.mean(all_rmses)), 3)) + "C")
    axes[1].axvline(1.5, color="#f59e0b", lw=1.5, ls="--",
                    label="Acceptance threshold (1.5C)")
    axes[1].set_xlabel("Surface Temperature RMSE (C)")
    axes[1].set_ylabel("Number of files")
    axes[1].set_title("RMSE distribution across all " +
                       str(len(all_rmses)) + " experimental runs")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "surface_validation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Validation plot saved: " + path)


# ============================================================
# MAIN
# ============================================================

def main():
    np.random.seed(42)
    print("\n" + "="*60)
    print("  CALB Real Data Preprocessor -- Paper 2")
    print("  CALB L148N58A 58Ah NMC | Real experimental data")
    print("="*60)

    all_dfs   = []
    all_rmses = []
    run_id    = 0
    failed    = 0
    sample_df = None

    for T_label, T_folder in sorted(TEMP_FOLDERS.items()):
        if not os.path.exists(T_folder):
            print("  WARNING: folder not found -- " + T_folder)
            continue

        for prof_folder, prof_name in sorted(PROFILE_FOLDERS.items()):
            prof_path = os.path.join(T_folder, prof_folder)
            if not os.path.exists(prof_path):
                continue

            mat_files = sorted([
                f for f in os.listdir(prof_path)
                if not f.endswith(".py") and not f.startswith(".")
            ])

            if len(mat_files) == 0:
                continue

            print("\n[" + T_label + " / " + prof_name + "]  " +
                  str(len(mat_files)) + " files")

            for fname in mat_files:
                fpath = os.path.join(prof_path, fname)
                cell_id = os.path.splitext(fname)[0]
                label   = prof_name + "_" + T_label + "_" + cell_id

                print("  " + label, end="  ")
                result = process_file(fpath, run_id, prof_name, T_label)

                if result is not None:
                    df_proc, surf_rmse = result
                    all_dfs.append(df_proc)
                    all_rmses.append(surf_rmse)
                    run_id += 1

                    # Keep one WLTP 25C run as validation example
                    if (sample_df is None and
                            "WLTP" in prof_name and T_label == "25C"):
                        sample_df = df_proc.copy()
                else:
                    failed += 1

    print("\n" + "="*60)
    if len(all_dfs) == 0:
        print("  ERROR: No files processed successfully.")
        return

    # Combine all runs
    print("  Combining " + str(run_id) + " runs (" +
          str(failed) + " failed)")
    dataset = pd.concat(all_dfs, ignore_index=True)

    # Stratified split
    dataset = stratified_split(dataset)

    # Surface validation summary
    mean_rmse   = float(np.mean(all_rmses))
    median_rmse = float(np.median(all_rmses))
    max_rmse    = float(np.max(all_rmses))
    pct_under   = float(np.mean(np.array(all_rmses) < 1.5) * 100)

    print("\n  SURFACE TEMPERATURE VALIDATION")
    print("  Mean RMSE    : " + str(round(mean_rmse,   3)) + "C")
    print("  Median RMSE  : " + str(round(median_rmse, 3)) + "C")
    print("  Max RMSE     : " + str(round(max_rmse,    3)) + "C")
    print("  Under 1.5C   : " + str(round(pct_under,   1)) + "% of runs")

    if mean_rmse < 1.5:
        print("  STATUS: VALIDATED -- thermal model accepted")
    else:
        print("  STATUS: WARNING -- mean RMSE above 1.5C threshold")
        print("  Consider adjusting K_COND or MCP_CORE parameters")

    # Dataset summary
    hs  = (dataset["T_internal_C"] >= HOTSPOT_THRESH).sum()
    dng = (dataset["T_internal_C"] >= DANGER_THRESH).sum()
    print("\n  DATASET SUMMARY")
    print("  Total rows   : " + str(len(dataset)))
    print("  Total runs   : " + str(dataset["run_id"].nunique()))
    print("  T_int min    : " +
          str(round(dataset["T_internal_C"].min(), 1)) + "C")
    print("  T_int max    : " +
          str(round(dataset["T_internal_C"].max(), 1)) + "C")
    print("  Hotspot>=44C : " + str(hs) +
          " (" + str(round(100*hs/len(dataset), 1)) + "%)")
    print("  Danger >=55C : " + str(dng) +
          " (" + str(round(100*dng/len(dataset), 1)) + "%)")
    print()

    for sp in ["train", "val", "test"]:
        sub = dataset[dataset["split"] == sp]
        sh  = (sub["T_internal_C"] >= HOTSPOT_THRESH).sum()
        print("  [" + sp.upper().ljust(5) + "] " +
              str(sub["run_id"].nunique()).rjust(3) + " runs  " +
              str(len(sub)).rjust(8) + " rows  " +
              str(sh).rjust(6) + " hotspot")

    # Save CSV files
    print("\n  Saving CSV files...")
    dataset.to_csv(
        os.path.join(OUTPUT_DIR, "calb_full_dataset.csv"), index=False
    )
    for sp in ["train", "val", "test"]:
        sub  = dataset[dataset["split"] == sp]
        path = os.path.join(OUTPUT_DIR, "calb_" + sp + ".csv")
        sub.to_csv(path, index=False)
        print("  calb_" + sp + ".csv  " + str(len(sub)) + " rows")

    # Save validation plot
    if sample_df is not None:
        save_validation_plot(all_rmses, sample_df)

    # Save validation numbers to text file
    val_path = os.path.join(OUTPUT_DIR, "thermal_validation.txt")
    with open(val_path, "w") as f:
        f.write("Two-Node Thermal Model Validation\n")
        f.write("="*40 + "\n")
        f.write("Mean surface RMSE    : " + str(round(mean_rmse,   3)) + " C\n")
        f.write("Median surface RMSE  : " + str(round(median_rmse, 3)) + " C\n")
        f.write("Max surface RMSE     : " + str(round(max_rmse,    3)) + " C\n")
        f.write("Runs under 1.5C      : " + str(round(pct_under,   1)) + "%\n")
        f.write("Total runs validated : " + str(len(all_rmses)) + "\n")
        f.write("\nIndividual run RMSEs:\n")
        for i, r in enumerate(all_rmses):
            f.write("  Run " + str(i) + ": " + str(round(r, 3)) + " C\n")
    print("  thermal_validation.txt saved")

    print("\n  Done. Now run: python battery_train_p2.py")
    print("="*60)


if __name__ == "__main__":
    main()
