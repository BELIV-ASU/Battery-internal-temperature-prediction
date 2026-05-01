"""
Battery Model Validation Script -- Paper 2
===========================================
Validates both trained models on the CALB real experimental dataset.
Produces all tables and figures needed for the paper.

HOW TO USE:
    Make sure battery_train_p2.py has already been run.
    python battery_validate_p2.py

OUTPUTS (saved to results_p2/validation/):
    validation_report.txt   -- all numbers for paper tables
    scatter_plot.png        -- predicted vs true (paper figure)
    error_distribution.png  -- error histogram (paper figure)
    rmse_by_profile.png     -- RMSE per drive cycle (paper figure)
    rmse_by_temperature.png -- RMSE per ambient temp (paper figure)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# -- Settings ----------------------------------------------
DATA_FOLDER    = "calb_data"
RESULTS_FOLDER = "results_p2"
VALID_FOLDER   = os.path.join(RESULTS_FOLDER, "validation")
os.makedirs(VALID_FOLDER, exist_ok=True)

FEATURE_COLS = [
    "current_A", "voltage_V", "SOC", "T_ambient_C",
    "T_surface_C", "Q_gen_W", "dI_dt", "dV_dt",
    "delta_T_surf", "I_mean_60s", "I_std_60s", "R_internal_Ohm",
]
TARGET_COL = "T_internal_C"
SEQ_LEN    = 60


# -- LSTM Model (same as training script) ------------------
class BatteryLSTM(nn.Module):
    def __init__(self, input_size=12, hidden_size=256,
                 num_layers=2, dropout=0.2):
        super(BatteryLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# -- Load data and fit scalers ------------------------------
def load_data():
    print("[Loading data...]")
    train_df = pd.read_csv(os.path.join(DATA_FOLDER, "calb_train.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_FOLDER, "calb_test.csv"))

    for df in [train_df, test_df]:
        for col in FEATURE_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    scaler_X.fit(train_df[FEATURE_COLS].values)
    scaler_y.fit(train_df[[TARGET_COL]].values)

    print("  Train rows : " + str(len(train_df)))
    print("  Test rows  : " + str(len(test_df)))
    return train_df, test_df, scaler_X, scaler_y


# -- Get predictions for entire test set -------------------
def get_predictions(model, test_df, scaler_X, scaler_y, device):
    model.eval()
    all_preds  = []
    all_truths = []
    all_info   = []

    for run_id, grp in test_df.groupby("run_id"):
        grp = grp.reset_index(drop=True)
        if len(grp) < SEQ_LEN + 1:
            continue

        X = scaler_X.transform(
            grp[FEATURE_COLS].values
        ).astype(np.float32)
        y_raw = grp[TARGET_COL].values

        for start in range(0, len(grp) - SEQ_LEN, 30):
            end   = start + SEQ_LEN
            X_seq = torch.tensor(
                X[start:end]
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_norm = model(X_seq).cpu().numpy()

            pred_C  = float(scaler_y.inverse_transform(
                [[pred_norm[0]]])[0][0])
            truth_C = float(y_raw[end - 1])

            profile = str(grp["profile"].iloc[0]) \
                if "profile" in grp.columns else "unknown"
            T_amb_label = str(grp["T_amb_label"].iloc[0]) \
                if "T_amb_label" in grp.columns else "unknown"

            all_preds.append(pred_C)
            all_truths.append(truth_C)
            all_info.append({
                "run_id"     : run_id,
                "profile"    : profile,
                "T_amb_label": T_amb_label,
            })

    preds  = np.array(all_preds)
    truths = np.array(all_truths)
    info   = pd.DataFrame(all_info)
    return preds, truths, info


# -- Overall metrics ----------------------------------------
def compute_metrics(preds, truths, label="Model"):
    errors = preds - truths
    rmse   = float(np.sqrt(np.mean(errors**2)))
    mae    = float(np.mean(np.abs(errors)))
    r2     = float(1 - np.sum(errors**2) /
                   np.sum((truths - truths.mean())**2))
    within_1 = float(np.mean(np.abs(errors) <= 1.0) * 100)

    print("\n  " + label + " -- Overall Metrics:")
    print("  " + "-"*35)
    print("  RMSE      : " + str(round(rmse, 4)) + " C")
    print("  MAE       : " + str(round(mae,  4)) + " C")
    print("  R-squared : " + str(round(r2,   5)))
    print("  Within 1C : " + str(round(within_1, 1)) + "%")

    return {
        "RMSE": rmse, "MAE": mae,
        "R2": r2, "within_1": within_1
    }


# -- Breakdown by drive cycle profile ----------------------
def metrics_by_profile(preds, truths, info, label):
    print("\n  " + label + " -- By Drive Cycle Profile:")
    print("  " + "-"*45)
    print("  Profile          RMSE     MAE      R2")

    rows = []
    for prof in sorted(info["profile"].unique()):
        mask = info["profile"].values == prof
        p    = preds[mask]
        t    = truths[mask]
        if len(p) < 10:
            continue
        rmse = float(np.sqrt(np.mean((p - t)**2)))
        mae  = float(np.mean(np.abs(p - t)))
        r2   = float(1 - np.sum((p-t)**2) /
                     np.sum((t - t.mean())**2))
        print("  " + str(prof).ljust(16) +
              str(round(rmse, 4)).ljust(9) +
              str(round(mae,  4)).ljust(9) +
              str(round(r2,   4)))
        rows.append({
            "profile": prof,
            "RMSE": rmse, "MAE": mae, "R2": r2
        })
    return pd.DataFrame(rows)


# -- Breakdown by ambient temperature ----------------------
def metrics_by_temperature(preds, truths, info, label):
    print("\n  " + label + " -- By Ambient Temperature:")
    print("  " + "-"*45)
    print("  T_ambient    RMSE     MAE      R2")

    rows = []
    for T_label in sorted(info["T_amb_label"].unique()):
        mask = info["T_amb_label"].values == T_label
        p    = preds[mask]
        t    = truths[mask]
        if len(p) < 10:
            continue
        rmse = float(np.sqrt(np.mean((p - t)**2)))
        mae  = float(np.mean(np.abs(p - t)))
        r2   = float(1 - np.sum((p-t)**2) /
                     np.sum((t - t.mean())**2))
        print("  " + str(T_label).ljust(13) +
              str(round(rmse, 4)).ljust(9) +
              str(round(mae,  4)).ljust(9) +
              str(round(r2,   4)))
        rows.append({
            "T_ambient": T_label,
            "RMSE": rmse, "MAE": mae, "R2": r2
        })
    return pd.DataFrame(rows)


# -- Figure 1: Scatter plot predicted vs true --------------
def plot_scatter(b_preds, b_truths, p_preds, p_truths, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Predicted vs True Internal Temperature -- Test Set",
        fontsize=13, fontweight="bold"
    )

    for ax, preds, truths, title, color in [
        (axes[0], b_preds, b_truths, "Baseline LSTM",  "#2563eb"),
        (axes[1], p_preds, p_truths, "LSTM + PINN",    "#dc2626"),
    ]:
        ax.scatter(truths, preds, alpha=0.2, s=6,
                   color=color, label="Predictions")
        lims = [
            min(truths.min(), preds.min()) - 1,
            max(truths.max(), preds.max()) + 1
        ]
        ax.plot(lims, lims, "k--", lw=1.5,
                label="Perfect prediction")
        ax.set_xlabel("True T_internal (C)", fontsize=11)
        ax.set_ylabel("Predicted T_internal (C)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        rmse = float(np.sqrt(np.mean((preds - truths)**2)))
        mae  = float(np.mean(np.abs(preds - truths)))
        r2   = float(1 - np.sum((preds-truths)**2) /
                     np.sum((truths - truths.mean())**2))
        ax.text(0.05, 0.95,
                "RMSE = " + str(round(rmse, 4)) + " C\n" +
                "MAE  = " + str(round(mae,  4)) + " C\n" +
                "R2   = " + str(round(r2,   5)),
                transform=ax.transAxes, fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round",
                          facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: " + path)


# -- Figure 2: Error distribution --------------------------
def plot_error_distribution(b_preds, b_truths,
                             p_preds, p_truths, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Prediction Error Distribution -- Test Set",
        fontsize=13, fontweight="bold"
    )

    for ax, preds, truths, title, color in [
        (axes[0], b_preds, b_truths, "Baseline LSTM",  "#2563eb"),
        (axes[1], p_preds, p_truths, "LSTM + PINN",    "#dc2626"),
    ]:
        errors = preds - truths
        ax.hist(errors, bins=60, color=color,
                alpha=0.7, edgecolor="white")
        ax.axvline(0,    color="black",   lw=1.5,
                   label="Zero error")
        ax.axvline( 0.5, color="#f59e0b", lw=1.0,
                   ls="--", alpha=0.8, label="+/- 0.5C")
        ax.axvline(-0.5, color="#f59e0b", lw=1.0,
                   ls="--", alpha=0.8)

        mean_e   = float(errors.mean())
        std_e    = float(errors.std())
        within_h = float(np.mean(np.abs(errors) <= 0.5) * 100)
        within_1 = float(np.mean(np.abs(errors) <= 1.0) * 100)

        ax.set_xlabel("Prediction Error (C)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.text(0.05, 0.95,
                "Mean : " + str(round(mean_e,   4)) + " C\n" +
                "Std  : " + str(round(std_e,    4)) + " C\n" +
                "Within 0.5C: " + str(round(within_h, 1)) + "%\n" +
                "Within 1.0C: " + str(round(within_1, 1)) + "%",
                transform=ax.transAxes, fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round",
                          facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: " + path)


# -- Figure 3: RMSE by drive cycle profile -----------------
def plot_rmse_by_profile(b_by_prof, p_by_prof, path):
    profiles  = b_by_prof["profile"].tolist()
    b_rmses   = b_by_prof["RMSE"].tolist()
    p_rmses   = p_by_prof["RMSE"].tolist()

    x = np.arange(len(profiles))
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - w/2, b_rmses, w,
                   label="Baseline LSTM", color="#2563eb", alpha=0.8)
    bars2 = ax.bar(x + w/2, p_rmses, w,
                   label="LSTM + PINN",  color="#dc2626", alpha=0.8)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.001,
                str(round(h, 4)),
                ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.001,
                str(round(h, 4)),
                ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Drive Cycle Profile", fontsize=12)
    ax.set_ylabel("RMSE (C)", fontsize=12)
    ax.set_title("RMSE by Drive Cycle Profile\n(Lower is better)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(profiles, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_ylim(0, max(max(b_rmses), max(p_rmses)) * 1.25)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: " + path)


# -- Figure 4: RMSE by ambient temperature -----------------
def plot_rmse_by_temperature(b_by_temp, p_by_temp, path):
    temps   = b_by_temp["T_ambient"].tolist()
    b_rmses = b_by_temp["RMSE"].tolist()
    p_rmses = p_by_temp["RMSE"].tolist()

    x = np.arange(len(temps))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - w/2, b_rmses, w,
                   label="Baseline LSTM", color="#2563eb", alpha=0.8)
    bars2 = ax.bar(x + w/2, p_rmses, w,
                   label="LSTM + PINN",  color="#dc2626", alpha=0.8)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.001,
                str(round(h, 4)),
                ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.001,
                str(round(h, 4)),
                ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Ambient Temperature", fontsize=12)
    ax.set_ylabel("RMSE (C)", fontsize=12)
    ax.set_title(
        "RMSE by Ambient Temperature\n"
        "(shows generalisation across thermal conditions)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(temps, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_ylim(0, max(max(b_rmses), max(p_rmses)) * 1.25)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: " + path)


# -- Save full validation report ---------------------------
def save_report(b_metrics, p_metrics,
                b_by_prof, p_by_prof,
                b_by_temp, p_by_temp):
    path = os.path.join(VALID_FOLDER, "validation_report.txt")
    with open(path, "w") as f:
        f.write("Battery Model Validation Report -- Paper 2\n")
        f.write("CALB L148N58A Real Experimental Data\n")
        f.write("=" * 50 + "\n\n")

        f.write("1. OVERALL PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        f.write("Metric        Baseline LSTM   LSTM+PINN\n")
        f.write("RMSE (C)      " +
                str(round(b_metrics["RMSE"], 4)).ljust(16) +
                str(round(p_metrics["RMSE"], 4)) + "\n")
        f.write("MAE  (C)      " +
                str(round(b_metrics["MAE"],  4)).ljust(16) +
                str(round(p_metrics["MAE"],  4)) + "\n")
        f.write("R-squared     " +
                str(round(b_metrics["R2"],   5)).ljust(16) +
                str(round(p_metrics["R2"],   5)) + "\n")
        f.write("Within 1C     " +
                str(round(b_metrics["within_1"], 1)).ljust(15) + "%" +
                "  " + str(round(p_metrics["within_1"], 1)) + "%\n\n")

        f.write("2. PERFORMANCE BY DRIVE CYCLE PROFILE\n")
        f.write("-" * 40 + "\n")
        f.write("Baseline LSTM:\n")
        f.write(b_by_prof.to_string(index=False) + "\n\n")
        f.write("LSTM+PINN:\n")
        f.write(p_by_prof.to_string(index=False) + "\n\n")

        f.write("3. PERFORMANCE BY AMBIENT TEMPERATURE\n")
        f.write("-" * 40 + "\n")
        f.write("Baseline LSTM:\n")
        f.write(b_by_temp.to_string(index=False) + "\n\n")
        f.write("LSTM+PINN:\n")
        f.write(p_by_temp.to_string(index=False) + "\n\n")

        f.write("4. COMPARISON AGAINST PUBLISHED WORK\n")
        f.write("-" * 40 + "\n")
        f.write("Method                RMSE (C)\n")
        f.write("Wang et al. 2021      0.850\n")
        f.write("Karnehm et al. 2024   0.751\n")
        f.write("Our Baseline LSTM     " +
                str(round(b_metrics["RMSE"], 4)) + "\n")
        f.write("Our LSTM+PINN         " +
                str(round(p_metrics["RMSE"], 4)) + "\n")

    print("  Saved: " + path)


# -- Main --------------------------------------------------


def plot_time_series(model_b, model_p, test_df,
                     scaler_X, scaler_y, device, path):
    candidates = test_df[
        (test_df["profile"] == "WLTP") &
        (test_df["T_amb_label"] == "25C")
    ]
    if len(candidates) == 0:
        candidates = test_df
    run_id = candidates.groupby("run_id").size().idxmax()
    grp    = test_df[test_df["run_id"] == run_id].reset_index(drop=True)

    X      = scaler_X.transform(grp[FEATURE_COLS].values).astype("float32")
    y_true = grp[TARGET_COL].values
    t_surf = grp["T_surface_C"].values
    t_mins = grp["time_s"].values / 60.0

    b_preds, p_preds, t_idxs = [], [], []
    for start in range(0, len(grp) - SEQ_LEN, 10):
        end   = start + SEQ_LEN
        X_seq = torch.tensor(X[start:end]).unsqueeze(0).to(device)
        with torch.no_grad():
            b_n = model_b(X_seq).cpu().numpy()
            p_n = model_p(X_seq).cpu().numpy()
        b_preds.append(float(scaler_y.inverse_transform([[b_n[0]]])[0][0]))
        p_preds.append(float(scaler_y.inverse_transform([[p_n[0]]])[0][0]))
        t_idxs.append(end - 1)

    t_plot  = t_mins[t_idxs]
    b_arr   = np.array(b_preds)
    p_arr   = np.array(p_preds)
    t_true  = y_true[t_idxs]
    t_s_arr = t_surf[t_idxs]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Temperature Prediction Over Time -- WLTP 25C",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(t_plot, t_true,  color="#7c3aed", lw=2.0, label="True T_internal")
    ax.plot(t_plot, b_arr,   color="#2563eb", lw=1.2, ls="--",
            alpha=0.8, label="Baseline LSTM")
    ax.plot(t_plot, p_arr,   color="#dc2626", lw=1.2, ls="-.",
            alpha=0.8, label="LSTM + PINN")
    ax.plot(t_plot, t_s_arr, color="#f97316", lw=1.0, ls=":",
            alpha=0.7, label="T_surface (sensor)")
    ax.fill_between(t_plot, t_true, t_s_arr, alpha=0.12,
                    color="#7c3aed", label="Core-surface gap")
    ax.set_ylabel("Temperature (C)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    ax2 = axes[1]
    b_err = b_arr - t_true
    p_err = p_arr - t_true
    ax2.plot(t_plot, b_err, color="#2563eb", lw=1.2,
             label="Baseline error RMSE=" +
             str(round(float(np.sqrt(np.mean(b_err**2))), 4)) + "C")
    ax2.plot(t_plot, p_err, color="#dc2626", lw=1.2,
             label="PINN error RMSE=" +
             str(round(float(np.sqrt(np.mean(p_err**2))), 4)) + "C")
    ax2.axhline(0, color="black", lw=1.0, ls="--")
    ax2.fill_between(t_plot, -0.5, 0.5, alpha=0.08,
                     color="green", label="+/- 0.5C band")
    ax2.set_xlabel("Time (minutes)", fontsize=11)
    ax2.set_ylabel("Prediction Error (C)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: " + path)


def plot_rmse_heatmap(b_preds, b_truths,
                      p_preds, p_truths, info, path):
    profiles = sorted(info["profile"].unique())
    temps    = sorted(info["T_amb_label"].unique())

    b_grid = np.zeros((len(profiles), len(temps)))
    p_grid = np.zeros((len(profiles), len(temps)))

    for i, prof in enumerate(profiles):
        for j, temp in enumerate(temps):
            mask = ((info["profile"].values == prof) &
                    (info["T_amb_label"].values == temp))
            if mask.sum() < 5:
                b_grid[i, j] = np.nan
                p_grid[i, j] = np.nan
                continue
            b_grid[i, j] = float(np.sqrt(
                np.mean((b_preds[mask] - b_truths[mask])**2)))
            p_grid[i, j] = float(np.sqrt(
                np.mean((p_preds[mask] - p_truths[mask])**2)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("RMSE Heatmap -- Drive Cycle x Ambient Temperature",
                 fontsize=13, fontweight="bold")

    vmin = float(np.nanmin([b_grid, p_grid]))
    vmax = float(np.nanmax([b_grid, p_grid]))

    for ax, grid, title in [
        (axes[0], b_grid, "Baseline LSTM"),
        (axes[1], p_grid, "LSTM + PINN"),
    ]:
        im = ax.imshow(grid, cmap="RdYlGn_r",
                       vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(temps)))
        ax.set_yticks(range(len(profiles)))
        ax.set_xticklabels(temps, fontsize=11)
        ax.set_yticklabels(profiles, fontsize=10)
        ax.set_xlabel("Ambient Temperature", fontsize=11)
        ax.set_ylabel("Drive Cycle Profile", fontsize=11)
        ax.set_title(title, fontsize=12)
        for i in range(len(profiles)):
            for j in range(len(temps)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, str(round(grid[i, j], 3)),
                            ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color="black")
        plt.colorbar(im, ax=ax, label="RMSE (C)")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: " + path)

def main():
    print("\n" + "=" * 55)
    print("  Battery Model Validation -- Paper 2")
    print("  CALB Real Experimental Data")
    print("=" * 55)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("  Device: " + DEVICE)

    # Load data
    train_df, test_df, scaler_X, scaler_y = load_data()

    # Load saved models
    print("\n[Loading saved models...]")
    baseline = BatteryLSTM(input_size=12).to(DEVICE)
    pinn     = BatteryLSTM(input_size=12).to(DEVICE)

    b_path = os.path.join(RESULTS_FOLDER, "p2_baseline_lstm.pt")
    p_path = os.path.join(RESULTS_FOLDER, "p2_pinn_lstm.pt")

    if not os.path.exists(b_path):
        print("  ERROR: " + b_path + " not found.")
        print("  Run battery_train_p2.py first.")
        return
    if not os.path.exists(p_path):
        print("  ERROR: " + p_path + " not found.")
        print("  Run battery_train_p2.py first.")
        return

    baseline.load_state_dict(
        torch.load(b_path, map_location=DEVICE))
    pinn.load_state_dict(
        torch.load(p_path, map_location=DEVICE))
    print("  Models loaded successfully.")

    # Get predictions
    print("\n[Getting predictions on test set...]")
    b_preds, b_truths, info = get_predictions(
        baseline, test_df, scaler_X, scaler_y, DEVICE)
    p_preds, p_truths, _    = get_predictions(
        pinn,     test_df, scaler_X, scaler_y, DEVICE)
    print("  Predictions: " + str(len(b_preds)) + " samples")

    # Compute all metrics
    print("\n[Computing metrics...]")
    b_metrics = compute_metrics(b_preds, b_truths, "Baseline LSTM")
    p_metrics = compute_metrics(p_preds, p_truths, "LSTM + PINN")

    print("\n[Breakdown by drive cycle profile...]")
    b_by_prof = metrics_by_profile(
        b_preds, b_truths, info, "Baseline LSTM")
    p_by_prof = metrics_by_profile(
        p_preds, p_truths, info, "LSTM + PINN")

    print("\n[Breakdown by ambient temperature...]")
    b_by_temp = metrics_by_temperature(
        b_preds, b_truths, info, "Baseline LSTM")
    p_by_temp = metrics_by_temperature(
        p_preds, p_truths, info, "LSTM + PINN")

    # Generate figures
    print("\n[Generating figures...]")
    plot_scatter(
        b_preds, b_truths, p_preds, p_truths,
        os.path.join(VALID_FOLDER, "scatter_plot.png"))

    plot_error_distribution(
        b_preds, b_truths, p_preds, p_truths,
        os.path.join(VALID_FOLDER, "error_distribution.png"))

    plot_rmse_by_profile(
        b_by_prof, p_by_prof,
        os.path.join(VALID_FOLDER, "rmse_by_profile.png"))

    plot_rmse_by_temperature(
        b_by_temp, p_by_temp,
        os.path.join(VALID_FOLDER, "rmse_by_temperature.png"))

    plot_time_series(
        baseline, pinn, test_df, scaler_X, scaler_y, DEVICE,
        os.path.join(VALID_FOLDER, "time_series_prediction.png"))

    plot_rmse_heatmap(
        b_preds, b_truths, p_preds, p_truths, info,
        os.path.join(VALID_FOLDER, "rmse_heatmap.png"))

    # Save report
    print("\n[Saving validation report...]")
    save_report(b_metrics, p_metrics,
                b_by_prof, p_by_prof,
                b_by_temp, p_by_temp)

    print("\n" + "=" * 55)
    print("  Validation Complete!")
    print("  All files saved to: " + os.path.abspath(VALID_FOLDER))
    print()
    print("  Files generated:")
    print("    validation_report.txt   -- numbers for paper tables")
    print("    scatter_plot.png        -- paper Figure 1")
    print("    error_distribution.png  -- paper Figure 2")
    print("    rmse_by_profile.png     -- paper Figure 3")
    print("    rmse_by_temperature.png -- paper Figure 4")
    print("=" * 55)


if __name__ == "__main__":
    main()
