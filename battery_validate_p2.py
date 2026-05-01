"""
Battery Model Validation Script -- Paper 2 (All Three Models)
=============================================================
Validates Baseline LSTM, GRU, and LSTM+PINN together.
Produces all figures and tables needed for the paper.

HOW TO USE:
    Run battery_train_p2.py and battery_train_gru.py first.
    python battery_validate_p2.py

OUTPUTS (saved to results_p2/validation/):
    validation_report.txt       -- all numbers for paper tables
    scatter_plot.png            -- predicted vs true (3 models)
    error_distribution.png      -- error histogram (3 models)
    rmse_by_profile.png         -- RMSE per drive cycle
    rmse_by_temperature.png     -- RMSE per ambient temp
    time_series_all_models.png  -- temperature vs time (KEY FIGURE)
    rmse_heatmap.png            -- profile x temperature grid
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
RESULTS_LSTM   = "results_p2"
RESULTS_GRU    = "results_gru"
VALID_FOLDER   = os.path.join(RESULTS_LSTM, "validation")
os.makedirs(VALID_FOLDER, exist_ok=True)

FEATURE_COLS = [
    "current_A", "voltage_V", "SOC", "T_ambient_C",
    "T_surface_C", "Q_gen_W", "dI_dt", "dV_dt",
    "delta_T_surf", "I_mean_60s", "I_std_60s", "R_internal_Ohm",
]
TARGET_COL = "T_internal_C"
SEQ_LEN    = 60


# -- Model definitions -------------------------------------

class BatteryLSTM(nn.Module):
    def __init__(self, input_size=12, hidden_size=256,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class BatteryGRU(nn.Module):
    def __init__(self, input_size=12, hidden_size=256,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# -- Load data ---------------------------------------------

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

    print("  Test rows: " + str(len(test_df)))
    return train_df, test_df, scaler_X, scaler_y


# -- Get predictions across all test runs ------------------

def get_predictions(model, test_df, scaler_X, scaler_y, device):
    model.eval()
    all_preds, all_truths, all_info = [], [], []

    for run_id, grp in test_df.groupby("run_id"):
        grp = grp.reset_index(drop=True)
        if len(grp) < SEQ_LEN + 1:
            continue
        X = scaler_X.transform(
            grp[FEATURE_COLS].values).astype(np.float32)
        y_raw = grp[TARGET_COL].values

        for start in range(0, len(grp) - SEQ_LEN, 30):
            end   = start + SEQ_LEN
            X_seq = torch.tensor(
                X[start:end]).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_norm = model(X_seq).cpu().numpy()
            pred_C  = float(scaler_y.inverse_transform(
                [[pred_norm[0]]])[0][0])
            truth_C = float(y_raw[end - 1])
            profile   = str(grp["profile"].iloc[0]) \
                if "profile" in grp.columns else "unknown"
            T_label   = str(grp["T_amb_label"].iloc[0]) \
                if "T_amb_label" in grp.columns else "unknown"
            all_preds.append(pred_C)
            all_truths.append(truth_C)
            all_info.append({
                "run_id": run_id,
                "profile": profile,
                "T_amb_label": T_label,
            })

    return (np.array(all_preds), np.array(all_truths),
            pd.DataFrame(all_info))


# -- Metrics -----------------------------------------------

def compute_metrics(preds, truths, label="Model"):
    errors   = preds - truths
    rmse     = float(np.sqrt(np.mean(errors**2)))
    mae      = float(np.mean(np.abs(errors)))
    r2       = float(1 - np.sum(errors**2) /
                     np.sum((truths - truths.mean())**2))
    within_1 = float(np.mean(np.abs(errors) <= 1.0) * 100)
    within_h = float(np.mean(np.abs(errors) <= 0.5) * 100)

    print("\n  " + label + ":")
    print("  " + "-"*35)
    print("  RMSE        : " + str(round(rmse, 4)) + " C")
    print("  MAE         : " + str(round(mae,  4)) + " C")
    print("  R-squared   : " + str(round(r2,   5)))
    print("  Within 0.5C : " + str(round(within_h, 1)) + "%")
    print("  Within 1.0C : " + str(round(within_1, 1)) + "%")

    return {"RMSE": rmse, "MAE": mae, "R2": r2,
            "within_1": within_1, "within_h": within_h}


def metrics_by_profile(preds, truths, info, label):
    print("\n  " + label + " -- By Profile:")
    print("  Profile          RMSE     MAE")
    rows = []
    for prof in sorted(info["profile"].unique()):
        mask = info["profile"].values == prof
        p, t = preds[mask], truths[mask]
        if len(p) < 5:
            continue
        rmse = float(np.sqrt(np.mean((p-t)**2)))
        mae  = float(np.mean(np.abs(p-t)))
        r2   = float(1 - np.sum((p-t)**2) /
                     np.sum((t-t.mean())**2))
        print("  " + str(prof).ljust(16) +
              str(round(rmse, 4)).ljust(9) +
              str(round(mae,  4)))
        rows.append({"profile": prof, "RMSE": rmse,
                     "MAE": mae, "R2": r2})
    return pd.DataFrame(rows)


def metrics_by_temperature(preds, truths, info, label):
    print("\n  " + label + " -- By Temperature:")
    print("  T_ambient    RMSE     MAE")
    rows = []
    for T_label in sorted(info["T_amb_label"].unique()):
        mask = info["T_amb_label"].values == T_label
        p, t = preds[mask], truths[mask]
        if len(p) < 5:
            continue
        rmse = float(np.sqrt(np.mean((p-t)**2)))
        mae  = float(np.mean(np.abs(p-t)))
        r2   = float(1 - np.sum((p-t)**2) /
                     np.sum((t-t.mean())**2))
        print("  " + str(T_label).ljust(13) +
              str(round(rmse, 4)).ljust(9) +
              str(round(mae,  4)))
        rows.append({"T_ambient": T_label, "RMSE": rmse,
                     "MAE": mae, "R2": r2})
    return pd.DataFrame(rows)


# -- Figure 1: Temperature vs Time -- ALL THREE MODELS -----
# This is the key figure -- shows all models on one graph

def plot_time_series_all_models(model_b, model_p, model_g,
                                 test_df, scaler_X, scaler_y,
                                 device, path):
    """
    The most important figure for the paper.
    Shows predicted vs real temperature over time
    for all three models on the same graph.
    Picks a WLTP run at 25C for best illustration.
    """
    # Find best run -- WLTP at 25C, longest available
    cands = test_df[
        (test_df["profile"] == "WLTP") &
        (test_df["T_amb_label"] == "25C")
    ]
    if len(cands) == 0:
        cands = test_df[test_df["profile"] == "WLTP"]
    if len(cands) == 0:
        cands = test_df

    run_id = cands.groupby("run_id").size().idxmax()
    grp    = test_df[test_df["run_id"] == run_id].reset_index(drop=True)

    X      = scaler_X.transform(
        grp[FEATURE_COLS].values).astype(np.float32)
    y_true = grp[TARGET_COL].values
    t_surf = grp["T_surface_C"].values
    t_mins = grp["time_s"].values / 60.0 \
        if "time_s" in grp.columns \
        else np.arange(len(grp)) / 60.0

    b_preds, p_preds, g_preds, t_idxs = [], [], [], []

    for start in range(0, len(grp) - SEQ_LEN, 5):
        end   = start + SEQ_LEN
        X_seq = torch.tensor(
            X[start:end]).unsqueeze(0).to(device)
        with torch.no_grad():
            b_n = model_b(X_seq).cpu().numpy()
            p_n = model_p(X_seq).cpu().numpy()
            g_n = model_g(X_seq).cpu().numpy()
        b_preds.append(float(scaler_y.inverse_transform(
            [[b_n[0]]])[0][0]))
        p_preds.append(float(scaler_y.inverse_transform(
            [[p_n[0]]])[0][0]))
        g_preds.append(float(scaler_y.inverse_transform(
            [[g_n[0]]])[0][0]))
        t_idxs.append(end - 1)

    t_plot  = t_mins[t_idxs]
    b_arr   = np.array(b_preds)
    p_arr   = np.array(p_preds)
    g_arr   = np.array(g_preds)
    t_true  = y_true[t_idxs]
    t_s_arr = t_surf[t_idxs]

    # Compute RMSE for each model on this run
    b_rmse = round(float(np.sqrt(np.mean((b_arr - t_true)**2))), 4)
    p_rmse = round(float(np.sqrt(np.mean((p_arr - t_true)**2))), 4)
    g_rmse = round(float(np.sqrt(np.mean((g_arr - t_true)**2))), 4)

    fig, axes = plt.subplots(2, 1, figsize=(15, 9))
    fig.suptitle(
        "Temperature vs Time -- All Three Models\n"
        "WLTP Drive Cycle at 25C Ambient",
        fontsize=14, fontweight="bold"
    )

    # Top panel -- all temperatures on one graph
    ax = axes[0]
    ax.plot(t_plot, t_true,
            color="#000000", lw=2.5,
            label="True T_internal (ground truth)", zorder=5)
    ax.plot(t_plot, b_arr,
            color="#2563eb", lw=1.5, ls="--",
            label="Baseline LSTM  (RMSE=" + str(b_rmse) + "C)",
            alpha=0.85)
    ax.plot(t_plot, g_arr,
            color="#16a34a", lw=1.5, ls="-.",
            label="GRU            (RMSE=" + str(g_rmse) + "C)",
            alpha=0.85)
    ax.plot(t_plot, p_arr,
            color="#dc2626", lw=1.5, ls=":",
            label="LSTM + PINN    (RMSE=" + str(p_rmse) + "C)",
            alpha=0.85)
    ax.plot(t_plot, t_s_arr,
            color="#f97316", lw=1.0, ls="-",
            alpha=0.5, label="T_surface (real sensor)")
    ax.fill_between(t_plot, t_true, t_s_arr,
                    alpha=0.08, color="#7c3aed",
                    label="Core-surface gap")
    ax.set_ylabel("Temperature (C)", fontsize=12)
    ax.set_title("All models vs ground truth -- "
                 "closer to black line = more accurate",
                 fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.25)

    # Bottom panel -- prediction errors for all three
    ax2 = axes[1]
    b_err = b_arr - t_true
    p_err = p_arr - t_true
    g_err = g_arr - t_true

    ax2.plot(t_plot, b_err,
             color="#2563eb", lw=1.2, ls="--",
             label="Baseline LSTM error", alpha=0.85)
    ax2.plot(t_plot, g_err,
             color="#16a34a", lw=1.2, ls="-.",
             label="GRU error", alpha=0.85)
    ax2.plot(t_plot, p_err,
             color="#dc2626", lw=1.2, ls=":",
             label="LSTM+PINN error", alpha=0.85)
    ax2.axhline(0,    color="black", lw=1.0, ls="-")
    ax2.axhline( 0.5, color="#f59e0b", lw=0.8,
                ls="--", alpha=0.6, label="+/- 0.5C band")
    ax2.axhline(-0.5, color="#f59e0b", lw=0.8,
                ls="--", alpha=0.6)
    ax2.fill_between(t_plot, -0.5, 0.5,
                     alpha=0.06, color="green")
    ax2.set_ylabel("Prediction Error (C)", fontsize=12)
    ax2.set_xlabel("Time (minutes)", fontsize=12)
    ax2.set_title(
        "Prediction error over time -- "
        "closer to zero = more accurate",
        fontsize=11
    )
    ax2.legend(fontsize=9, loc="best")
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: " + path)


# -- Figure 2: Scatter plot (all 3 models) -----------------

def plot_scatter_3(b_preds, b_truths,
                   g_preds, g_truths,
                   p_preds, p_truths, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Predicted vs True Internal Temperature -- Test Set",
        fontsize=13, fontweight="bold")

    for ax, preds, truths, title, color in [
        (axes[0], b_preds, b_truths, "Baseline LSTM",  "#2563eb"),
        (axes[1], g_preds, g_truths, "GRU",             "#16a34a"),
        (axes[2], p_preds, p_truths, "LSTM + PINN",    "#dc2626"),
    ]:
        ax.scatter(truths, preds, alpha=0.15, s=4,
                   color=color)
        lims = [min(truths.min(), preds.min()) - 1,
                max(truths.max(), preds.max()) + 1]
        ax.plot(lims, lims, "k--", lw=1.5,
                label="Perfect prediction")
        ax.set_xlabel("True T_internal (C)", fontsize=11)
        ax.set_ylabel("Predicted T_internal (C)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        rmse = float(np.sqrt(np.mean((preds-truths)**2)))
        mae  = float(np.mean(np.abs(preds-truths)))
        r2   = float(1 - np.sum((preds-truths)**2) /
                     np.sum((truths-truths.mean())**2))
        ax.text(0.05, 0.95,
                "RMSE=" + str(round(rmse, 4)) + "C" + chr(10) +
                "MAE =" + str(round(mae,  4)) + "C" + chr(10) +
                "R2  =" + str(round(r2,   4)),
                transform=ax.transAxes, fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round",
                          facecolor="white", alpha=0.85))

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: " + path)


# -- Figure 3: Error distribution (all 3) ------------------

def plot_error_dist_3(b_preds, b_truths,
                       g_preds, g_truths,
                       p_preds, p_truths, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Prediction Error Distribution -- Test Set",
        fontsize=13, fontweight="bold")

    for ax, preds, truths, title, color in [
        (axes[0], b_preds, b_truths, "Baseline LSTM",  "#2563eb"),
        (axes[1], g_preds, g_truths, "GRU",             "#16a34a"),
        (axes[2], p_preds, p_truths, "LSTM + PINN",    "#dc2626"),
    ]:
        errors = preds - truths
        ax.hist(errors, bins=60, color=color,
                alpha=0.7, edgecolor="white")
        ax.axvline(0,    color="black",   lw=1.5)
        ax.axvline( 0.5, color="#f59e0b", lw=1.0,
                   ls="--", alpha=0.8, label="+/-0.5C")
        ax.axvline(-0.5, color="#f59e0b", lw=1.0,
                   ls="--", alpha=0.8)
        rmse     = float(np.sqrt(np.mean(errors**2)))
        mae      = float(np.mean(np.abs(errors)))
        within_h = float(np.mean(np.abs(errors) <= 0.5) * 100)
        within_1 = float(np.mean(np.abs(errors) <= 1.0) * 100)
        ax.set_xlabel("Prediction Error (C)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.text(0.05, 0.95,
                "RMSE=" + str(round(rmse, 4)) + "C" + chr(10) +
                "MAE =" + str(round(mae,  4)) + "C" + chr(10) +
                "W0.5C=" + str(round(within_h, 1)) + "%" + chr(10) +
                "W1.0C=" + str(round(within_1, 1)) + "%",
                transform=ax.transAxes, fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round",
                          facecolor="white", alpha=0.85))

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: " + path)


# -- Figure 4: RMSE by profile (all 3) ---------------------

def plot_rmse_by_profile_3(b_by_prof, g_by_prof,
                            p_by_prof, path):
    profiles = b_by_prof["profile"].tolist()
    x = np.arange(len(profiles))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - w, b_by_prof["RMSE"].tolist(), w,
                   label="Baseline LSTM", color="#2563eb", alpha=0.85)
    bars2 = ax.bar(x,     g_by_prof["RMSE"].tolist(), w,
                   label="GRU",           color="#16a34a", alpha=0.85)
    bars3 = ax.bar(x + w, p_by_prof["RMSE"].tolist(), w,
                   label="LSTM + PINN",  color="#dc2626", alpha=0.85)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.001,
                    str(round(h, 4)),
                    ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Drive Cycle Profile", fontsize=12)
    ax.set_ylabel("RMSE (C)", fontsize=12)
    ax.set_title(
        "RMSE by Drive Cycle Profile -- All Three Models\n"
        "(lower is better)",
        fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(profiles, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_ylim(0, max(
        b_by_prof["RMSE"].max(),
        g_by_prof["RMSE"].max(),
        p_by_prof["RMSE"].max()) * 1.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: " + path)


# -- Figure 5: RMSE by temperature (all 3) -----------------

def plot_rmse_by_temperature_3(b_by_temp, g_by_temp,
                                p_by_temp, path):
    temps = b_by_temp["T_ambient"].tolist()
    x = np.arange(len(temps))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - w, b_by_temp["RMSE"].tolist(), w,
                   label="Baseline LSTM", color="#2563eb", alpha=0.85)
    bars2 = ax.bar(x,     g_by_temp["RMSE"].tolist(), w,
                   label="GRU",           color="#16a34a", alpha=0.85)
    bars3 = ax.bar(x + w, p_by_temp["RMSE"].tolist(), w,
                   label="LSTM + PINN",  color="#dc2626", alpha=0.85)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.001,
                    str(round(h, 4)),
                    ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Ambient Temperature", fontsize=12)
    ax.set_ylabel("RMSE (C)", fontsize=12)
    ax.set_title(
        "RMSE by Ambient Temperature -- All Three Models\n"
        "(shows generalisation across cold, normal, hot conditions)",
        fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(temps, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_ylim(0, max(
        b_by_temp["RMSE"].max(),
        g_by_temp["RMSE"].max(),
        p_by_temp["RMSE"].max()) * 1.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: " + path)


# -- Figure 6: RMSE heatmap (all 3) ------------------------

def plot_heatmap_3(b_preds, b_truths,
                   g_preds, g_truths,
                   p_preds, p_truths, info, path):
    profiles = sorted(info["profile"].unique())
    temps    = sorted(info["T_amb_label"].unique())

    grids = {}
    for name, preds, truths in [
        ("Baseline LSTM", b_preds, b_truths),
        ("GRU",           g_preds, g_truths),
        ("LSTM+PINN",     p_preds, p_truths),
    ]:
        grid = np.zeros((len(profiles), len(temps)))
        for i, prof in enumerate(profiles):
            for j, temp in enumerate(temps):
                mask = ((info["profile"].values == prof) &
                        (info["T_amb_label"].values == temp))
                if mask.sum() < 5:
                    grid[i, j] = np.nan
                else:
                    grid[i, j] = float(np.sqrt(np.mean(
                        (preds[mask] - truths[mask])**2)))
        grids[name] = grid

    vmin = min(np.nanmin(g) for g in grids.values())
    vmax = max(np.nanmax(g) for g in grids.values())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "RMSE Heatmap: Drive Cycle x Ambient Temperature\n"
        "(green = accurate, red = less accurate)",
        fontsize=13, fontweight="bold")

    for ax, (name, grid) in zip(axes, grids.items()):
        im = ax.imshow(grid, cmap="RdYlGn_r",
                       vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(temps)))
        ax.set_yticks(range(len(profiles)))
        ax.set_xticklabels(temps, fontsize=10)
        ax.set_yticklabels(profiles, fontsize=9)
        ax.set_xlabel("Ambient Temp", fontsize=10)
        ax.set_ylabel("Drive Cycle", fontsize=10)
        ax.set_title(name, fontsize=12)
        for i in range(len(profiles)):
            for j in range(len(temps)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, str(round(grid[i, j], 3)),
                            ha="center", va="center",
                            fontsize=9, fontweight="bold",
                            color="black")
        plt.colorbar(im, ax=ax, label="RMSE (C)")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: " + path)


# -- Save full report --------------------------------------

def save_report(b_metrics, g_metrics, p_metrics,
                b_by_prof, g_by_prof, p_by_prof,
                b_by_temp, g_by_temp, p_by_temp):
    path = os.path.join(VALID_FOLDER, "validation_report.txt")
    with open(path, "w") as f:
        f.write("Battery Model Validation Report -- Paper 2\n")
        f.write("CALB L148N58A Real Experimental Data\n")
        f.write("=" * 58 + "\n\n")

        f.write("1. OVERALL PERFORMANCE (Test Set)\n")
        f.write("-" * 58 + "\n")
        f.write("Metric        Baseline LSTM   GRU             LSTM+PINN\n")
        f.write("RMSE (C)      " +
                str(round(b_metrics["RMSE"], 4)).ljust(16) +
                str(round(g_metrics["RMSE"], 4)).ljust(16) +
                str(round(p_metrics["RMSE"], 4)) + "\n")
        f.write("MAE  (C)      " +
                str(round(b_metrics["MAE"],  4)).ljust(16) +
                str(round(g_metrics["MAE"],  4)).ljust(16) +
                str(round(p_metrics["MAE"],  4)) + "\n")
        f.write("R-squared     " +
                str(round(b_metrics["R2"],   5)).ljust(16) +
                str(round(g_metrics["R2"],   5)).ljust(16) +
                str(round(p_metrics["R2"],   5)) + "\n")
        f.write("Within 0.5C   " +
                str(round(b_metrics["within_h"], 1)).ljust(15) + "%" +
                "  " + str(round(g_metrics["within_h"], 1)).ljust(14) + "%" +
                "  " + str(round(p_metrics["within_h"], 1)) + "%\n")
        f.write("Within 1.0C   " +
                str(round(b_metrics["within_1"], 1)).ljust(15) + "%" +
                "  " + str(round(g_metrics["within_1"], 1)).ljust(14) + "%" +
                "  " + str(round(p_metrics["within_1"], 1)) + "%\n\n")

        f.write("2. BY DRIVE CYCLE PROFILE\n")
        f.write("-" * 58 + "\n")
        f.write("Baseline LSTM:\n")
        f.write(b_by_prof.to_string(index=False) + "\n\n")
        f.write("GRU:\n")
        f.write(g_by_prof.to_string(index=False) + "\n\n")
        f.write("LSTM+PINN:\n")
        f.write(p_by_prof.to_string(index=False) + "\n\n")

        f.write("3. BY AMBIENT TEMPERATURE\n")
        f.write("-" * 58 + "\n")
        f.write("Baseline LSTM:\n")
        f.write(b_by_temp.to_string(index=False) + "\n\n")
        f.write("GRU:\n")
        f.write(g_by_temp.to_string(index=False) + "\n\n")
        f.write("LSTM+PINN:\n")
        f.write(p_by_temp.to_string(index=False) + "\n\n")

        f.write("4. COMPARISON AGAINST PUBLISHED WORK\n")
        f.write("-" * 58 + "\n")
        f.write("Method                  RMSE (C)   MAE (C)\n")
        f.write("Wang et al. 2021        0.850      --\n")
        f.write("Karnehm 2024 LSTM       0.854      0.550\n")
        f.write("Karnehm 2024 KAN        0.751      0.469\n")
        f.write("Shen et al. 2025        0.570      --\n")
        f.write("Our GRU                 " +
                str(round(g_metrics["RMSE"], 4)).ljust(11) +
                str(round(g_metrics["MAE"],  4)) + "\n")
        f.write("Our LSTM+PINN           " +
                str(round(p_metrics["RMSE"], 4)).ljust(11) +
                str(round(p_metrics["MAE"],  4)) + "\n")
        f.write("Our Baseline LSTM       " +
                str(round(b_metrics["RMSE"], 4)).ljust(11) +
                str(round(b_metrics["MAE"],  4)) + "\n")

    print("  Saved: " + path)


# -- Main --------------------------------------------------

def main():
    print("\n" + "=" * 58)
    print("  Battery Validation -- All Three Models")
    print("  CALB Real Experimental Data")
    print("=" * 58)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("  Device: " + DEVICE)

    # Load data
    train_df, test_df, scaler_X, scaler_y = load_data()

    # Load all three models
    print("\n[Loading models...]")
    baseline = BatteryLSTM(input_size=12).to(DEVICE)
    pinn     = BatteryLSTM(input_size=12).to(DEVICE)
    gru      = BatteryGRU(input_size=12).to(DEVICE)

    b_path = os.path.join(RESULTS_LSTM, "p2_baseline_lstm.pt")
    p_path = os.path.join(RESULTS_LSTM, "p2_pinn_lstm.pt")
    g_path = os.path.join(RESULTS_GRU,  "gru_model.pt")

    for path, name in [(b_path, "Baseline LSTM"),
                       (p_path, "LSTM+PINN"),
                       (g_path, "GRU")]:
        if not os.path.exists(path):
            print("  ERROR: " + path + " not found.")
            print("  Run the training scripts first.")
            return
        print("  Loaded: " + name)

    baseline.load_state_dict(torch.load(b_path, map_location=DEVICE))
    pinn.load_state_dict(    torch.load(p_path, map_location=DEVICE))
    gru.load_state_dict(     torch.load(g_path, map_location=DEVICE))

    # Get predictions for all three models
    print("\n[Getting predictions...]")
    b_preds, b_truths, info = get_predictions(
        baseline, test_df, scaler_X, scaler_y, DEVICE)
    p_preds, p_truths, _    = get_predictions(
        pinn,     test_df, scaler_X, scaler_y, DEVICE)
    g_preds, g_truths, _    = get_predictions(
        gru,      test_df, scaler_X, scaler_y, DEVICE)
    print("  Samples: " + str(len(b_preds)))

    # Metrics
    print("\n[Overall metrics...]")
    b_metrics = compute_metrics(b_preds, b_truths, "Baseline LSTM")
    g_metrics = compute_metrics(g_preds, g_truths, "GRU")
    p_metrics = compute_metrics(p_preds, p_truths, "LSTM + PINN")

    print("\n[By drive cycle profile...]")
    b_by_prof = metrics_by_profile(b_preds, b_truths, info, "Baseline LSTM")
    g_by_prof = metrics_by_profile(g_preds, g_truths, info, "GRU")
    p_by_prof = metrics_by_profile(p_preds, p_truths, info, "LSTM+PINN")

    print("\n[By ambient temperature...]")
    b_by_temp = metrics_by_temperature(b_preds, b_truths, info, "Baseline LSTM")
    g_by_temp = metrics_by_temperature(g_preds, g_truths, info, "GRU")
    p_by_temp = metrics_by_temperature(p_preds, p_truths, info, "LSTM+PINN")

    # Generate all figures
    print("\n[Generating figures...]")

    plot_time_series_all_models(
        baseline, pinn, gru,
        test_df, scaler_X, scaler_y, DEVICE,
        os.path.join(VALID_FOLDER, "time_series_all_models.png"))

    plot_scatter_3(
        b_preds, b_truths, g_preds, g_truths, p_preds, p_truths,
        os.path.join(VALID_FOLDER, "scatter_plot.png"))

    plot_error_dist_3(
        b_preds, b_truths, g_preds, g_truths, p_preds, p_truths,
        os.path.join(VALID_FOLDER, "error_distribution.png"))

    plot_rmse_by_profile_3(
        b_by_prof, g_by_prof, p_by_prof,
        os.path.join(VALID_FOLDER, "rmse_by_profile.png"))

    plot_rmse_by_temperature_3(
        b_by_temp, g_by_temp, p_by_temp,
        os.path.join(VALID_FOLDER, "rmse_by_temperature.png"))

    plot_heatmap_3(
        b_preds, b_truths, g_preds, g_truths, p_preds, p_truths,
        info, os.path.join(VALID_FOLDER, "rmse_heatmap.png"))

    # Save report
    print("\n[Saving report...]")
    save_report(b_metrics, g_metrics, p_metrics,
                b_by_prof, g_by_prof, p_by_prof,
                b_by_temp, g_by_temp, p_by_temp)

    print("\n" + "=" * 58)
    print("  Validation Complete!")
    print("  Files saved to: " + os.path.abspath(VALID_FOLDER))
    print()
    print("  Figures:")
    print("    time_series_all_models.png  -- key paper figure")
    print("    scatter_plot.png            -- paper Figure 1")
    print("    error_distribution.png      -- paper Figure 2")
    print("    rmse_by_profile.png         -- paper Figure 3")
    print("    rmse_by_temperature.png     -- paper Figure 4")
    print("    rmse_heatmap.png            -- paper Figure 5")
    print("    validation_report.txt       -- all numbers")
    print("=" * 58)


if __name__ == "__main__":
    main()
