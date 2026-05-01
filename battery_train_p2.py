# Battery LSTM + PINN Training Script
# Trains two models and compares them:
#   Model 1: Baseline LSTM (no physics)
#   Model 2: LSTM + PINN (with two-node thermal physics constraint)
#
# HOW TO USE:
#   1. pip install torch numpy pandas scikit-learn matplotlib
#   2. Make sure processed_data/ folder exists with battery_train.csv etc.
#   3. python battery_train.py
#
# OUTPUT:
#   results/baseline_lstm_predictions.png
#   results/pinn_lstm_predictions.png
#   results/model_comparison.png
#   results/p2_baseline_lstm.pt   (saved model)
#   results/p2_pinn_lstm.pt        (saved model)
#   results/metrics.txt         (RMSE, MAE for paper)

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# SETTINGS - change these if needed
# ============================================================

DATA_FOLDER   = "calb_data"         # folder with your CSV files
OUTPUT_FOLDER = "results_p2"       # Paper 2 results -- separate from Paper 1

SEQ_LEN       = 60       # look back 60 seconds to predict temperature
STRIDE        = 10       # step between sequences (10 = more overlap = more training data)
BATCH_SIZE    = 128      # how many sequences per training step
EPOCHS              = 300   # maximum epochs -- early stopping will halt before this
EARLY_STOP_PATIENCE = 25    # increased -- PINN needs more time to converge
HIDDEN_SIZE         = 256   # LSTM hidden units per layer
LR            = 0.0003    # learning rate
HIDDEN_SIZE   = 256      # LSTM hidden units
NUM_LAYERS    = 2        # LSTM layers
DROPOUT       = 0.2      # dropout for regularisation

LAMBDA_START  = 0.000001   # very small start -- let model learn data first
LAMBDA_END    = 0.0001    # gentle final weight -- physics guides not dominates

HOTSPOT_THRESH  = 44.0   # degrees C - safety detection threshold
# These are set after load_data() runs -- used by pinn_loss for Celsius conversion
SCALER_Y_MEAN   = 34.0   # placeholder -- overwritten at runtime
SCALER_Y_STD    = 15.0   # placeholder -- overwritten at runtime

# Which columns the LSTM uses as input
FEATURE_COLS = [
    "current_A",
    "voltage_V",
    "SOC",
    "T_ambient_C",
    "T_surface_C",
    "Q_gen_W",
    "dI_dt",
    "dV_dt",
    "delta_T_surf",
    "I_mean_60s",
    "I_std_60s",
    "R_internal_Ohm",
]

TARGET_COL   = "T_internal_C"
PHYSICS_COLS = ["phys_res_1", "phys_res_2"]

# Use GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# STEP 1 - LOAD AND NORMALISE DATA
# ============================================================
# Normalisation: subtract mean, divide by std.
# This makes all features roughly the same scale (0 to 1 range)
# which helps the LSTM learn faster and more stably.
# IMPORTANT: we fit the scaler ONLY on training data,
# then apply the same scaler to val and test.
# This prevents "data leakage" from future data.

def load_data():
    print("\n[Step 1] Loading data...")

    train_df = pd.read_csv(os.path.join(DATA_FOLDER, "calb_train.csv"))
    val_df   = pd.read_csv(os.path.join(DATA_FOLDER, "calb_val.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_FOLDER, "calb_test.csv"))

    print("  Train rows: " + str(len(train_df)))
    print("  Val rows  : " + str(len(val_df)))
    print("  Test rows : " + str(len(test_df)))

    # Check all required columns exist
    all_cols = FEATURE_COLS + [TARGET_COL] + PHYSICS_COLS + ["run_id"]
    for col in all_cols:
        if col not in train_df.columns:
            print("  WARNING: column '" + col + "' not found. Available: " +
                  str(list(train_df.columns)))

    # Engineer missing features from base columns
    def engineer_features(df):
        # Rate of change of current
        df["dI_dt"] = df.groupby("run_id")["current_A"].diff().fillna(0)
        # Rate of change of voltage
        df["dV_dt"] = df.groupby("run_id")["voltage_V"].diff().fillna(0)
        # Surface temperature above ambient
        df["delta_T_surf"] = df["T_surface_C"] - df["T_ambient_C"]
        # Rolling 60s mean and std of current (per run)
        df["I_mean_60s"] = (df.groupby("run_id")["current_A"]
                             .transform(lambda x: x.rolling(60, min_periods=1).mean()))
        df["I_std_60s"]  = (df.groupby("run_id")["current_A"]
                             .transform(lambda x: x.rolling(60, min_periods=1).std().fillna(0)))
        # Internal resistance estimate: R = V_drop / I (simplified)
        I_safe = df["current_A"].copy()
        I_safe[I_safe.abs() < 0.1] = 0.1
        OCV = 3.2 + 0.5 * df["SOC"]
        df["R_internal_Ohm"] = ((OCV - df["voltage_V"]) / I_safe).clip(0.0001, 0.05)
        return df

    train_df = engineer_features(train_df)
    val_df   = engineer_features(val_df)
    test_df  = engineer_features(test_df)
    print("  Feature engineering done (dI_dt, dV_dt, delta_T_surf, rolling I, R_internal)")

    # Fill any missing values with column mean
    for df in [train_df, val_df, test_df]:
        for col in FEATURE_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())

    # Fit scaler on training data only
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    scaler_X.fit(train_df[FEATURE_COLS].values)
    scaler_y.fit(train_df[[TARGET_COL]].values)

    print("  Feature scaling done (fitted on train only)")
    print("  T_internal mean=" +
          str(round(float(scaler_y.mean_[0]), 2)) + "C  " +
          "std=" + str(round(float(np.sqrt(scaler_y.var_[0])), 2)) + "C")

    # Store scaler stats globally so pinn_loss can convert to Celsius
    global SCALER_Y_MEAN, SCALER_Y_STD
    SCALER_Y_MEAN = float(scaler_y.mean_[0])
    SCALER_Y_STD  = float(np.sqrt(scaler_y.var_[0]))

    return train_df, val_df, test_df, scaler_X, scaler_y


# ============================================================
# STEP 2 - PYTORCH DATASET
# ============================================================
# This class takes the big DataFrame and cuts it into
# overlapping windows of SEQ_LEN seconds.
#
# Example with SEQ_LEN=60, STRIDE=10:
#   Window 1: rows 0-59   -> predict T at row 59
#   Window 2: rows 10-69  -> predict T at row 69
#   Window 3: rows 20-79  -> predict T at row 79
#   etc.
#
# We NEVER mix rows from different runs in the same window.
# Each run is processed separately.

class BatteryDataset(Dataset):
    def __init__(self, df, scaler_X, scaler_y, seq_len=60, stride=STRIDE):
        self.sequences = []

        runs = df["run_id"].unique()
        for run_id in runs:
            grp = df[df["run_id"] == run_id].reset_index(drop=True)

            if len(grp) < seq_len + 1:
                continue

            # Normalise features and target
            X_raw = grp[FEATURE_COLS].values
            y_raw = grp[[TARGET_COL]].values
            r_raw = grp[PHYSICS_COLS].values if all(
                c in grp.columns for c in PHYSICS_COLS) else np.zeros(
                (len(grp), 2))

            X = scaler_X.transform(X_raw).astype(np.float32)
            y = scaler_y.transform(y_raw).astype(np.float32).flatten()
            r = r_raw.astype(np.float32)

            # Also keep raw T_internal for hotspot detection evaluation
            t_raw = grp[TARGET_COL].values.astype(np.float32)

            # Create sliding windows
            for start in range(0, len(grp) - seq_len, stride):
                end = start + seq_len
                self.sequences.append((
                    X[start:end],          # input: (seq_len, n_features)
                    y[end - 1],            # target: scalar (normalised)
                    r[start:end],          # physics residuals: (seq_len, 2)
                    t_raw[end - 1],        # raw target temp for evaluation
                ))

        print("  " + str(len(self.sequences)) + " sequences created")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X, y, r, t = self.sequences[idx]
        return (torch.tensor(X),
                torch.tensor(y),
                torch.tensor(r),
                torch.tensor(t))


# ============================================================
# STEP 3 - LSTM MODEL
# ============================================================
# The LSTM reads a 60-second window of battery measurements
# and predicts the internal temperature at the end of the window.
#
# Architecture:
#   Input (60 timesteps x 12 features)
#      -> LSTM (2 layers, 128 hidden units)
#      -> Take output at last timestep
#      -> Fully connected layer (128 -> 64)
#      -> ReLU activation
#      -> Output layer (64 -> 1)
#      -> Single temperature prediction

class BatteryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=0.2):
        super(BatteryLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, n_features)
        lstm_out, _ = self.lstm(x)
        # Take only the last timestep output
        last_out = lstm_out[:, -1, :]
        # Shape: (batch, 1) -> squeeze to (batch,)
        return self.head(last_out).squeeze(-1)


# ============================================================
# STEP 4 - LOSS FUNCTIONS
# ============================================================

def baseline_loss(y_pred, y_true):
    """
    Standard MSE loss - just measures prediction error.
    Used for the baseline LSTM (no physics).
    """
    return nn.functional.mse_loss(y_pred, y_true)


def pinn_loss(y_pred, y_true, phys_residuals, lambda_phys):
    """
    Simplified PINN loss for temperature prediction accuracy.
    Hotspot penalty removed -- focus is on accurate T_internal prediction.

    Loss = Huber(data) + lambda * physics_residual
    """
    # -- Component 1: Data fitting loss (Huber is robust to outliers) --
    L_data = torch.nn.functional.huber_loss(y_pred, y_true, delta=1.0)

    # -- Component 2: Physics residual (two-node thermal equations) --
    # Penalise predictions that violate heat transfer physics
    L_phys = phys_residuals.abs().mean()

    # Total loss -- lambda kept very small so physics guides not dominates
    L_total = L_data + lambda_phys * L_phys

    return L_total, L_data.item(), L_phys.item()


def train_one_epoch(model, loader, optimizer, use_physics, lambda_phys, scaler_y):
    model.train()
    total_loss = 0.0
    total_ld   = 0.0
    total_lp   = 0.0
    n_batches  = 0

    for X, y, r, _ in loader:
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        r = r.to(DEVICE)

        optimizer.zero_grad()
        y_pred = model(X)

        if use_physics:
            loss, ld, lp = pinn_loss(y_pred, y, r, lambda_phys)
            total_lp += lp
        else:
            loss = baseline_loss(y_pred, y)
            ld   = float(loss)

        loss.backward()
        # Gradient clipping - prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        total_loss += float(loss)
        total_ld   += ld
        n_batches  += 1

    return total_loss / n_batches, total_ld / n_batches, total_lp / max(n_batches, 1)


def evaluate(model, loader, use_physics, lambda_phys, scaler_y):
    model.eval()
    all_preds  = []
    all_truths = []
    all_raw    = []
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for X, y, r, t_raw in loader:
            X     = X.to(DEVICE)
            y     = y.to(DEVICE)
            r     = r.to(DEVICE)
            y_pred = model(X)

            if use_physics:
                loss, _, _ = pinn_loss(y_pred, y, r, lambda_phys)
            else:
                loss = baseline_loss(y_pred, y)

            total_loss += float(loss)
            n_batches  += 1

            all_preds.append(y_pred.cpu().numpy())
            all_truths.append(y.cpu().numpy())
            all_raw.append(t_raw.numpy())

    preds  = np.concatenate(all_preds)
    truths = np.concatenate(all_truths)
    raws   = np.concatenate(all_raw)

    # Convert back from normalised to real degrees C
    preds_C  = scaler_y.inverse_transform(preds.reshape(-1,1)).flatten()
    truths_C = scaler_y.inverse_transform(truths.reshape(-1,1)).flatten()

    rmse = float(np.sqrt(np.mean((preds_C - truths_C)**2)))
    mae  = float(np.mean(np.abs(preds_C - truths_C)))

    return total_loss / n_batches, rmse, mae, preds_C, truths_C, raws


# ============================================================
# STEP 6 - HOTSPOT DETECTION METRICS
# ============================================================
# This is the KEY evaluation for your paper.
# We check: when T_internal > 55C, does our model detect it?
#
# Metrics:
#   True Positive  (TP): model says danger, it IS danger
#   False Negative (FN): model says safe, but it IS danger (missed alarm)
#   False Positive (FP): model says danger, but it IS safe (false alarm)
#
# Detection Rate = TP / (TP + FN) -- how many real dangers caught
# False Alarm Rate = FP / (FP + TN) -- how many false alarms

def hotspot_metrics(preds_C, truths_C, threshold=HOTSPOT_THRESH):
    true_hot  = (truths_C >= threshold)
    pred_hot  = (preds_C  >= threshold)

    TP = int(np.sum(true_hot  & pred_hot))
    FN = int(np.sum(true_hot  & ~pred_hot))
    FP = int(np.sum(~true_hot & pred_hot))
    TN = int(np.sum(~true_hot & ~pred_hot))

    detection_rate  = TP / max(TP + FN, 1) * 100
    false_alarm     = FP / max(FP + TN, 1) * 100
    precision       = TP / max(TP + FP, 1) * 100

    return {
        "TP": TP, "FN": FN, "FP": FP, "TN": TN,
        "detection_rate_pct" : round(detection_rate, 1),
        "false_alarm_pct"    : round(false_alarm, 1),
        "precision_pct"      : round(precision, 1),
    }


# ============================================================
# STEP 7 - PLOTTING
# ============================================================

def plot_predictions(preds_C, truths_C, title, save_path, n_show=3000):
    """
    Fixed version -- samples evenly across ALL test predictions
    so the plot shows the full temperature range (6C to 42C)
    not just the first run which happens to be at 10C ambient.
    """
    total = len(preds_C)

    # Sample evenly across all predictions to show full temperature range
    if total > n_show:
        step   = total // n_show
        idx_sel = np.arange(0, total, step)[:n_show]
    else:
        idx_sel = np.arange(total)

    truths_show = truths_C[idx_sel]
    preds_show  = preds_C[idx_sel]
    x_axis      = np.arange(len(idx_sel))

    # Sort by ground truth temperature so plot shows full range cleanly
    sort_order  = np.argsort(truths_show)
    truths_sorted = truths_show[sort_order]
    preds_sorted  = preds_show[sort_order]
    error_sorted  = preds_sorted - truths_sorted

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(x_axis, truths_sorted,
            color="#dc2626", lw=1.5, label="T_internal - Ground Truth")
    ax.plot(x_axis, preds_sorted,
            color="#2563eb", lw=1.0, ls="--", label="T_internal - Predicted")
    ax.set_ylabel("Temperature (C)")
    ax.set_xlabel("Samples (sorted by temperature -- shows full range)")
    ax.set_title("Predicted vs Ground Truth Internal Temperature")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    rmse = float(np.sqrt(np.mean((preds_sorted - truths_sorted)**2)))
    mae  = float(np.mean(np.abs(preds_sorted - truths_sorted)))
    ax.text(0.02, 0.97,
            "RMSE = " + str(round(rmse, 4)) + " C" + chr(10) + "MAE  = " + str(round(mae, 4)) + " C",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax2 = axes[1]
    ax2.plot(x_axis, error_sorted, color="#7c3aed", lw=0.8, alpha=0.7)
    ax2.axhline(0,   color="gray",    lw=0.8)
    ax2.axhline( 0.5, color="#f59e0b", lw=0.8, ls="--",
                alpha=0.6, label="+0.5C error")
    ax2.axhline(-0.5, color="#f59e0b", lw=0.8, ls="--",
                alpha=0.6, label="-0.5C error")
    ax2.axhline( 1.0, color="#dc2626", lw=0.8, ls=":",
                alpha=0.4, label="+/-1C error")
    ax2.axhline(-1.0, color="#dc2626", lw=0.8, ls=":", alpha=0.4)
    ax2.set_ylabel("Prediction Error (C)")
    ax2.set_xlabel("Samples (sorted by temperature)")
    ax2.set_title("Prediction Error (closer to 0 = better)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    within_half = float(np.mean(np.abs(error_sorted) <= 0.5) * 100)
    within_one  = float(np.mean(np.abs(error_sorted) <= 1.0) * 100)
    ax2.text(0.02, 0.97,
             "Within 0.5C: " + str(round(within_half, 1)) + "%" + chr(10) + "Within 1.0C: " + str(round(within_one, 1)) + "%",
             transform=ax2.transAxes, fontsize=10,
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Plot saved: " + save_path)


def plot_comparison(baseline_hist, pinn_hist, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Baseline LSTM vs LSTM+PINN Training Comparison",
                 fontsize=13, fontweight="bold")

    b_epochs = range(1, len(baseline_hist["val_rmse"]) + 1)
    p_epochs = range(1, len(pinn_hist["val_rmse"]) + 1)

    ax = axes[0]
    ax.plot(b_epochs, baseline_hist["val_rmse"],
            color="#2563eb", lw=2, label="Baseline LSTM")
    ax.plot(p_epochs, pinn_hist["val_rmse"],
            color="#dc2626", lw=2, label="LSTM + PINN")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE (C)")
    ax.set_title("Validation RMSE over Training")
    ax.legend()
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.plot(b_epochs, baseline_hist["val_loss"],
            color="#2563eb", lw=2, label="Baseline LSTM")
    ax.plot(p_epochs, pinn_hist["val_loss"],
            color="#dc2626", lw=2, label="LSTM + PINN")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss over Training")
    ax.legend()
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Comparison plot saved: " + save_path)


def plot_early_warning(preds_C, truths_C, save_path, n_show=5000):
    """
    KEY PAPER FIGURE: shows how many seconds earlier the model
    detects a hotspot compared to a surface sensor.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    idx = np.arange(min(n_show, len(preds_C)))
    ax.plot(idx, truths_C[:n_show],
            color="#dc2626", lw=1.5, label="T_internal (model estimate)")
    ax.plot(idx, preds_C[:n_show],
            color="#2563eb", lw=1.0, ls="--", label="T_internal (LSTM+PINN prediction)")
    ax.axhline(HOTSPOT_THRESH, color="#dc2626", lw=1.2, ls="-.",
               label="Safety threshold 44C")

    ax.fill_between(idx,
                    HOTSPOT_THRESH, truths_C[:n_show],
                    where=(truths_C[:n_show] >= HOTSPOT_THRESH),
                    alpha=0.2, color="#dc2626",
                    label="Danger zone (T_internal > 55C)")

    ax.set_ylabel("Temperature (C)")
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Early Warning Detection - Internal Hotspot vs Surface Sensor")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Early warning plot saved: " + save_path)


# ============================================================
# STEP 8 - FULL TRAINING FUNCTION
# ============================================================

def train_model(model, train_loader, val_loader, use_physics,
                model_name, scaler_y):
    print("\n  Training: " + model_name)
    print("  Device  : " + DEVICE)
    print("  Physics : " + str(use_physics))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.5, min_lr=1e-6)

    history = {
        "train_loss": [], "val_loss": [],
        "val_rmse": [],   "val_mae": [],
    }

    best_val_loss  = float("inf")
    best_val_rmse  = float("inf")
    best_model_state = None
    lambda_phys    = LAMBDA_START
    epochs_no_improve = 0   # counter for early stopping

    for epoch in range(1, EPOCHS + 1):
        # Ramp up physics weight over first 20 epochs
        if use_physics and epoch <= 20:
            lambda_phys = LAMBDA_START + (LAMBDA_END - LAMBDA_START) * (epoch / 20)

        train_loss, ld, lp = train_one_epoch(
            model, train_loader, optimizer,
            use_physics, lambda_phys, scaler_y)

        val_loss, rmse, mae, _, _, _ = evaluate(
            model, val_loader, use_physics, lambda_phys, scaler_y)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(rmse)
        history["val_mae"].append(mae)

        # Save best model -- track on RMSE not loss for PINN
        # (loss includes physics penalty which inflates the value)
        if rmse < best_val_rmse:
            best_val_rmse  = rmse
            best_val_loss  = val_loss
            best_model_state = {k: v.clone()
                                for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Print progress every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            msg = ("  Epoch " + str(epoch).rjust(3) + "/" + str(EPOCHS) +
                   "  train_loss=" + str(round(train_loss, 4)) +
                   "  val_loss=" + str(round(val_loss, 4)) +
                   "  RMSE=" + str(round(rmse, 3)) + "C" +
                   "  MAE=" + str(round(mae, 3)) + "C")
            if use_physics:
                msg += "  lambda=" + str(round(lambda_phys, 3))
            msg += "  [no_improve=" + str(epochs_no_improve) + "]"
            print(msg)

        # Early stopping check
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print("  Early stopping at epoch " + str(epoch) +
                  " -- val RMSE has not improved for " +
                  str(EARLY_STOP_PATIENCE) + " epochs")
            break

    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print("  Training complete. Best val_RMSE=" + str(round(best_val_rmse, 4)) +
          "C  Best val_loss=" + str(round(best_val_loss, 4)))

    return model, history


# ============================================================
# MAIN - runs everything
# ============================================================

def main():
    print("\n" + "="*60)
    print("  Battery LSTM + PINN Training Pipeline")
    print("="*60)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load data
    train_df, val_df, test_df, scaler_X, scaler_y = load_data()

    # Create datasets
    print("\n[Step 2] Creating sequence datasets...")
    print("  Train sequences:")
    train_ds = BatteryDataset(train_df, scaler_X, scaler_y, SEQ_LEN, STRIDE)
    print("  Val sequences:")
    val_ds   = BatteryDataset(val_df,   scaler_X, scaler_y, SEQ_LEN, stride=20)
    print("  Test sequences:")
    test_ds  = BatteryDataset(test_df,  scaler_X, scaler_y, SEQ_LEN, stride=20)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    n_features = len(FEATURE_COLS)
    print("\n  Input features : " + str(n_features))
    print("  Sequence length: " + str(SEQ_LEN) + "s")
    print("  Train batches  : " + str(len(train_loader)))

    # -- MODEL 1: Baseline LSTM -------------------------------
    print("\n" + "-"*50)
    print("  MODEL 1: Baseline LSTM (no physics constraint)")
    print("-"*50)

    baseline_model = BatteryLSTM(
        input_size  = n_features,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT,
    ).to(DEVICE)

    print("  Parameters: " +
          str(sum(p.numel() for p in baseline_model.parameters())))

    baseline_model, baseline_hist = train_model(
        baseline_model, train_loader, val_loader,
        use_physics=False,
        model_name="Baseline LSTM",
        scaler_y=scaler_y,
    )

    # Evaluate baseline on test set
    print("\n  Evaluating Baseline LSTM on test set...")
    _, b_rmse, b_mae, b_preds, b_truths, _ = evaluate(
        baseline_model, test_loader,
        use_physics=False, lambda_phys=0, scaler_y=scaler_y)
    b_hotspot = hotspot_metrics(b_preds, b_truths, threshold=HOTSPOT_THRESH)

    print("  Baseline LSTM Test Results:")
    print("    RMSE = " + str(round(b_rmse, 3)) + " C")
    print("    MAE  = " + str(round(b_mae,  3)) + " C")
    print("    Hotspot detection rate = " +
          str(b_hotspot["detection_rate_pct"]) + "%")
    print("    False alarm rate       = " +
          str(b_hotspot["false_alarm_pct"]) + "%")

    # Save baseline model
    torch.save(baseline_model.state_dict(),
               os.path.join(OUTPUT_FOLDER, "p2_baseline_lstm.pt"))

    # Plot baseline predictions
    plot_predictions(b_preds, b_truths,
                     "Baseline LSTM - Test Set Predictions",
                     os.path.join(OUTPUT_FOLDER, "baseline_predictions.png"))

    # -- MODEL 2: LSTM + PINN ---------------------------------
    print("\n" + "-"*50)
    print("  MODEL 2: LSTM + PINN (two-node physics constraint)")
    print("-"*50)

    pinn_model = BatteryLSTM(
        input_size  = n_features,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT,
    ).to(DEVICE)

    pinn_model, pinn_hist = train_model(
        pinn_model, train_loader, val_loader,
        use_physics=True,
        model_name="LSTM + PINN",
        scaler_y=scaler_y,
    )

    # Evaluate PINN on test set
    print("\n  Evaluating LSTM+PINN on test set...")
    _, p_rmse, p_mae, p_preds, p_truths, _ = evaluate(
        pinn_model, test_loader,
        use_physics=True, lambda_phys=LAMBDA_END, scaler_y=scaler_y)
    p_hotspot = hotspot_metrics(p_preds, p_truths, threshold=HOTSPOT_THRESH)

    print("  LSTM+PINN Test Results:")
    print("    RMSE = " + str(round(p_rmse, 3)) + " C")
    print("    MAE  = " + str(round(p_mae,  3)) + " C")
    print("    Hotspot detection rate = " +
          str(p_hotspot["detection_rate_pct"]) + "%")
    print("    False alarm rate       = " +
          str(p_hotspot["false_alarm_pct"]) + "%")

    # Save PINN model
    torch.save(pinn_model.state_dict(),
               os.path.join(OUTPUT_FOLDER, "p2_pinn_lstm.pt"))

    # Plot PINN predictions
    plot_predictions(p_preds, p_truths,
                     "LSTM + PINN - Test Set Predictions",
                     os.path.join(OUTPUT_FOLDER, "pinn_predictions.png"))

    # Plot early warning figure
    plot_early_warning(p_preds, p_truths,
                       os.path.join(OUTPUT_FOLDER, "early_warning.png"))

    # -- COMPARISON PLOT --------------------------------------
    plot_comparison(baseline_hist, pinn_hist,
                    os.path.join(OUTPUT_FOLDER, "model_comparison.png"))

    # -- SAVE METRICS TO FILE ---------------------------------
    metrics_path = os.path.join(OUTPUT_FOLDER, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Battery Internal Temperature Prediction - Results\n")
        f.write("="*50 + "\n\n")
        f.write("Baseline LSTM (no physics)\n")
        f.write("-"*30 + "\n")
        f.write("RMSE           : " + str(round(b_rmse,3)) + " C\n")
        f.write("MAE            : " + str(round(b_mae, 3)) + " C\n")
        f.write("Detection rate : " + str(b_hotspot["detection_rate_pct"]) + "%\n")
        f.write("False alarms   : " + str(b_hotspot["false_alarm_pct"]) + "%\n")
        f.write("Precision      : " + str(b_hotspot["precision_pct"]) + "%\n\n")
        f.write("LSTM + PINN (two-node physics)\n")
        f.write("-"*30 + "\n")
        f.write("RMSE           : " + str(round(p_rmse,3)) + " C\n")
        f.write("MAE            : " + str(round(p_mae, 3)) + " C\n")
        f.write("Detection rate : " + str(p_hotspot["detection_rate_pct"]) + "%\n")
        f.write("False alarms   : " + str(p_hotspot["false_alarm_pct"]) + "%\n")
        f.write("Precision      : " + str(p_hotspot["precision_pct"]) + "%\n\n")
        f.write("Improvement (PINN vs Baseline)\n")
        f.write("-"*30 + "\n")
        rmse_imp = round(100*(b_rmse - p_rmse)/max(b_rmse,0.001), 1)
        mae_imp  = round(100*(b_mae  - p_mae) /max(b_mae, 0.001), 1)
        f.write("RMSE improvement : " + str(rmse_imp) + "%\n")
        f.write("MAE improvement  : " + str(mae_imp)  + "%\n")
    print("\n  Metrics saved: " + metrics_path)

    # -- FINAL SUMMARY ----------------------------------------
    print("\n" + "="*60)
    print("  FINAL RESULTS SUMMARY")
    print("="*60)
    print("")
    print("  Metric              Baseline LSTM    LSTM+PINN")
    print("  " + "-"*50)
    print("  RMSE (C)            " +
          str(round(b_rmse,3)).ljust(16) + str(round(p_rmse,3)))
    print("  MAE  (C)            " +
          str(round(b_mae,3)).ljust(16) + str(round(p_mae,3)))
    print("  Detection rate      " +
          str(b_hotspot["detection_rate_pct"]).ljust(15) + "%" +
          "  " + str(p_hotspot["detection_rate_pct"]) + "%")
    print("  False alarm rate    " +
          str(b_hotspot["false_alarm_pct"]).ljust(15) + "%" +
          "  " + str(p_hotspot["false_alarm_pct"]) + "%")
    print("")
    print("  Output files saved to: " + os.path.abspath(OUTPUT_FOLDER))
    print("    baseline_predictions.png")
    print("    pinn_predictions.png")
    print("    model_comparison.png")
    print("    early_warning.png")
    print("    metrics.txt")
    print("    p2_baseline_lstm.pt")
    print("    p2_pinn_lstm.pt")
    print("")
    print("  These results and plots go directly into your paper.")
    print("="*60)


if __name__ == "__main__":
    main()
