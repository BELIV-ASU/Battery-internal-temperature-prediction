# Battery GRU Training Script
# ============================
# Trains a GRU model to predict internal core temperature.
# Used to compare against Baseline LSTM and LSTM+PINN from battery_train_p2.py
#
# Architecture is identical to the LSTM except GRU replaces LSTM cells.
# This gives a fair comparison -- same features, same data, same hyperparameters.
#
# GRU vs LSTM:
#   LSTM has 3 gates (input, forget, output) -- 819,329 parameters
#   GRU  has 2 gates (reset, update)         -- ~615,000 parameters
#   GRU trains faster, uses less memory, often matches LSTM accuracy.
#
# HOW TO USE:
#   Run battery_train_p2.py first to generate calb_data/ folder.
#   Then: python battery_train_gru.py
#
# OUTPUT (saved to results_gru/):
#   gru_predictions.png     -- predicted vs true temperature plot
#   gru_training.png        -- training curve
#   gru_metrics.txt         -- RMSE, MAE for paper comparison table
#   gru_model.pt            -- saved model weights

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
# SETTINGS -- identical to battery_train_p2.py for fair comparison
# ============================================================

DATA_FOLDER   = "calb_data"
OUTPUT_FOLDER = "results_gru"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

SEQ_LEN       = 60
STRIDE        = 10
BATCH_SIZE    = 128
EPOCHS        = 300
PATIENCE      = 25
HIDDEN_SIZE   = 256
NUM_LAYERS    = 2
DROPOUT       = 0.2
LR            = 0.0003

FEATURE_COLS = [
    "current_A", "voltage_V", "SOC", "T_ambient_C",
    "T_surface_C", "Q_gen_W", "dI_dt", "dV_dt",
    "delta_T_surf", "I_mean_60s", "I_std_60s", "R_internal_Ohm",
]
TARGET_COL   = "T_internal_C"
PHYSICS_COLS = ["phys_res_1", "phys_res_2"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# STEP 1 -- LOAD DATA
# ============================================================

def load_data():
    print("\n[Step 1] Loading data...")

    train_df = pd.read_csv(os.path.join(DATA_FOLDER, "calb_train.csv"))
    val_df   = pd.read_csv(os.path.join(DATA_FOLDER, "calb_val.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_FOLDER, "calb_test.csv"))

    print("  Train rows: " + str(len(train_df)))
    print("  Val rows  : " + str(len(val_df)))
    print("  Test rows : " + str(len(test_df)))

    def engineer(df):
        df["dI_dt"]       = df.groupby("run_id")["current_A"].diff().fillna(0)
        df["dV_dt"]       = df.groupby("run_id")["voltage_V"].diff().fillna(0)
        df["delta_T_surf"] = df["T_surface_C"] - df["T_ambient_C"]
        df["I_mean_60s"]  = (df.groupby("run_id")["current_A"]
                               .transform(lambda x: x.rolling(60, min_periods=1).mean()))
        df["I_std_60s"]   = (df.groupby("run_id")["current_A"]
                               .transform(lambda x: x.rolling(60, min_periods=1).std().fillna(0)))
        if "R_internal_Ohm" not in df.columns:
            df["R_internal_Ohm"] = np.clip(
                np.abs(3.7 + 0.5*df["SOC"] - df["voltage_V"]) /
                np.clip(np.abs(df["current_A"]), 0.5, 999.0),
                0.0001, 0.05
            )
        for col in PHYSICS_COLS:
            if col not in df.columns:
                df[col] = 0.0
        return df

    train_df = engineer(train_df)
    val_df   = engineer(val_df)
    test_df  = engineer(test_df)

    for df in [train_df, val_df, test_df]:
        for col in FEATURE_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())

    # Fit scalers on training data only
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    scaler_X.fit(train_df[FEATURE_COLS].values)
    scaler_y.fit(train_df[[TARGET_COL]].values)

    print("  Feature scaling done")
    print("  T_internal mean=" + str(round(float(scaler_y.mean_[0]), 2)) +
          "C  std=" + str(round(float(np.sqrt(scaler_y.var_[0])), 2)) + "C")

    return train_df, val_df, test_df, scaler_X, scaler_y


# ============================================================
# STEP 2 -- DATASET WITH SLIDING WINDOWS
# ============================================================

class BatteryDataset(Dataset):
    def __init__(self, df, scaler_X, scaler_y, seq_len, stride):
        self.sequences = []

        for run_id, grp in df.groupby("run_id"):
            grp = grp.reset_index(drop=True)
            if len(grp) < seq_len + 1:
                continue

            X = scaler_X.transform(
                grp[FEATURE_COLS].values
            ).astype(np.float32)
            y = scaler_y.transform(
                grp[[TARGET_COL]].values
            ).astype(np.float32).flatten()
            r = grp[PHYSICS_COLS].values.astype(np.float32)
            y_raw = grp[TARGET_COL].values.astype(np.float32)

            for start in range(0, len(grp) - seq_len, stride):
                end = start + seq_len
                self.sequences.append((
                    X[start:end],
                    y[end - 1],
                    r[start:end],
                    y_raw[end - 1],
                ))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X, y, r, y_raw = self.sequences[idx]
        return (
            torch.tensor(X),
            torch.tensor(y),
            torch.tensor(r),
            torch.tensor(y_raw),
        )


# ============================================================
# STEP 3 -- GRU MODEL
# ============================================================
# GRU (Gated Recurrent Unit) is a simpler version of LSTM.
# LSTM has 3 gates: input gate, forget gate, output gate.
# GRU  has 2 gates: reset gate, update gate.
#
# Reset gate  -- decides how much past information to forget
# Update gate -- decides how much new information to add
#
# Fewer gates = fewer parameters = faster training.
# In practice GRU and LSTM perform very similarly on most tasks.
# This comparison tells us whether the extra complexity of LSTM
# is actually needed for this specific problem.

class BatteryGRU(nn.Module):
    def __init__(self, input_size=12, hidden_size=256,
                 num_layers=2, dropout=0.2):
        super(BatteryGRU, self).__init__()

        # GRU replaces LSTM -- everything else stays identical
        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )

        # Output head -- same as LSTM version
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, n_features)
        out, _ = self.gru(x)
        # Take last timestep output
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# STEP 4 -- LOSS FUNCTION
# ============================================================

def gru_loss(y_pred, y_true):
    """
    Simple Huber loss -- same as baseline LSTM.
    No physics constraint -- this is a pure data-driven GRU.
    This gives a fair comparison against the baseline LSTM.
    """
    return nn.functional.huber_loss(y_pred, y_true, delta=1.0)


# ============================================================
# STEP 5 -- TRAINING AND EVALUATION
# ============================================================

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for X, y, r, y_raw in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(X)
        loss   = gru_loss(y_pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, loader, scaler_y):
    model.eval()
    total_loss = 0.0
    preds_C    = []
    truths_C   = []
    n_batches  = 0

    with torch.no_grad():
        for X, y, r, y_raw in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_pred = model(X)
            loss   = gru_loss(y_pred, y)
            total_loss += loss.item()
            n_batches  += 1

            # Convert back to Celsius
            pred_np  = y_pred.cpu().numpy().reshape(-1, 1)
            true_np  = y_raw.numpy().reshape(-1, 1)
            pred_C   = scaler_y.inverse_transform(pred_np).flatten()
            preds_C.extend(pred_C.tolist())
            truths_C.extend(true_np.flatten().tolist())

    preds_C  = np.array(preds_C)
    truths_C = np.array(truths_C)
    errors   = preds_C - truths_C
    rmse     = float(np.sqrt(np.mean(errors**2)))
    mae      = float(np.mean(np.abs(errors)))

    return total_loss / max(n_batches, 1), rmse, mae, preds_C, truths_C


# ============================================================
# STEP 6 -- PLOTS
# ============================================================

def plot_training_curve(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("GRU Training Curve", fontsize=13, fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"],
                 color="#2563eb", lw=2, label="Train loss")
    axes[0].plot(epochs, history["val_loss"],
                 color="#dc2626", lw=2, label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(epochs, history["val_rmse"],
                 color="#7c3aed", lw=2, label="Val RMSE")
    axes[1].axhline(min(history["val_rmse"]),
                    color="#dc2626", lw=1, ls="--",
                    label="Best RMSE = " +
                    str(round(min(history["val_rmse"]), 4)) + "C")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE (C)")
    axes[1].set_title("Validation RMSE over Training")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Plot saved: " + path)


def plot_predictions(preds_C, truths_C, path):
    total = len(preds_C)
    if total > 3000:
        step    = total // 3000
        idx_sel = np.arange(0, total, step)[:3000]
    else:
        idx_sel = np.arange(total)

    truths_show = truths_C[idx_sel]
    preds_show  = preds_C[idx_sel]

    sort_order    = np.argsort(truths_show)
    truths_sorted = truths_show[sort_order]
    preds_sorted  = preds_show[sort_order]
    error_sorted  = preds_sorted - truths_sorted
    x_axis        = np.arange(len(sort_order))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("GRU -- Test Set Predictions", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(x_axis, truths_sorted, color="#dc2626",
            lw=1.5, label="T_internal -- Ground Truth")
    ax.plot(x_axis, preds_sorted,  color="#2563eb",
            lw=1.0, ls="--", label="T_internal -- GRU Predicted")
    ax.set_ylabel("Temperature (C)")
    ax.set_xlabel("Samples (sorted by temperature)")
    ax.set_title("Predicted vs Ground Truth -- Full Temperature Range")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    rmse = float(np.sqrt(np.mean(error_sorted**2)))
    mae  = float(np.mean(np.abs(error_sorted)))
    ax.text(0.02, 0.97,
            "RMSE = " + str(round(rmse, 4)) + " C" + chr(10) +
            "MAE  = " + str(round(mae,  4)) + " C",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax2 = axes[1]
    ax2.plot(x_axis, error_sorted, color="#7c3aed", lw=0.8, alpha=0.7)
    ax2.axhline(0,     color="gray",    lw=0.8)
    ax2.axhline( 0.5,  color="#f59e0b", lw=0.8, ls="--",
                alpha=0.6, label="+/- 0.5C")
    ax2.axhline(-0.5,  color="#f59e0b", lw=0.8, ls="--", alpha=0.6)
    ax2.axhline( 1.0,  color="#dc2626", lw=0.8, ls=":",
                alpha=0.4, label="+/- 1.0C")
    ax2.axhline(-1.0,  color="#dc2626", lw=0.8, ls=":", alpha=0.4)
    ax2.set_ylabel("Prediction Error (C)")
    ax2.set_xlabel("Samples (sorted by temperature)")
    ax2.set_title("Prediction Error (closer to 0 = better)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    within_half = float(np.mean(np.abs(error_sorted) <= 0.5) * 100)
    within_one  = float(np.mean(np.abs(error_sorted) <= 1.0) * 100)
    ax2.text(0.02, 0.97,
             "Within 0.5C: " + str(round(within_half, 1)) + "%" + chr(10) +
             "Within 1.0C: " + str(round(within_one,  1)) + "%",
             transform=ax2.transAxes, fontsize=10,
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Plot saved: " + path)


# ============================================================
# STEP 7 -- SAVE METRICS
# ============================================================

def save_metrics(rmse, mae, r2, within_half, within_one,
                 n_params, best_epoch, path):
    with open(path, "w") as f:
        f.write("GRU Model Results -- Paper 2 Comparison\n")
        f.write("=" * 45 + "\n\n")
        f.write("Model architecture:\n")
        f.write("  Type         : GRU (Gated Recurrent Unit)\n")
        f.write("  Layers       : " + str(NUM_LAYERS) + "\n")
        f.write("  Hidden units : " + str(HIDDEN_SIZE) + "\n")
        f.write("  Parameters   : " + str(n_params) + "\n")
        f.write("  Sequence len : " + str(SEQ_LEN) + "s\n")
        f.write("  Input feat.  : " + str(len(FEATURE_COLS)) + "\n\n")
        f.write("Test set results:\n")
        f.write("  RMSE         : " + str(round(rmse, 4)) + " C\n")
        f.write("  MAE          : " + str(round(mae,  4)) + " C\n")
        f.write("  R-squared    : " + str(round(r2,   5)) + "\n")
        f.write("  Within 0.5C  : " + str(round(within_half, 1)) + "%\n")
        f.write("  Within 1.0C  : " + str(round(within_one,  1)) + "%\n")
        f.write("  Best epoch   : " + str(best_epoch) + "\n\n")
        f.write("Comparison table (for paper):\n")
        f.write("-" * 45 + "\n")
        f.write("Model              RMSE     MAE\n")
        f.write("GRU (this file)    " +
                str(round(rmse, 4)).ljust(9) +
                str(round(mae, 4)) + "\n")
        f.write("Baseline LSTM      0.0990   0.0740  (from results_p2)\n")
        f.write("LSTM + PINN        0.1160   0.0890  (from results_p2)\n")
        f.write("Karnehm 2024 KAN   0.7510   0.4690  (published)\n")
        f.write("Karnehm 2024 LSTM  0.8540   0.5500  (published)\n")
    print("  Metrics saved: " + path)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*55)
    print("  Battery GRU Training -- Comparison Model")
    print("  CALB L148N58A Real Experimental Data")
    print("="*55)
    print("  Device: " + DEVICE)

    # Load data
    train_df, val_df, test_df, scaler_X, scaler_y = load_data()

    # Create datasets
    print("\n[Step 2] Creating sequence datasets...")
    train_ds = BatteryDataset(train_df, scaler_X, scaler_y, SEQ_LEN, STRIDE)
    val_ds   = BatteryDataset(val_df,   scaler_X, scaler_y, SEQ_LEN, 30)
    test_ds  = BatteryDataset(test_df,  scaler_X, scaler_y, SEQ_LEN, 20)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("  Train sequences: " + str(len(train_ds)))
    print("  Val sequences  : " + str(len(val_ds)))
    print("  Test sequences : " + str(len(test_ds)))

    # Build GRU model
    print("\n[Step 3] Building GRU model...")
    model     = BatteryGRU(
        input_size  = len(FEATURE_COLS),
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT
    ).to(DEVICE)

    n_params = model.count_parameters()
    print("  Parameters: " + str(n_params))
    print("  (LSTM had 819,329 -- GRU is smaller)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.5, verbose=False)

    # Training loop
    print("\n[Step 4] Training GRU...")
    print("-" * 55)

    history = {
        "train_loss": [], "val_loss": [],
        "val_rmse":   [], "val_mae":  []
    }

    best_rmse       = float("inf")
    best_epoch      = 0
    no_improve      = 0
    best_state      = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_dl, optimizer)
        val_loss, val_rmse, val_mae, _, _ = evaluate(
            model, val_dl, scaler_y)
        scheduler.step(val_rmse)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)
        history["val_mae"].append(val_mae)

        if val_rmse < best_rmse:
            best_rmse  = val_rmse
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print("  Epoch " + str(epoch).rjust(3) + "/" + str(EPOCHS) +
                  "  train_loss=" + str(round(train_loss, 4)) +
                  "  val_loss=" + str(round(val_loss, 4)) +
                  "  RMSE=" + str(round(val_rmse, 3)) + "C" +
                  "  MAE=" + str(round(val_mae, 3)) + "C" +
                  "  [no_improve=" + str(no_improve) + "]")

        if no_improve >= PATIENCE:
            print("  Early stopping at epoch " + str(epoch) +
                  " -- val RMSE has not improved for " +
                  str(PATIENCE) + " epochs")
            break

    print("  Training complete. Best val_RMSE=" +
          str(round(best_rmse, 4)) + "C at epoch " + str(best_epoch))

    # Restore best weights
    model.load_state_dict(best_state)

    # Save model
    model_path = os.path.join(OUTPUT_FOLDER, "gru_model.pt")
    torch.save(model.state_dict(), model_path)
    print("  Model saved: " + model_path)

    # Evaluate on test set
    print("\n[Step 5] Evaluating on test set...")
    _, test_rmse, test_mae, preds_C, truths_C = evaluate(
        model, test_dl, scaler_y)

    errors      = preds_C - truths_C
    r2          = float(1 - np.sum(errors**2) /
                        np.sum((truths_C - truths_C.mean())**2))
    within_half = float(np.mean(np.abs(errors) <= 0.5) * 100)
    within_one  = float(np.mean(np.abs(errors) <= 1.0) * 100)

    print("\n  GRU Test Results:")
    print("  " + "-"*35)
    print("  RMSE       = " + str(round(test_rmse, 4)) + " C")
    print("  MAE        = " + str(round(test_mae,  4)) + " C")
    print("  R-squared  = " + str(round(r2,        5)))
    print("  Within 0.5C: " + str(round(within_half, 1)) + "%")
    print("  Within 1.0C: " + str(round(within_one,  1)) + "%")

    # Generate plots
    print("\n[Step 6] Generating plots...")
    plot_training_curve(
        history,
        os.path.join(OUTPUT_FOLDER, "gru_training.png"))
    plot_predictions(
        preds_C, truths_C,
        os.path.join(OUTPUT_FOLDER, "gru_predictions.png"))

    # Save metrics
    print("\n[Step 7] Saving metrics...")
    save_metrics(
        test_rmse, test_mae, r2,
        within_half, within_one,
        n_params, best_epoch,
        os.path.join(OUTPUT_FOLDER, "gru_metrics.txt"))

    # Final comparison summary
    print("\n" + "="*55)
    print("  FINAL COMPARISON SUMMARY")
    print("="*55)
    print("  Model              RMSE      MAE")
    print("  " + "-"*40)
    print("  GRU (this)         " +
          str(round(test_rmse, 4)).ljust(10) +
          str(round(test_mae, 4)))
    print("  Baseline LSTM      0.0990    0.0740")
    print("  LSTM + PINN        0.1160    0.0890")
    print("  " + "-"*40)
    print("  Karnehm 2024 KAN   0.7510    0.4690  (published)")
    print("  Karnehm 2024 LSTM  0.8540    0.5500  (published)")
    print()
    print("  All results saved to: " + os.path.abspath(OUTPUT_FOLDER))
    print("="*55)


if __name__ == "__main__":
    main()
