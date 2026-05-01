import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("calb_data", exist_ok=True)

df = pd.read_csv(r'C:\Users\yuvav\battery_data\calb_data\calb_full_dataset.csv')

print("=" * 55)
print("  CALB Real Dataset Analysis -- Paper 2")
print("=" * 55)

print("\n1. BASIC STATS")
print("  Total rows  : " + str(len(df)))
print("  Total runs  : " + str(df['run_id'].nunique()))
print("  Profiles    : " + str(list(df['profile'].unique())))
print("  Temperatures: " + str(list(df['T_amb_label'].unique())))

print("\n2. T_INTERNAL RANGE")
print("  Min  : " + str(round(df['T_internal_C'].min(), 2)) + "C")
print("  Max  : " + str(round(df['T_internal_C'].max(), 2)) + "C")
print("  Mean : " + str(round(df['T_internal_C'].mean(), 2)) + "C")
print("  Std  : " + str(round(df['T_internal_C'].std(), 2)) + "C")

print("\n3. CORE-TO-SURFACE GAP (the key safety metric)")
gap = df['T_core_surf_gap']
print("  Mean gap   : " + str(round(gap.mean(), 3)) + "C")
print("  Max gap    : " + str(round(gap.max(), 3)) + "C")
print("  Rows gap>1C: " + str((gap > 1.0).sum()) +
      " (" + str(round(100*(gap > 1.0).mean(), 1)) + "%)")
print("  Rows gap>2C: " + str((gap > 2.0).sum()) +
      " (" + str(round(100*(gap > 2.0).mean(), 1)) + "%)")
print("  Rows gap>3C: " + str((gap > 3.0).sum()) +
      " (" + str(round(100*(gap > 3.0).mean(), 1)) + "%)")

print("\n4. THRESHOLD ANALYSIS -- which threshold makes sense?")
for thresh in [35, 36, 37, 38, 39, 40, 41, 42]:
    count = (df['T_internal_C'] >= thresh).sum()
    pct   = round(100 * count / len(df), 1)
    print("  >=" + str(thresh) + "C : " + str(count) + " rows (" + str(pct) + "%)")

print("\n5. SURFACE VALIDATION SUMMARY")
if 'T_surface_predicted' in df.columns:
    rmse_all = float(np.sqrt(np.mean(
        (df['T_surface_predicted'] - df['T_surface_real'])**2
    )))
    print("  Overall surface RMSE: " + str(round(rmse_all, 3)) + "C")
    for prof in df['profile'].unique():
        sub  = df[df['profile'] == prof]
        rmse = float(np.sqrt(np.mean(
            (sub['T_surface_predicted'] - sub['T_surface_real'])**2
        )))
        print("  " + str(prof).ljust(16) + ": RMSE = " +
              str(round(rmse, 3)) + "C")

print("\n6. GAP BY PROFILE (shows which profiles create biggest blind spot)")
for prof in sorted(df['profile'].unique()):
    sub = df[df['profile'] == prof]
    g   = sub['T_core_surf_gap']
    print("  " + str(prof).ljust(16) +
          ": mean_gap=" + str(round(g.mean(), 3)) + "C" +
          "  max_gap=" + str(round(g.max(), 3)) + "C")

print("\n7. RECOMMENDED PAPER FRAMING")
max_gap  = round(gap.max(), 2)
mean_gap = round(gap.mean(), 3)
pct_2c   = round(100*(gap > 2.0).mean(), 1)
print("  Max core-surface gap   : " + str(max_gap) + "C")
print("  Mean core-surface gap  : " + str(mean_gap) + "C")
print("  Time gap>2C            : " + str(pct_2c) + "% of all measurements")
print()
print("  KEY PAPER STATEMENT:")
print("  'The internal core temperature exceeded the surface")
print("   temperature by up to " + str(max_gap) + "C, with " + str(pct_2c) + "% of")
print("   measurements showing a gap greater than 2C --")
print("   demonstrating that surface sensors systematically")
print("   underestimate the thermal stress experienced by")
print("   the battery core during real drive cycle conditions.'")

print("\n" + "=" * 55)
