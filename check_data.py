import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\yuvav\battery_data\calb_data\calb_full_dataset.csv')

print("T_internal percentiles:")
for p in [70, 75, 80, 85, 90, 95]:
    val = np.percentile(df['T_internal_C'], p)
    print("  " + str(p) + "th percentile: " + str(round(val, 2)) + "C")

print("Max: " + str(round(df['T_internal_C'].max(), 2)) + "C")

print("\nProfiles in dataset:")
print(df['profile'].unique())
