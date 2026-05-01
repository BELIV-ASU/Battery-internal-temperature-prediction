# Battery Internal Temperature Prediction

Real-time internal core temperature estimation for CALB L148N58A 58Ah NMC 
prismatic lithium-ion cells using LSTM and Physics-Informed Neural Networks.

## Results

| Model | RMSE | MAE |
|---|---|---|
| Baseline LSTM | 0.099°C | 0.074°C |
| GRU | 0.114°C | 0.091°C |
| LSTM + PINN | 0.116°C | 0.089°C |
| Karnehm 2024 (published) | 0.751°C | 0.469°C |

## How to Run

1. Download CALB dataset from Mendeley Data
2. `python calb_preprocess.py`
3. `python battery_train_p2.py`
4. `python battery_train_gru.py`
5. `python battery_validate_p2.py`

## Documentation

See `BELIV_Lab_Battery_Guide.docx` for complete step-by-step instructions.

## Cell Specifications

- **Cell**: CALB L148N58A
- **Chemistry**: NMC (Nickel Manganese Cobalt)
- **Capacity**: 58 Ah
- **Dataset**: 6,834,132 rows, 164 experimental runs
- **Profiles**: WLTP, UDDS, US06, C20 charge, C20 discharge
- **Temperatures**: 10°C, 25°C, 40°C

