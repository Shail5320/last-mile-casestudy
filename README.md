# Last Mile — Delivery ETA Prediction

Predicting food delivery time from real-world, messy data — with an emphasis on explaining *why* the model predicts what it does, not just reporting an accuracy number.

**[Live demo →](https://last-mile-casestudy.vercel.app/)** &nbsp;|&nbsp; **[Notebook →](https://github.com/Shail5320/last-mile-casestudy/blob/main/main.ipynb)**

## The problem

Delivery apps show a live countdown the moment you order. That number is a prediction, built from patterns in past deliveries — distance, traffic, weather, and how much else the rider is juggling at once. This project builds that predictor end-to-end from a real dataset, and treats the explanation as seriously as the model itself.

## Dataset

[Zomato Delivery Operations Analytics Dataset](https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset) (Kaggle) — 45,584 real delivery records including rider details, restaurant and drop-off GPS coordinates, weather, traffic density, and actual delivery time.

## Pipeline

1. **Data cleaning**
   - Identified and removed ~3,640 rows with corrupted GPS coordinates (silently defaulted to `(0,0)` / near-zero)
   - Handled 8 columns with missing data, using median/mode/row-drop depending on the column
   - Removed ~431 rows with physically impossible delivery distances (>30 km, jumping straight to thousands of km — clearly corrupted, not just "far")

2. **Feature engineering**
   - Derived `distance_km` from raw lat/long using the **Haversine formula** (accounts for Earth's curvature — flat-plane distance math is wrong for GPS coordinates)
   - Encoded `Road_traffic_density` **ordinally** (Low < Medium < High < Jam), since order carries real information
   - One-hot encoded nominal categories: weather, order type, vehicle type, city

3. **EDA — the assumption that got overturned**
   - Distance vs. delivery time: correlation of only **0.32** — weak, and visually near-random in a scatter plot
   - Traffic density: clear step-up pattern, more time as congestion increases
   - Weather: Sunny days fastest (median 20 min), Fog/Cloudy slowest (median 29 min)
   - Festival days: nearly **2x slower** (median ~25 min → ~45 min)
   - Multiple simultaneous deliveries: clean, near-monotonic staircase — the single strongest visual pattern found

4. **Modeling**

   | Model | MAE | RMSE | R² |
   |---|---|---|---|
   | Linear Regression (deployed) | 5.17 min | 6.48 | 0.530 |
   | Random Forest (100 trees) | 3.42 min | 4.27 | 0.795 |

   Linear Regression is deployed in the live demo because every prediction can be traced back to a visible, explainable equation. Random Forest performs meaningfully better (likely capturing non-linear interactions, e.g. traffic × festival compounding) and is documented here as the next planned upgrade once I've properly learned how tree ensembles work internally — not deployed as a black box in the meantime.

5. **Data leakage check**
   `Delivery_person_Ratings` came out as the single most important feature in the Random Forest — a red flag, since ratings can be influenced by delivery speed itself. Retraining without it dropped R² only modestly (0.828 → 0.795), suggesting it's mostly genuine signal (rider experience) rather than pure leakage. Excluded from the final model anyway, as the more defensible choice.

6. **Small-sample coefficient flag**
   The `City_Semi-Urban` coefficient came out unusually large (+10.6 min), even bigger than the festival effect — but that category is only 141 of 40,188 rows (0.35% of the data), so it's flagged as likely noise rather than a reliable effect.

## Live demo

The deployed site runs the trained Linear Regression model **entirely client-side** — the model's intercept and coefficients are embedded as plain numbers, and predictions are computed directly in JavaScript. No backend, no API calls, no server costs.

## Tech stack

- **Analysis & modeling:** Python, pandas, scikit-learn, matplotlib, seaborn (Jupyter notebook)
- **Demo site:** vanilla HTML/CSS/JS, deployed as a static page

## Repo structure

```
├── notebook/
│   └── eta_prediction.ipynb      # full analysis, cleaning, EDA, modeling
├── model_export.json             # trained Linear Regression weights
├── index.html                    # live demo (static, client-side inference)
└── README.md
```

## What I'd do differently next time

- Engineer time-of-day features from `Time_Ordered` instead of dropping it
- Properly learn and deploy the Random Forest once its internals aren't a black box to me
- Get a real data dictionary to resolve the ratings-leakage question with certainty rather than inference

## Credits

Analysis, cleaning, feature engineering, and modeling done by hand. The demo site's design and code were built with AI assistance (Claude).
