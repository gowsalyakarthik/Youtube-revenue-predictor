YouTube Ad Revenue Predictor (Lasso + Streamlit)

Predict per-video YouTube ad revenue (USD) from early performance & audience features using a linear regression family pipeline (Lasso winner) in a clean Streamlit app.

Problem

Creators need fast revenue estimates for planning. Build a linear model (OLS/Ridge/Lasso) to predict ad_revenue_usd, and deploy it for interactive use.

What’s inside

app/model.pkl – serialized sklearn Pipeline (impute → scale → OHE → Lasso)

app/streamlit_app.py – Streamlit UI

requirements.txt, README.md

model_comparison_results.csv, model_top_features.csv (explainability)

Quick start
pip install -r requirements.txt
cd app
python -m streamlit run streamlit_app.py


Open the printed URL (e.g., http://localhost:8501
).

Inputs (must match training)

Numeric: views, likes, comments, watch_time_minutes, video_length_minutes, subscribers

Engineered: engagement_rate = (likes + comments) / views (computed safely in app)

Categorical: category, device, country

Results (your run)

Winner: Lasso — R² 0.9526, RMSE 13.48, MAE 3.10 (on test set).

Notes

Estimates are indicative, not official YouTube payouts.

Pipeline ensures the same preprocessing at train & inference.
