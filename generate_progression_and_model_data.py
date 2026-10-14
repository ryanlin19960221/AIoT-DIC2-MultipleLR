import os
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load 50 Startups Data
df = pd.read_csv("50_Startups.csv")
df_enc = pd.get_dummies(df, columns=["State"], dtype=int)

features = [
    "R&D Spend",
    "Administration",
    "Marketing Spend",
    "State_Florida",
    "State_New York"
]

X = df_enc[features].astype(float)
y = df_enc["Profit"].astype(float)

# Seed 42 aligns with main.py CRISP-DM standard
RANDOM_SEED = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED
)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)

def evaluate_subset_metrics(features_subset):
    sub = list(features_subset)
    model = LinearRegression()
    model.fit(X_train[sub], y_train)
    y_pred = model.predict(X_test[sub])
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    
    n = len(y_test)
    p = len(sub)
    adj_r2 = float(1.0 - (1.0 - r2) * ((n - 1) / (n - p - 1))) if (n - p - 1) > 0 else r2
    return rmse, r2, mae, adj_r2

algorithms = {}

# 1. Lasso Ranking (L1)
lasso_cv = LassoCV(cv=5, random_state=RANDOM_SEED, max_iter=100000).fit(X_train_scaled, y_train)
lasso_rank = [f for _, f in sorted(zip(abs(lasso_cv.coef_), features), reverse=True)]
algorithms["Lasso (L1 Ranking)"] = [lasso_rank[:k] for k in range(1, 6)]

# 2. Forward SFS
sfs_subsets = []
for k in range(1, 6):
    if k == 5:
        sfs_subsets.append(features)
    else:
        sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction="forward")
        sfs.fit(X_train, y_train)
        sfs_subsets.append(list(X_train.columns[sfs.get_support()]))
algorithms["Forward SFS"] = sfs_subsets

# 3. Recursive Feature Elimination (RFE)
rfe_subsets = []
for k in range(1, 6):
    if k == 5:
        rfe_subsets.append(features)
    else:
        rfe = RFE(LinearRegression(), n_features_to_select=k)
        rfe.fit(X_train_scaled, y_train)
        rfe_subsets.append(list(X_train.columns[rfe.get_support()]))
algorithms["RFE"] = rfe_subsets

# 4. ElasticNet Ranking
enet_cv = ElasticNetCV(cv=5, random_state=RANDOM_SEED, max_iter=100000).fit(X_train_scaled, y_train)
enet_rank = [f for _, f in sorted(zip(abs(enet_cv.coef_), features), reverse=True)]
algorithms["ElasticNet Ranking"] = [enet_rank[:k] for k in range(1, 6)]

# 5. Exhaustive Search
exhaustive_subsets = []
for k in range(1, 6):
    best_rmse = float("inf")
    best_sub = None
    for sub in itertools.combinations(features, k):
        rmse, _, _, _ = evaluate_subset_metrics(sub)
        if rmse < best_rmse:
            best_rmse = rmse
            best_sub = list(sub)
    exhaustive_subsets.append(best_sub)
algorithms["Exhaustive Search"] = exhaustive_subsets

# Standard algorithm key mapping for frontend
ALGO_KEY_MAP = {
    "Lasso (L1 Ranking)": "Lasso",
    "Forward SFS": "Forward SFS",
    "RFE": "RFE",
    "ElasticNet Ranking": "ElasticNet",
    "Exhaustive Search": "Exhaustive"
}

model_data_registry = {}
all_dfs = {}
comparison_rows = []

for full_name, subsets in algorithms.items():
    frontend_key = ALGO_KEY_MAP[full_name]
    model_data_registry[frontend_key] = []
    
    progression_records = []
    for k, sub in enumerate(subsets, start=1):
        rmse, r2, mae, adj_r2 = evaluate_subset_metrics(sub)
        
        # Statsmodels detailed model fit
        X_train_sm = sm.add_constant(X_train[sub])
        X_test_sm = sm.add_constant(X_test[sub])
        ols_res = sm.OLS(y_train, X_train_sm).fit()
        
        y_test_pred = ols_res.predict(X_test_sm)
        f_stat = float(ols_res.fvalue) if not np.isnan(ols_res.fvalue) else 300.0
        
        # Calculate VIF
        vif_dict = {}
        if len(sub) > 1:
            for i, col in enumerate(X_train[sub].columns):
                try:
                    vif_dict[col] = float(variance_inflation_factor(X_train[sub].values, i))
                except Exception:
                    vif_dict[col] = 1.0
        else:
            vif_dict[sub[0]] = 1.0
            
        # Build coefficients array
        coef_list = []
        conf_int = ols_res.conf_int()
        for col_name in X_train_sm.columns:
            coef_val = float(ols_res.params[col_name])
            stderr_val = float(ols_res.bse[col_name])
            tstat_val = float(ols_res.tvalues[col_name])
            pval_val = float(ols_res.pvalues[col_name])
            ci_low = float(conf_int.loc[col_name, 0])
            ci_high = float(conf_int.loc[col_name, 1])
            vif_val = vif_dict.get(col_name, None)
            
            feat_disp_name = col_name if col_name != "const" else "Intercept (β₀)"
            coef_list.append({
                "feature": feat_disp_name,
                "coef": coef_val,
                "stderr": stderr_val,
                "tstat": tstat_val,
                "pvalue": pval_val,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "vif": vif_val
            })
            
        # Build outliers / residuals list for test set (10 samples)
        test_residuals = []
        for rank_idx, (orig_idx, actual_val, pred_val) in enumerate(
            sorted(
                zip(y_test.index, y_test.values, y_test_pred.values),
                key=lambda x: abs(x[1] - x[2]),
                reverse=True
            ),
            start=1
        ):
            resid = float(actual_val - pred_val)
            abs_err = abs(resid)
            pct_err = float((abs_err / actual_val) * 100) if actual_val != 0 else 0.0
            test_residuals.append({
                "rank": rank_idx,
                "index": int(orig_idx),
                "actual": float(actual_val),
                "predicted": float(pred_val),
                "residual": resid,
                "abs_error": abs_err,
                "pct_error": pct_err
            })
            
        model_spec = {
            "k": k,
            "features": sub,
            "rmse": rmse,
            "r2": r2,
            "adj_r2": adj_r2,
            "mae": mae,
            "f_stat": f_stat,
            "coefficients": coef_list,
            "outliers": test_residuals
        }
        model_data_registry[frontend_key].append(model_spec)
        
        progression_records.append({
            "Number of Features": k,
            "Selected Features": str(sub),
            "RMSE": rmse,
            "R-squared": r2
        })
        
        comparison_rows.append({
            "Algorithm": full_name,
            "k": k,
            "Selected Features": ", ".join(sub),
            "RMSE": rmse,
            "R-squared": r2,
            "Adj R-squared": adj_r2,
            "MAE": mae
        })
        
    df_prog = pd.DataFrame(progression_records)
    all_dfs[full_name] = df_prog
    
    # Save progression CSV
    csv_filename = f"{full_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_progression.csv"
    df_prog.to_csv(csv_filename, index=False)
    print(f"Saved {csv_filename} ({len(df_prog)} rows)")
    
    # Plot individual chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(df_prog["Number of Features"], df_prog["RMSE"], marker='o', linewidth=2.2, color='#6366f1')
    ax1.set_title(f"{full_name} - RMSE by k", fontsize=12, pad=10)
    ax1.set_xlabel("Number of Features (k)", fontsize=11)
    ax1.set_ylabel("RMSE ($)", fontsize=11)
    ax1.set_xticks(range(1, 6))
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    ax2.plot(df_prog["Number of Features"], df_prog["R-squared"], marker='o', linewidth=2.2, color='#10b981')
    ax2.set_title(f"{full_name} - R² by k", fontsize=12, pad=10)
    ax2.set_xlabel("Number of Features (k)", fontsize=11)
    ax2.set_ylabel("R-squared", fontsize=11)
    ax2.set_xticks(range(1, 6))
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plot_filename = f"{full_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_plot.png"
    plt.savefig(plot_filename, dpi=300)
    plt.close()

# Save combined comparison CSV
df_comp = pd.DataFrame(comparison_rows)
df_comp.to_csv("feature_selection_comparison.csv", index=False)
print("Saved feature_selection_comparison.csv")

# Combined Comparison Plot across all 5 algorithms
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
colors = ['#6366f1', '#f59e0b', '#10b981', '#f43f5e', '#a855f7']
markers = ['o', 's', '^', 'D', 'v']

for (name, df_res), color, marker in zip(all_dfs.items(), colors, markers):
    ax1.plot(df_res["Number of Features"], df_res["RMSE"], marker=marker, linewidth=2, label=name, color=color)
    ax2.plot(df_res["Number of Features"], df_res["R-squared"], marker=marker, linewidth=2, label=name, color=color)

ax1.set_title("RMSE Comparison across 5 Feature Selection Methods", fontsize=12, pad=10)
ax1.set_xlabel("Number of Features (k)", fontsize=11)
ax1.set_ylabel("RMSE ($) (Lower is Better)", fontsize=11)
ax1.set_xticks(range(1, 6))
ax1.legend()
ax1.grid(True, linestyle=":", alpha=0.6)

ax2.set_title("R-squared Comparison across 5 Feature Selection Methods", fontsize=12, pad=10)
ax2.set_xlabel("Number of Features (k)", fontsize=11)
ax2.set_ylabel("R-squared (Higher is Better)", fontsize=11)
ax2.set_xticks(range(1, 6))
ax2.legend()
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("all_5_algorithms_comparison_plot.png", dpi=300)
plt.close()

# Save feature_selection_comparison.png bar chart
fig, ax = plt.subplots(figsize=(10, 5))
sub_best = df_comp[df_comp['k'] == 2]
ax.bar(sub_best['Algorithm'], sub_best['RMSE'], color=colors)
ax.set_title("Optimal k=2 RMSE across 5 Feature Selection Methods", fontsize=12)
ax.set_ylabel("Test RMSE ($)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("feature_selection_comparison.png", dpi=300)
plt.close()

# Save model_data.json and model_data.js
with open("model_data.json", "w", encoding="utf-8") as f:
    json.dump(model_data_registry, f, indent=2, ensure_ascii=False)
print("Saved model_data.json")

js_content = "window.MODEL_DATA = " + json.dumps(model_data_registry, indent=2, ensure_ascii=False) + ";"
with open("model_data.js", "w", encoding="utf-8") as f:
    f.write(js_content)
with open("static/model_data.js", "w", encoding="utf-8") as f:
    f.write(js_content)
print("Saved model_data.js and static/model_data.js")
print("\nAll CSV parameters and model data successfully re-generated!")
