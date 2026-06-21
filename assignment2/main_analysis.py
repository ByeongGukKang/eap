# %%
import pickle

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from model_eval import (
    aggregate_importance_by_characteristic,
    calculate_model_oos_r2,
    calculate_oos_decile_portfolio,
    calculate_variable_importance,
)
from utils import characteristics_dict, scan_ghz, scan_macro

# %%
# Load Data
lz_ghz = scan_ghz()
lz_macro = scan_macro()
GLOBAL_SEED = 0  # global random seed


# %%
# Aggregate GHZ and Macro(Goyal and Welch) data + Generate sic2 dummy variables
lz_merged = lz_ghz.join(lz_macro, on="date", how="inner")
sic_dummies = (
    lz_merged.select(pl.col("date"), pl.col("permno"), pl.col("sic2"))
    .collect()
    .to_dummies("sic2")
)
lz_merged = lz_merged.join(sic_dummies.lazy(), on=["date", "permno"], how="left")

# Create interaction terms
macro_vars = ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar"]
expr_intreactions = []
for mvar in macro_vars:
    for char in characteristics_dict.keys():
        expr_intreactions.append((pl.col(char) * pl.col(mvar)).alias(f"{char}_{mvar}"))
lz_merged = lz_merged.with_columns(expr_intreactions)

# final df
common_cols = ["permno", "date", "ret", "mve"]
feature_cols = list(characteristics_dict.keys())
feature_cols.extend(
    [col for col in lz_merged.collect_schema().names() if col.startswith("sic2_")]
)
for mvar in macro_vars:
    for char in characteristics_dict.keys():
        feature_cols.append(f"{char}_{mvar}")

# The number of features is smaller than the original paper due to missing sic2 dummy
# In EAP via ML, there exists 74 sic dummy variables, but here I only have 73.
# Final data
df_data = (
    lz_merged.filter(pl.col("date") >= pl.date(2004, 1, 1))
    .select(common_cols + feature_cols)
    .sort(["date", "permno"])
    .collect()
)
# To reduce computation, use a random subset of permnos
random_permnos = df_data["permno"].unique().shuffle(GLOBAL_SEED)[:3000].to_list()
df_data = df_data.filter(pl.col("permno").is_in(random_permnos))

# %%
# Get models
with open("./model/ols.pkl", "rb") as f:
    model_ols = pickle.load(f)
with open("./model/ols3.pkl", "rb") as f:
    model_ols3 = pickle.load(f)
with open("./model/pcr.pkl", "rb") as f:
    model_pcr = pickle.load(f)
with open("./model/enet.pkl", "rb") as f:
    model_enet = pickle.load(f)
with open("./model/rf.pkl", "rb") as f:
    model_rf = pickle.load(f)
with open("./model/nn2.pkl", "rb") as f:
    model_nn2 = pickle.load(f)
with open("./model/nn4.pkl", "rb") as f:
    model_nn4 = pickle.load(f)

# %%
# Table 1: OOS R-squared for each model
oos_r2_dict = {}
model_names = ["OLS+H", "OLS-3+H", "PCR", "ENet+H", "RF", "NN2", "NN4"]
for models, name in zip(
    [model_ols, model_ols3, model_pcr, model_enet, model_rf, model_nn2, model_nn4],
    model_names,
):
    match name:
        case "OLS+H" | "ENet+H":
            oos_r2 = calculate_model_oos_r2(
                models,
                df_data.select(["date"] + feature_cols),
                df_data.select(["date", "ret"]),
                "linear",
                name,
            )
            oos_r2_dict[name] = oos_r2
        case "OLS-3+H":
            oos_r2 = calculate_model_oos_r2(
                models,
                df_data.select(["date"] + ["mve_m", "bm", "mom12m"]),
                df_data.select(["date", "ret"]),
                "linear",
                name,
            )
            oos_r2_dict[name] = oos_r2
        case "PCR":
            oos_r2 = calculate_model_oos_r2(
                models,
                df_data.select(["date"] + feature_cols),
                df_data.select(["date", "ret"]),
                "pcr",
                name,
            )
            oos_r2_dict[name] = oos_r2
        case "RF":
            oos_r2 = calculate_model_oos_r2(
                models,
                df_data.select(["date"] + feature_cols),
                df_data.select(["date", "ret"]),
                "rf",
                name,
            )
            oos_r2_dict[name] = oos_r2
        case "NN2" | "NN4":
            oos_r2 = calculate_model_oos_r2(
                models,
                df_data.select(["date"] + feature_cols),
                df_data.select(["date", "ret"]),
                "nn",
                name,
            )
            oos_r2_dict[name] = oos_r2
print()
for k, v in oos_r2_dict.items():
    print(f"{k}: {v * 100:.2f}%")


# %%
# Figure 4: Variable importance by model
var_import_dict = {}
for models, name in zip(
    [model_ols, model_ols3, model_pcr, model_enet, model_rf, model_nn2, model_nn4],
    model_names,
):
    match name:
        case "OLS+H" | "ENet+H":
            var_import = calculate_variable_importance(
                models,
                df_data.select(["date"] + feature_cols),
                df_data.select(["date", "ret"]),
                "linear",
            )
            var_import_dict[name] = var_import
        case "OLS-3+H":
            var_import = calculate_variable_importance(
                models,
                df_data.select(["date"] + ["mve_m", "bm", "mom12m"]),
                df_data.select(["date", "ret"]),
                "linear",
            )
            var_import_dict[name] = var_import
        case "PCR":
            var_import = calculate_variable_importance(
                models,
                df_data.select(["date"] + feature_cols),
                df_data.select(["date", "ret"]),
                "pcr",
            )
            var_import_dict[name] = var_import
        case "RF":
            var_import = calculate_variable_importance(
                models,
                df_data.select(["date"] + feature_cols),
                df_data.select(["date", "ret"]),
                "rf",
            )
            var_import_dict[name] = var_import
        case "NN2" | "NN4":
            var_import = calculate_variable_importance(
                models,
                df_data.select(["date"] + feature_cols),
                df_data.select(["date", "ret"]),
                "nn",
            )
            var_import_dict[name] = var_import

# Aggregate importance by characteristic
for k, v in var_import_dict.items():
    var_import_dict[k] = aggregate_importance_by_characteristic(
        v, list(characteristics_dict.keys())
    )


# %%
# Prepare data for plotting
ordered_model_names = list(var_import_dict.keys())

base_df = None
for model in ordered_model_names:
    df_renamed = var_import_dict[model].rename({"importance": model})

    if base_df is None:
        base_df = df_renamed
    else:
        base_df = base_df.join(df_renamed, on="characteristic", how="left")
assert base_df is not None
base_df = base_df.fill_null(0.0)

for model in ordered_model_names:
    max_val = base_df[model].max()
    min_val = base_df[model].min()
    if max_val != min_val:
        base_df = base_df.with_columns(
            ((pl.col(model) - min_val) / (max_val - min_val)).alias(model)
        )

base_df = base_df.with_columns(
    pl.sum_horizontal(ordered_model_names).alias("total_importance")
).sort("total_importance", descending=True)

y_labels = base_df["characteristic"].to_list()
matrix_data = base_df[ordered_model_names].to_numpy()

# Plot
fig, ax = plt.subplots(figsize=(8, 10), dpi=200)

cax = ax.imshow(matrix_data, cmap="Blues", aspect="auto", origin="upper")

ax.set_xticks(np.arange(len(ordered_model_names)))
ax.set_xticklabels(ordered_model_names, fontsize=11)
ax.set_yticks(np.arange(len(y_labels)))
ax.set_yticklabels(y_labels, fontsize=9)
ax.set_xticks(np.arange(len(ordered_model_names)) - 0.5, minor=True)
ax.set_yticks(np.arange(len(y_labels)) - 0.5, minor=True)

ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
ax.tick_params(which="both", bottom=False, left=False, labelbottom=True, labelleft=True)
for spine in ax.spines.values():
    spine.set_visible(False)

plt.savefig("./output/variable_importance.png")
plt.show()


# %%
# Cumulative Return of Long-Short Portfolio
pf_perf_dict = {}
for models, name in zip(
    [model_ols, model_ols3, model_pcr, model_enet, model_rf, model_nn2, model_nn4],
    model_names,
):
    match name:
        case "OLS+H" | "ENet+H":
            pf_perf = calculate_oos_decile_portfolio(
                models,
                df_data.select(["date", "permno"] + feature_cols),
                df_data.select(["date", "ret"]),
                "linear",
            )
            pf_perf_dict[name] = pf_perf
        case "OLS-3+H":
            pf_perf = calculate_oos_decile_portfolio(
                models,
                df_data.select(["date", "permno"] + ["mve_m", "bm", "mom12m"]),
                df_data.select(["date", "ret"]),
                "linear",
            )
            pf_perf_dict[name] = pf_perf
        case "PCR":
            pf_perf = calculate_oos_decile_portfolio(
                models,
                df_data.select(["date", "permno"] + feature_cols),
                df_data.select(["date", "ret"]),
                "pcr",
            )
            pf_perf_dict[name] = pf_perf
        case "RF":
            pf_perf = calculate_oos_decile_portfolio(
                models,
                df_data.select(["date", "permno"] + feature_cols),
                df_data.select(["date", "ret"]),
                "rf",
            )
            pf_perf_dict[name] = pf_perf
        case "NN2" | "NN4":
            pf_perf = calculate_oos_decile_portfolio(
                models,
                df_data.select(["date", "permno"] + feature_cols),
                df_data.select(["date", "ret"]),
                "nn",
            )
            pf_perf_dict[name] = pf_perf


# %%
# Read recession periods & S&P 500 returns
us_recession_indicator = pl.read_csv("./data/USREC.csv").with_columns(
    pl.col("observation_date").str.to_date().dt.month_end()
)
sp500_returns = (
    pl.read_csv("./data/SP500.csv")
    .with_columns(pl.col("observation_date").str.to_date().dt.month_end())
    .with_columns(ret=pl.col("SP500").pct_change().fill_null(0))
)
rf_rate = (
    pl.read_csv("./data/DTB3.csv")
    .with_columns(pl.col("observation_date").str.to_date().dt.month_end())
    .with_columns(DTB3=pl.col("DTB3") / 3 / 365)
)
sp500_minus_rf = (
    sp500_returns.join(rf_rate, on="observation_date", how="left")
    .with_columns(exret=pl.col("ret") - pl.col("DTB3"))
    .select(["observation_date", "exret"])
)

sdate = max(sp500_minus_rf["observation_date"].min(), pf_perf_dict["NN2"]["date"].min())  # type: ignore
edate = min(sp500_minus_rf["observation_date"].max(), pf_perf_dict["NN2"]["date"].max())  # type: ignore
sp500_bench = sp500_minus_rf.filter(
    (pl.col("observation_date") >= sdate) & (pl.col("observation_date") <= edate)
).sort("observation_date")
recession_zone = us_recession_indicator.filter(
    (pl.col("observation_date") >= sdate) & (pl.col("observation_date") <= edate)
).sort("observation_date")


# %%
# plot
plt.figure(figsize=(12, 6))

plt.fill_between(
    recession_zone["observation_date"].to_list(),
    0,
    1,
    where=(recession_zone["USREC"] == 1).to_list(),
    color="grey",
    alpha=0.2,
    label="NBER Recession",
    transform=plt.gca().get_xaxis_transform(),
)
plt.plot(
    sp500_bench["observation_date"],
    sp500_bench["exret"].cum_sum(),
    color="black",
    linewidth=2.0,
    label="S&P 500 - Rf",
)
for k, v in pf_perf_dict.items():
    v_filtered = v.filter((pl.col("date") >= sdate) & (pl.col("date") <= edate)).sort(
        "date"
    )
    plt.plot(v_filtered["date"], v_filtered["Long_Short"].cum_sum(), label=k)
plt.xlim(sdate, edate)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left", frameon=True)

plt.savefig("./output/cumulative_return.png", bbox_inches="tight", dpi=300)
plt.show()
