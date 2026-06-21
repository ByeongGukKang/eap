# %%
import pickle

import polars as pl
from model_run import (
    run_enet_huber,
    run_nn_ensemble,
    run_ols3_huber,
    run_ols_huber,
    run_pcr,
    run_random_forest,
)
from utils import characteristics_dict, scan_ghz, scan_macro

# %%
# Read data
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
print("Number of features:", len(feature_cols))
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
# OLS+Huber
model_ols = run_ols_huber(
    df_data.select(["date"] + feature_cols),
    df_data.select(["date", "ret"]),
    GLOBAL_SEED,
)
pickle.dump(model_ols, open("./model/ols.pkl", "wb"))

# %%
# OLS3+Huber
model_ols3 = run_ols3_huber(
    df_data.select(["date"] + feature_cols),
    df_data.select(["date", "ret"]),
    GLOBAL_SEED,
)
pickle.dump(model_ols3, open("./model/ols3.pkl", "wb"))

# %%
# PCR
model_pcr = run_pcr(
    df_data.select(["date"] + feature_cols),
    df_data.select(["date", "ret"]),
    GLOBAL_SEED,
)
pickle.dump(model_pcr, open("./model/pcr.pkl", "wb"))

# %%
# ENet+Huber
model_enet = run_enet_huber(
    df_data.select(["date"] + feature_cols),
    df_data.select(["date", "ret"]),
    GLOBAL_SEED,
)
pickle.dump(model_enet, open("./model/enet.pkl", "wb"))

# %%
# RF
model_rf = run_random_forest(
    df_data.select(["date"] + feature_cols),
    df_data.select(["date", "ret"]),
    GLOBAL_SEED,
)
pickle.dump(model_rf, open("./model/rf.pkl", "wb"))

# %%
# NN2
model_nn2 = run_nn_ensemble(
    df_data.select(["date"] + feature_cols),
    df_data.select(["date", "ret"]),
    [32, 16],
    [i for i in range(GLOBAL_SEED + 10)],
    "NN2",
)
pickle.dump(model_nn2, open("./model/nn2.pkl", "wb"))

# %%
# NN4
model_nn4 = run_nn_ensemble(
    df_data.select(["date"] + feature_cols),
    df_data.select(["date", "ret"]),
    [32, 16, 8, 4],
    [i for i in range(GLOBAL_SEED + 10)],
    "NN4",
)
pickle.dump(model_nn4, open("./model/nn4.pkl", "wb"))
