from typing import Literal

import numpy as np
import polars as pl
import torch
from tqdm import tqdm


def calculate_model_oos_r2(
    models_dict: dict,
    x_df: pl.DataFrame,
    y_df: pl.DataFrame,
    model_type: Literal["linear", "pcr", "rf", "nn"],
    model_name: str,
) -> float:
    """
    Gu, Kelly, Xiu (2020) Grand Panel OOS R2

    Args:
        models_dict: {year: model} dictionary
        x_df: feature df with date and permno columns
        y_df: return df with date and ret columns
        model_type: ['linear', 'pcr', 'rf', 'nn']
        model_name: model name (from tqdm)
    """
    feat_cols = [c for c in x_df.columns if c not in ["date", "permno"]]
    years = x_df["date"].dt.year()
    test_years = sorted(models_dict.keys())  # 2014 ~ 2024

    all_preds = []
    all_actuals = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for test_year in tqdm(test_years, desc=model_name):
        test_mask = years == test_year
        x_test = x_df.filter(test_mask).select(feat_cols).to_numpy()
        y_test = y_df.filter(test_mask)["ret"].to_numpy()

        if len(x_test) == 0:
            continue

        model = models_dict[test_year]

        match model_type:
            case "linear":
                pred = model.predict(x_test)
            case "rf":
                pred = model.predict(x_test)
            case "pcr":
                pred = model["reg"].predict(model["pca"].transform(x_test))
            case "nn":
                model_list = models_dict[test_year]
                seed_preds = []
                for model in model_list:
                    model.to(device)
                    with torch.no_grad():
                        seed_preds.append(
                            model(torch.tensor(x_test, dtype=torch.float32).to(device))
                            .cpu()
                            .numpy()
                        )
                # Ensemble forecast
                pred = np.mean(seed_preds, axis=0)
            case _:
                raise Exception(f"Unknown model type: {model_type}")

        all_preds.append(pred)
        all_actuals.append(y_test)

    actuals = np.concatenate(all_actuals)
    preds = np.concatenate(all_preds)

    numerator = np.sum((actuals - preds) ** 2)
    denominator = np.sum(actuals**2)

    r2_oos = 1 - (numerator / denominator)

    return r2_oos


def calculate_variable_importance(
    models_dict: dict, x_df: pl.DataFrame, y_df: pl.DataFrame, model_type: str
) -> pl.DataFrame:
    """
    Gu, Kelly, Xiu (2020) Feature Importance (per feature)

    Args:
        models_dict: {year: model} dictionary
        x_df: feature df with date and permno columns
        y_df: return df with date and ret columns
        model_type: ['linear', 'pcr', 'rf', 'nn']
    """
    feature_cols = [c for c in x_df.columns if c not in ["date", "permno"]]
    num_features = len(feature_cols)
    test_years = sorted(models_dict.keys())
    years = x_df["date"].dt.year()

    yearly_importances = []

    if model_type in ["linear", "pcr", "rf"]:
        for test_year in test_years:
            model = models_dict[test_year]

            if model_type == "linear":
                importance = np.abs(model.coef_)

            elif model_type == "rf":
                importance = model.feature_importances_

            elif model_type == "pcr":
                pca_comp = model["pca"].components_  # (K, Num_Features)
                reg_coef = model["reg"].coef_  # (K,)
                importance = np.abs(reg_coef @ pca_comp)
            else:
                raise Exception(f"Unknown model type: {model_type}")

            yearly_importances.append(importance)

    elif model_type == "nn":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for test_year in test_years:
            test_mask = years == test_year
            x_test = x_df.filter(test_mask).select(feature_cols).to_numpy()
            y_test = y_df.filter(test_mask)["ret"].to_numpy()

            if len(x_test) == 0:
                continue

            model_list = models_dict[test_year]
            x_test_t = torch.tensor(x_test, dtype=torch.float32).to(device)

            base_preds = []
            for m in model_list:
                m.to(device)
                with torch.no_grad():
                    base_preds.append(m(x_test_t).cpu().numpy())
            base_pred_avg = np.mean(base_preds, axis=0)
            base_mse = np.mean((y_test - base_pred_avg) ** 2)

            nn_importance = np.zeros(num_features)

            for j in tqdm(
                range(num_features), desc=f"NN Importance ({test_year})", leave=False
            ):
                x_test_perturbed = x_test_t.clone()
                x_test_perturbed[:, j] = 0.0

                perturbed_preds = []
                for m in model_list:
                    with torch.no_grad():
                        perturbed_preds.append(m(x_test_perturbed).cpu().numpy())
                perturbed_pred_avg = np.mean(perturbed_preds, axis=0)
                perturbed_mse = np.mean((y_test - perturbed_pred_avg) ** 2)

                nn_importance[j] = max(0.0, perturbed_mse - base_mse)

            if np.sum(nn_importance) > 0:
                nn_importance = nn_importance / np.sum(nn_importance)

            yearly_importances.append(nn_importance)

    else:
        raise Exception(f"Unknown model type: {model_type}")

    # Take the mean importance across all years
    final_importance_vector = np.mean(yearly_importances, axis=0)

    df_importance = pl.DataFrame(
        {"feature": feature_cols, "importance": final_importance_vector}
    ).sort("importance", descending=True)

    return df_importance


def aggregate_importance_by_characteristic(
    df_importance: pl.DataFrame, base_characteristics: list[str]
) -> pl.DataFrame:
    """Aggregate feature importance by characteristic."""
    all_features = df_importance["feature"].to_list()
    feature_to_base = {}

    sorted_base_chars = sorted(base_characteristics, key=len, reverse=True)

    for col in all_features:
        if col.startswith("sic2_") or col == "sic2":
            feature_to_base[col] = "sic2"
        else:
            matched = False
            for char in sorted_base_chars:
                # feature name == characteristic or starts with characteristic_(case of macro variable merged)
                if col == char or col.startswith(f"{char}_"):
                    feature_to_base[col] = char
                    matched = True
                    break
            if not matched:
                feature_to_base[col] = col

    df_aggregated = (
        df_importance.with_columns(
            pl.col("feature").replace(feature_to_base).alias("characteristic")
        )
        .group_by("characteristic")
        .agg(pl.col("importance").sum())
        .sort("importance", descending=True)
    )

    return df_aggregated


def calculate_oos_decile_portfolio(
    models_dict: dict, x_df: pl.DataFrame, y_df: pl.DataFrame, model_type: str
) -> pl.DataFrame:
    """
    Generate decile equal-weighted portfolio and Long-Short (D10 - D1) return time series for each model.

    Args:
        models_dict: {year: model} dictionary
        x_df: feature df with date and permno columns
        y_df: return df with date and ret columns
        model_type: ['linear', 'pcr', 'rf', 'nn']
    """
    feat_cols = [c for c in x_df.columns if c not in ["date", "permno"]]
    years = x_df["date"].dt.year()
    test_years = sorted(models_dict.keys())  # 2014 ~ 2024

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_year_results = []

    for test_year in test_years:
        test_mask = years == test_year

        x_test = x_df.filter(test_mask).select(feat_cols).to_numpy()
        meta_test = x_df.filter(test_mask).select(["date", "permno"])
        y_test_ret = y_df.filter(test_mask)["ret"].to_numpy()

        if len(x_test) == 0:
            continue

        model = models_dict[test_year]

        match model_type:
            case "linear":
                pred = model.predict(x_test)
            case "rf":
                pred = model.predict(x_test)
            case "pcr":
                x_test_pca = model["pca"].transform(x_test)
                pred = model["reg"].predict(x_test_pca)
            case "nn":
                model_list = model
                seed_preds = []
                x_test_t = torch.tensor(x_test, dtype=torch.float32).to(device)
                for m in model_list:
                    m.to(device)
                    with torch.no_grad():
                        seed_preds.append(m(x_test_t).cpu().numpy())
                pred = np.mean(seed_preds, axis=0)
            case _:
                raise Exception(f"Unknown model type: {model_type}")

        year_df = meta_test.with_columns(
            [pl.lit(y_test_ret).alias("ret"), pl.lit(pred).alias("pred")]
        )
        all_year_results.append(year_df)

    oos_grand_df = pl.concat(all_year_results)
    rank_expr = pl.col("pred").rank(method="ordinal").over("date")
    count_expr = pl.col("pred").count().over("date")

    oos_grand_df = oos_grand_df.with_columns(
        (((rank_expr - 1) * 10 / count_expr).cast(pl.Int8) + 1).alias("decile")
    )

    # aggregate returns by date and decile
    portfolio_ts = (
        oos_grand_df.group_by(["date", "decile"])
        .agg(pl.col("ret").mean().alias("ew_ret"))
        .sort(["date", "decile"])
    )

    portfolio_wide = portfolio_ts.pivot(on="decile", index="date", values="ew_ret")

    # rename columns from 1~10 to D1~D10
    rename_dict = {str(i): f"D{i}" for i in range(1, 11)}
    portfolio_wide = portfolio_wide.rename(rename_dict)

    # add Long-Short column (D10 - D1)
    portfolio_wide = portfolio_wide.with_columns(
        (pl.col("D10") - pl.col("D1")).alias("Long_Short")
    )

    return portfolio_wide.sort("date")
