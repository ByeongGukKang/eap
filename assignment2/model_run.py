import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as torch_optim
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression, SGDRegressor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def filter_nan(x_train: np.ndarray, y_train: np.ndarray):
    x_nan_mask = np.isnan(x_train).any(axis=1)
    y_nan_mask = np.isnan(y_train)
    valid_mask = ~(x_nan_mask | y_nan_mask)
    x_train = x_train[valid_mask]
    y_train = y_train[valid_mask]
    return x_train, y_train


def run_ols_huber(x_df: pl.DataFrame, y_df: pl.DataFrame, seed: int) -> dict:
    models = {}
    feature_cols = [c for c in x_df.columns if c not in ["date", "permno"]]
    years = x_df["date"].dt.year()
    test_years = sorted([y for y in set(years.to_list()) if y >= 2014])

    for test_year in tqdm(test_years, desc="OLS+H"):
        train_mask = years < test_year - 4

        x_train = x_df.filter(train_mask).select(feature_cols).to_numpy()
        y_train = y_df.filter(train_mask)["ret"].to_numpy()
        x_train, y_train = filter_nan(x_train, y_train)

        model = HuberRegressor(max_iter=300, epsilon=1.35)
        model.fit(x_train, y_train)
        models[test_year] = model

    return models


def run_ols3_huber(x_df: pl.DataFrame, y_df: pl.DataFrame, seed: int) -> dict:
    models = {}
    ols3_cols = ["mve_m", "bm", "mom12m"]
    years = x_df["date"].dt.year()
    test_years = sorted([y for y in set(years.to_list()) if y >= 2014])

    for test_year in tqdm(test_years, desc="OLS-3+H"):
        train_mask = years < test_year - 4

        x_train = x_df.filter(train_mask).select(ols3_cols).to_numpy()
        y_train = y_df.filter(train_mask)["ret"].to_numpy()
        x_train, y_train = filter_nan(x_train, y_train)

        model = HuberRegressor(max_iter=100)
        model.fit(x_train, y_train)
        models[test_year] = model

    return models


def run_pcr(x_df: pl.DataFrame, y_df: pl.DataFrame, seed: int) -> dict:
    models = {}
    feature_cols = [c for c in x_df.columns if c not in ["date", "permno"]]
    years = x_df["date"].dt.year()
    test_years = sorted([y for y in set(years.to_list()) if y >= 2014])
    k_grid = [5, 10, 20, 30, 50, 75]

    for test_year in tqdm(test_years, desc="PCR"):
        train_mask = years < test_year - 4
        val_mask = (years >= test_year - 4) & (years < test_year)

        x_train = x_df.filter(train_mask).select(feature_cols).to_numpy()
        y_train = y_df.filter(train_mask)["ret"].to_numpy()
        x_train, y_train = filter_nan(x_train, y_train)

        x_val = x_df.filter(val_mask).select(feature_cols).to_numpy()
        y_val = y_df.filter(val_mask)["ret"].to_numpy()
        x_val, y_val = filter_nan(x_val, y_val)

        best_val_mse = float("inf")
        best_pca = None

        for k in k_grid:
            pca = PCA(n_components=k, random_state=seed)
            x_train_pca = pca.fit_transform(x_train)
            x_val_pca = pca.transform(x_val)

            reg = LinearRegression()
            reg.fit(x_train_pca, y_train)
            val_mse = np.mean((y_val - reg.predict(x_val_pca)) ** 2)

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_pca = pca

        assert best_pca is not None, "No best PCA found"
        x_train_pca_final = best_pca.fit_transform(x_train)
        final_reg = LinearRegression()
        final_reg.fit(x_train_pca_final, y_train)

        # return both best_pca and final_reg for Variable Importance
        models[test_year] = {"pca": best_pca, "reg": final_reg}

    return models


def run_enet_huber(x_df: pl.DataFrame, y_df: pl.DataFrame, seed: int) -> dict:
    models = {}
    feature_cols = [c for c in x_df.columns if c not in ["date", "permno"]]
    years = x_df["date"].dt.year()
    test_years = sorted([y for y in set(years.to_list()) if y >= 2014])
    param_grid = [
        {"alpha": 1e-4, "l1_ratio": 0.1},
        {"alpha": 1e-3, "l1_ratio": 0.5},
        {"alpha": 1e-2, "l1_ratio": 0.9},
    ]

    for test_year in tqdm(test_years, desc="ENet+H"):
        train_mask = years < test_year - 4
        val_mask = (years >= test_year - 4) & (years < test_year)

        x_train = x_df.filter(train_mask).select(feature_cols).to_numpy()
        y_train = y_df.filter(train_mask)["ret"].to_numpy()
        x_train, y_train = filter_nan(x_train, y_train)

        x_val = x_df.filter(val_mask).select(feature_cols).to_numpy()
        y_val = y_df.filter(val_mask)["ret"].to_numpy()
        x_val, y_val = filter_nan(x_val, y_val)

        best_params = param_grid[0]
        best_val_mse = float("inf")

        for params in param_grid:
            model = SGDRegressor(
                loss="huber",
                penalty="elasticnet",
                alpha=params["alpha"],
                l1_ratio=params["l1_ratio"],
                max_iter=100,
                random_state=seed,
            )
            model.fit(x_train, y_train)
            val_mse = np.mean((y_val - model.predict(x_val)) ** 2)

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_params = params

        final_model = SGDRegressor(
            loss="huber",
            penalty="elasticnet",
            alpha=best_params["alpha"],
            l1_ratio=best_params["l1_ratio"],
            max_iter=100,
            random_state=seed,
        )
        final_model.fit(x_train, y_train)
        models[test_year] = final_model

    return models


def run_random_forest(x_df: pl.DataFrame, y_df: pl.DataFrame, seed: int) -> dict:
    models = {}
    feature_cols = [c for c in x_df.columns if c not in ["date", "permno"]]
    years = x_df["date"].dt.year()
    test_years = sorted([y for y in set(years.to_list()) if y >= 2014])
    depth_grid = [1, 2, 3, 5]

    for test_year in tqdm(test_years, desc="RF"):
        train_mask = years < test_year - 4
        val_mask = (years >= test_year - 4) & (years < test_year)

        x_train = x_df.filter(train_mask).select(feature_cols).to_numpy()
        y_train = y_df.filter(train_mask)["ret"].to_numpy()
        x_train, y_train = filter_nan(x_train, y_train)

        x_val = x_df.filter(val_mask).select(feature_cols).to_numpy()
        y_val = y_df.filter(val_mask)["ret"].to_numpy()
        x_val, y_val = filter_nan(x_val, y_val)

        best_depth = depth_grid[0]
        best_val_mse = float("inf")

        for depth in depth_grid:
            model = RandomForestRegressor(
                n_estimators=100, max_depth=depth, n_jobs=-1, random_state=seed
            )
            model.fit(x_train, y_train)
            val_mse = np.mean((y_val - model.predict(x_val)) ** 2)

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_depth = depth

        final_model = RandomForestRegressor(
            n_estimators=100, max_depth=best_depth, n_jobs=-1, random_state=seed
        )
        final_model.fit(x_train, y_train)
        models[test_year] = final_model

    return models


# PyTorch Baseline NN class
class AssetPricingMLP(nn.Module):
    def __init__(self, input_dim, layers):
        super().__init__()
        net_layers = []
        curr_dim = input_dim
        for next_dim in layers:
            net_layers.append(nn.Linear(curr_dim, next_dim))
            net_layers.append(nn.ReLU())
            curr_dim = next_dim
        net_layers.append(nn.Linear(curr_dim, 1))
        self.network = nn.Sequential(*net_layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def train_nn_engine(
    x_train,
    y_train,
    x_val,
    y_val,
    hidden_layers,
    seed,
    epochs=30,
    batch_size=2048,
    patience=3,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    model = AssetPricingMLP(x_train.shape[1], hidden_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch_optim.Adam(model.parameters(), lr=1e-5)

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val_t), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    assert best_weights is not None, "No best weights found"
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    return model.eval()


def run_nn_ensemble(
    x_df: pl.DataFrame,
    y_df: pl.DataFrame,
    hidden_layers: list[int],
    seeds: list[int],
    desc: str,
) -> dict:
    """
    Args:
        x_df: Input features DataFrame
        y_df: Target DataFrame
        hidden_layers: List of hidden layer sizes, [32, 16], [32, 16, 8, 4]
        seeds: List of random seeds
        desc: Description for the progress bar
    Returns:
        Dictionary of list[model] for each test year
    """
    models = {}
    feat_cols = [c for c in x_df.columns if c not in ["date", "permno"]]
    years = x_df["date"].dt.year()
    test_years = sorted([y for y in set(years.to_list()) if y >= 2014])

    for test_year in tqdm(test_years, desc=desc):
        train_mask = (years >= 2004) & (years < test_year - 4)
        val_mask = (years >= test_year - 4) & (years < test_year)

        x_train = x_df.filter(train_mask).select(feat_cols).to_numpy()
        y_train = y_df.filter(train_mask)["ret"].to_numpy()
        x_train, y_train = filter_nan(x_train, y_train)

        x_val = x_df.filter(val_mask).select(feat_cols).to_numpy()
        y_val = y_df.filter(val_mask)["ret"].to_numpy()
        x_val, y_val = filter_nan(x_val, y_val)

        year_models = []
        for seed in seeds:
            model = train_nn_engine(
                x_train, y_train, x_val, y_val, hidden_layers=hidden_layers, seed=seed
            )
            year_models.append(model)

        models[test_year] = year_models

    return models
