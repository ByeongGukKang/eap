import io
import os
import pickle
import zipfile

import curl_cffi
import numpy as np
import pandas as pd
import wrds
from linearmodels.panel import PanelOLS
from scipy import stats
from statsmodels.api import OLS, add_constant
from tqdm import tqdm

# global variable for factor names
G_FACTOR_NAMES = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', 'CAT']


def _check_folder_path(origin_path: str, folder_name: str) -> str:
    if origin_path == '':
        folder_path = os.path.join(os.path.dirname(origin_path), folder_name)
        os.makedirs(folder_path, exist_ok=True)
    else:
        if not os.path.exists(origin_path):
            raise FileNotFoundError(f"{folder_name} folder path '{origin_path}' does not exist.")
        folder_path = origin_path
    return folder_path

def _calculate_grs_test(grs_data, grs_reg_results, pf_nms) -> tuple[float, np.ndarray]:
    """Performs the GRS test using the results from individual OLS regressions.

    Args:
        grs_data (pd.DataFrame): A DataFrame containing the returns of the 25 portfolios and the 7 factors, indexed by date.
        grs_reg_results (dict): A dictionary where keys are portfolio names and values are the fitted OLS regression results for that portfolio.
        pf_nms (list): A list of portfolio names corresponding to the keys in grs_reg_results.
    """
    factor_nms = list(grs_reg_results.keys())
    T, N, K = len(grs_data), len(pf_nms), len(factor_nms)

    alphas = np.array([grs_reg_results[port].params['const'] for port in pf_nms])
    resid_matrix = np.column_stack([grs_reg_results[port].resid for port in pf_nms])
    sigma_hat = np.dot(resid_matrix.T, resid_matrix) / (T-K-1)
    
    factors = grs_data[factor_nms].to_numpy()
    factor_means = factors.mean(axis=0)
    omega_hat = np.cov(factors, rowvar=False)
    
    inv_sigma_alpha = np.linalg.solve(sigma_hat, alphas)
    alpha_quad = np.dot(alphas.T, inv_sigma_alpha)
    
    inv_omega_mu = np.linalg.solve(omega_hat, factor_means)
    factor_adj = 1 + np.dot(factor_means.T, inv_omega_mu)
    
    # Final Formula
    grs_stat = ((T-N-K) / N) * (alpha_quad / factor_adj)
    
    # P-Value
    p_value = 1 - stats.f.cdf(grs_stat, N, T-N-K)
    
    return grs_stat, p_value

def get_french_data(data_folder_path: str = '') -> None:
    """Get required data from Ken French's website and save it as a parquet file.
    
    Args:
        data_folder_path (str): The folder path to save the data. If empty, it will be saved in the 'data' folder in the current directory.
    
    Note:
    - Data is downloaded from Ken French's website:
        https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
    """
    data_folder_path = _check_folder_path(data_folder_path, 'data')

    print('SYS|Downloading factor data from French\'s website')
    resp = curl_cffi.get('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip')
    with zipfile.ZipFile(io.BytesIO(resp.content), 'r') as zip_file:
        lines: list[list[str]] = []
        with zip_file.open('F-F_Research_Data_5_Factors_2x3.csv') as f:
            for line in f.readlines()[4:]:
                split = line.decode('utf-8').strip().split(',')
                if len(split) == 1:
                    break
                lines.append(split)
    cols = lines[0]
    cols[0] = 'date'
    df = pd.DataFrame(lines[1:], columns=cols) # type: ignore
    df.rename(columns={'Mkt-RF':'MKT_RF'}, inplace=True)

    print('SYS|Downloading momentum factor data from French\'s website')
    resp = curl_cffi.get('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip')
    with zipfile.ZipFile(io.BytesIO(resp.content), 'r') as zip_file:
        lines: list[list[str]] = []
        with zip_file.open('F-F_Momentum_Factor.csv') as f:
            for line in f.readlines()[13:]:
                split = line.decode('utf-8').strip().split(',')
                if len(split) == 1:
                    break
                lines.append(split)
    cols = lines[0]
    cols[0] = 'date'
    df = pd.merge(df, pd.DataFrame(lines[1:], columns=cols), on='date', how='outer')
    df.rename(columns={'Mom':'MOM'}, inplace=True)
    df = df.astype({col: 'float64' for col in df.columns if col != 'date'})
    
    # Apply same date conversion as 'get_crsp_msf_data' funciton
    df['date'] = pd.to_datetime(df['date'], format='%Y%m').dt.to_period('M').dt.to_timestamp()
    df.set_index('date', drop=True, inplace=True)
    df = df/100 # Convert from percentage to decimal
    df['RF'] = df['RF'] / 12 # Convert annual risk-free rate to monthly
    df.to_parquet(os.path.join(data_folder_path, 'rd_french_data.parquet'), index=True)

def get_crsp_msf_data(data_folder_path: str = '') -> None:
    """Get required data from WRDS and save it as a parquet file.
    
    Args:
        data_folder_path (str): The folder path to save the data. If empty, it will be saved in the 'data' folder in the current directory.
    
    Note:
    - Returns are winsorized at 1% and 99% levels within each date (-2.33, 2.33).
    - Same delisting return adjustments as Chen, Andrew Y. and Tom Zimmermann (2022) are applied.
        https://github.com/OpenSourceAP/CrossSection/blob/master/Signals/pyCode/DataDownloads/CRSPMonthly.py
    """
    data_folder_path = _check_folder_path(data_folder_path, 'data')

    print('SYS|Connecting to WRDS')
    wrds_db = wrds.Connection()
    print('SYS|Connected')
    print('SYS|Downloading CRSP monthly stock data from WRDS')
    query_result = wrds_db.connection.execute(
        """
        SELECT 
            a.permno, a.date, a.ret, 
            b.ticker, b.exchcd, b.shrcd,
            c.dlstcd, c.dlret
        FROM 
            crsp.msf AS a
        LEFT JOIN crsp.msenames as b
            ON a.permno = b.permno
            AND b.namedt <= a.date AND a.date <= b.nameendt
        LEFT JOIN 
            crsp.msedelist as c
            ON a.permno = c.permno 
            AND date_trunc('month', a.date) = date_trunc('month', c.dlstdt)
        WHERE 
            b.exchcd IN (1, 2, 3)
            AND b.shrcd IN (10, 11)
            AND a.date BETWEEN '2017-01-01' AND '2025-12-31';
        """
    )
    df = pd.DataFrame(query_result.fetchall()).astype({
        'permno': 'int64',
        'date':   'datetime64[ns]',
        'ret':    'float64',
        'ticker': 'string',
        'exchcd': 'int64',
        'shrcd':  'int64',
        'dlstcd': 'float64', 
        'dlret':  'float64', 
    })

    ### Apply delisting return adjustments according to Chen, Andrew Y. and Tom Zimmermann (2022)
    # Refer https://github.com/OpenSourceAP/CrossSection/blob/master/Signals/pyCode/DataDownloads/CRSPMonthly.py
    
    # Convert date to monthly timestamp (equivalent to Stata's mofd function)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d').dt.to_period('M').dt.to_timestamp()
    
    # Preserve original returns before delisting adjustment
    df['ret_b4_dl'] = df['ret'].copy(deep=True)

    # Process delisting returns with exchange-specific defaults
    mask1 = (
        df['dlret'].isna() 
        & (
            (df['dlstcd'] == 500) 
            | ((df['dlstcd'] >= 520) & (df['dlstcd'] <= 584))
        ) 
        & ((df['exchcd'] == 1) | (df['exchcd'] == 2)))
    df.loc[mask1, 'dlret'] = -0.35

    mask2 = (
        df['dlret'].isna() 
        & ((df['dlstcd'] == 500) | ((df['dlstcd'] >= 520) & (df['dlstcd'] <= 584))) 
        & (df['exchcd'] == 3)
    )
    df.loc[mask2, 'dlret'] = -0.55

    df.loc[(df['dlret'] < -1) & df['dlret'].notna(), 'dlret'] = -1
    df['dlret'] = df['dlret'].fillna(0)

    # Incorporate delisting returns into main returns
    df['ret'] = (1 + df['ret']) * (1 + df['dlret']) - 1
    mask3 = df['ret'].isna() & (df['dlret'] != 0)
    df.loc[mask3, 'ret'] = df.loc[mask3, 'dlret']

    # Winsorize returns at 1% and 99% levels within each date
    group_mean = df.groupby('date')['ret'].mean()
    group_std = df.groupby('date')['ret'].std()
    lower_bound = group_mean - 2.33 * group_std
    upper_bound = group_mean + 2.33 * group_std
    lower_mask = df['ret'] < lower_bound[df['date']].to_numpy()
    upper_mask = df['ret'] > upper_bound[df['date']].to_numpy()
    df.loc[upper_mask, 'ret'] = upper_bound[df['date']][upper_mask.to_numpy()].to_numpy()
    df.loc[lower_mask, 'ret'] = lower_bound[df['date']][lower_mask.to_numpy()].to_numpy()

    df.to_parquet(os.path.join(data_folder_path, 'rd_crsp_msf.parquet'), index=False)

def create_cat_factor(data_folder_path: str = '') -> None:
    """Create the CAT factor and save it as a parquet file.
    
    Args:
        data_folder_path (str): The folder path to save the data. If empty, it will be saved in the 'data' folder in the current directory.
    
    Note:
    - CAT factor is calculated as the average return of stocks with tickers starting with 'C' minus the average return of stocks with tickers starting with 'T' for each month.
    """
    print('SYS|Creating CAT factor')
    data_folder_path = _check_folder_path(data_folder_path, 'data')

    df = pd.read_parquet(os.path.join(data_folder_path, 'rd_crsp_msf.parquet'))
    wideform_ticker = df.pivot(index='date', columns='permno', values='ticker')
    wideform_ret = df.pivot(index='date', columns='permno', values='ret')

    mask_long = wideform_ticker.map(lambda x: str(x).startswith('C'))
    mask_short = wideform_ticker.map(lambda x: str(x).startswith('T'))

    cat_factor = pd.DataFrame(
        wideform_ret[mask_long].mean(axis=1) - wideform_ret[mask_short].mean(axis=1),
        index=wideform_ret.index,
        columns=['CAT']
    )

    cat_factor.to_parquet(os.path.join(data_folder_path, 'rd_cat_factor.parquet'), index=True)

def estimate_factor_loadings(data_folder_path: str = '') -> None:
    """Estimate factor loadings for each stock using the seven-factor model.
    
    Args:
        data_folder_path (str): The folder path where the data is saved. If empty, it will be read from the 'data' folder in the current directory.
    """
    print('SYS|Estimating factor loadings')
    data_folder_path = _check_folder_path(data_folder_path, 'data')

    ff_factors = pd.read_parquet(os.path.join(data_folder_path, 'rd_french_data.parquet'))
    cat_factor = pd.read_parquet(os.path.join(data_folder_path, 'rd_cat_factor.parquet'))
    agg_factors = ff_factors.merge(cat_factor, left_index=True, right_index=True)
    crsp_data = pd.read_parquet(os.path.join(data_folder_path, 'rd_crsp_msf.parquet'))
    
    wideform_ret = crsp_data.pivot(index='date', columns='permno', values='ret')

    factor_loadings = {f:np.full(wideform_ret.shape, np.nan, dtype=np.float64) for f in G_FACTOR_NAMES}
    factor_loadings['const'] = np.full(wideform_ret.shape, np.nan, dtype=np.float64)

    arr_ret_all = np.asfortranarray(wideform_ret.to_numpy())
    arr_factors = np.ascontiguousarray(agg_factors[G_FACTOR_NAMES].to_numpy())

    for idx_stk in tqdm(range(arr_ret_all.shape[1]), desc='Rolling window regression'):
        arr_ret_ith = arr_ret_all[:, idx_stk]
        for idx_time in range(24, arr_ret_all.shape[0]):
            y = arr_ret_ith[idx_time-24:idx_time] # Python indexing convention [:T] means up to T-1
            notna_mask = ~np.isnan(y)
            if notna_mask.sum() < 10:
                # Drop if there are fewer than 10 months of data in the estimation window.
                continue
            x = arr_factors[idx_time-24:idx_time, :]
            y = y[notna_mask]
            x = x[notna_mask, :]
            ols_result = OLS(y, add_constant(x)).fit()
            params = ols_result.params

            factor_loadings['const'][idx_time, idx_stk] = params[0]
            for f_idx, f_name in enumerate(G_FACTOR_NAMES):
                factor_loadings[f_name][idx_time, idx_stk] = params[f_idx+1] # +1 to skip the constant term

    for f_name, f_loadings in factor_loadings.items():
        pd.DataFrame(f_loadings, index=wideform_ret.index, columns=wideform_ret.columns).to_parquet(os.path.join(data_folder_path, f'F_loading_{f_name}.parquet'), index=True)
        
def run_regressions(data_folder_path: str = '', result_folder_path: str = '') -> None:
    """Run the specified cross-sectional regressions and save the results.

    Args:
        data_folder_path (str): The folder path where the data is saved. If empty, it will be read from the 'data' folder in the current directory.
        result_folder_path (str): The folder path to save the regression results. If empty, it will be saved in the 'result' folder in the current directory.
    """
    data_folder_path = _check_folder_path(data_folder_path, 'data')
    result_folder_path = _check_folder_path(result_folder_path, 'result')

    # Load data
    print('SYS|Loading data for regressions')
    factor_loadings = {f:pd.read_parquet(os.path.join(data_folder_path, f"F_loading_{f}.parquet")) for f in G_FACTOR_NAMES}
    factor_loadings['const'] = pd.read_parquet(os.path.join(data_folder_path, "F_loading_const.parquet"))
    crsp_data = pd.read_parquet(os.path.join(data_folder_path, 'rd_crsp_msf.parquet'))
    ff_factors = pd.read_parquet(os.path.join(data_folder_path, 'rd_french_data.parquet'))
    cat_factor = pd.read_parquet(os.path.join(data_folder_path, 'rd_cat_factor.parquet'))
    agg_factors = ff_factors.merge(cat_factor, left_index=True, right_index=True)

    # Prepare returns
    wideform_ret = crsp_data.pivot(index='date', columns='permno', values='ret')
    wideform_retex = wideform_ret.apply(lambda x: x - agg_factors['RF'], axis=0)
    wideform_retex = wideform_retex.loc[agg_factors.index]
    longform_retex = wideform_retex.stack().reset_index(name='retex') # type: ignore
    
    # Prepare factor loadings
    factor_loadings = {}
    for f_name in G_FACTOR_NAMES:
        factor_loadings[f_name] = pd.read_parquet(os.path.join(data_folder_path, f"F_loading_{f_name}.parquet")).stack().rename(f_name) # type: ignore
    longform_factor_loadings = pd.concat(factor_loadings.values(), axis=1).reset_index()

    # Prepre OLS data by merging returns and factor loadings
    ols_data = longform_retex.merge(longform_factor_loadings, on=['date', 'permno'], how='left')
    ols_data = ols_data.dropna(subset=G_FACTOR_NAMES + ['retex']) # Drop rows with missing values in factor loadings or returns
    ols_data.set_index(['date', 'permno'], inplace=True)

    ### Pooled OLS regressions
    ols_y = ols_data['retex']
    ols_x = ols_data[G_FACTOR_NAMES]

    # 1) Simple pooled OLS regression
    print('SYS|Running simple pooled OLS regression')
    ols_res_simple = OLS(ols_y, add_constant(ols_x)).fit()
    with open(os.path.join(result_folder_path, 'ols_pooled_simple.pickle'), 'wb') as f:
        pickle.dump(ols_res_simple, f)

    # 3) Pooled OLS regression with fixed effects
    print('SYS|Running pooled OLS regression with fixed effects')
    ols_res_fe = PanelOLS(ols_y, ols_x, entity_effects=True, time_effects=True).fit()
    with open(os.path.join(result_folder_path, 'ols_pooled_fe.pickle'), 'wb') as f:
        pickle.dump(ols_res_fe, f)

    # 4) Pooled OLS regression with fixed effects and clustered standard errors
    print('SYS|Running pooled OLS regression with fixed effects and clustered standard errors')
    ols_res_fe_clustered = PanelOLS(ols_y, ols_x, entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
    with open(os.path.join(result_folder_path, 'ols_pooled_fe_clustered.pickle'), 'wb') as f:
        pickle.dump(ols_res_fe_clustered, f)

    ### Fama-MacBeth regression
    print('SYS|Running Fama-MacBeth regression')
    fm_results = {}
    for f_name in G_FACTOR_NAMES:
        y_date = ols_data['retex']
        x_date = ols_data[f_name]
        ols_result = OLS(y_date, add_constant(x_date)).fit(cov_type='HAC', cov_kwds={'maxlags': 6}) # Newey-West standard errors with 6 lags
        fm_results[f_name] = ols_result
    with open(os.path.join(result_folder_path, 'fama_macbeth.pickle'), 'wb') as f:
        pickle.dump(fm_results, f)


def run_grs_test(data_folder_path: str = '', result_folder_path: str = '') -> None:
    """Run the GRS test and save the results.

    Args:
        data_folder_path (str): The folder path where the data is saved. If empty, it will be read from the 'data' folder in the current directory.
        result_folder_path (str): The folder path to save the GRS test results. If empty, it will be saved in the 'result' folder in the current directory.
    """
    data_folder_path = _check_folder_path(data_folder_path, 'data')
    result_folder_path = _check_folder_path(result_folder_path, 'result')

    # Load data
    print('SYS|Loading data for GRS test')
    factor_loadings = {f:pd.read_parquet(os.path.join(data_folder_path, f"F_loading_{f}.parquet")) for f in G_FACTOR_NAMES}
    factor_loadings['const'] = pd.read_parquet(os.path.join(data_folder_path, "F_loading_const.parquet"))
    crsp_data = pd.read_parquet(os.path.join(data_folder_path, 'rd_crsp_msf.parquet'))
    ff_factors = pd.read_parquet(os.path.join(data_folder_path, 'rd_french_data.parquet'))
    cat_factor = pd.read_parquet(os.path.join(data_folder_path, 'rd_cat_factor.parquet'))
    agg_factors = ff_factors.merge(cat_factor, left_index=True, right_index=True)

    # Prepare returns
    wideform_ret = crsp_data.pivot(index='date', columns='permno', values='ret')
    wideform_retex = wideform_ret.apply(lambda x: x - agg_factors['RF'], axis=0)
    wideform_retex = wideform_retex.loc[agg_factors.index]
    longform_retex = wideform_retex.stack().reset_index(name='retex') # type: ignore
    
    # Prepare factor loadings
    factor_loadings = {}
    for f_name in G_FACTOR_NAMES:
        factor_loadings[f_name] = pd.read_parquet(os.path.join(data_folder_path, f"F_loading_{f_name}.parquet")).stack().rename(f_name) # type: ignore
    longform_factor_loadings = pd.concat(factor_loadings.values(), axis=1).reset_index()

    # Prepre OLS data by merging returns and factor loadings
    ols_data = longform_retex.merge(longform_factor_loadings, on=['date', 'permno'], how='left')
    ols_data = ols_data.dropna(subset=G_FACTOR_NAMES + ['retex']) # Drop rows with missing values in factor loadings or returns
    ols_data.set_index(['date', 'permno'], inplace=True)

    def assign_q(group):
        group['HML_Q'] = pd.qcut(group['HML'], 5, labels=False, duplicates='drop') + 1
        group['CAT_Q'] = pd.qcut(group['CAT'], 5, labels=False, duplicates='drop') + 1
        return group
    
    # Assign quintiles for HML and CAT factors within each date
    sorted_data = ols_data.groupby('date', group_keys=False).apply(assign_q)
    longform_pf_returns = sorted_data.groupby(['date', 'HML_Q', 'CAT_Q'])['retex'].mean().reset_index()
    longform_pf_returns['port_id'] = 'HML' + longform_pf_returns['HML_Q'].astype(str) + '_CAT' + longform_pf_returns['CAT_Q'].astype(str)
    wideform_pf_returns = longform_pf_returns.pivot(index='date', columns='port_id', values='retex')

    ### GRS Test
    grs_data = pd.concat([wideform_pf_returns, agg_factors[G_FACTOR_NAMES]], axis=1).dropna()

    ### GRS Test with all 7 factors 
    print('SYS|Running GRS test with all 7 factors')
    # Prepare data for GRS test by merging portfolio returns and factor data
    grs_x = add_constant(grs_data[G_FACTOR_NAMES]) # The 7 Factors + Alpha (Intercept)

    grs_reg_results = {}
    for pf_nm in wideform_pf_returns.columns:
        grs_y = grs_data[pf_nm]
        grs_reg_results[pf_nm] = OLS(grs_y, grs_x).fit()

    # GRS test statistic calculation
    grs_stat, p_val = _calculate_grs_test(grs_data, grs_reg_results, wideform_pf_returns.columns.tolist())

    # Save GRS test results
    with open(os.path.join(result_folder_path, 'grs_all.pickle'), 'wb') as f:
        pickle.dump({'grs_reg': grs_reg_results, 'grs_stat': grs_stat, 'p_value': p_val}, f)

    ### GRS TEST without CAT factor (only 6 factors)
    print('SYS|Running GRS test without CAT factor')
    grs_x = add_constant(grs_data[[f for f in G_FACTOR_NAMES if f != 'CAT']])

    grs_reg_results = {}
    for pf_nm in wideform_pf_returns.columns:
        grs_y = grs_data[pf_nm]
        grs_reg_results[pf_nm] = OLS(grs_y, grs_x).fit()
    
    # GRS test statistic calculation
    grs_stat, p_val = _calculate_grs_test(grs_data, grs_reg_results, wideform_pf_returns.columns.tolist())

    # Save GRS test results
    with open(os.path.join(result_folder_path, 'grs_no_cat.pickle'), 'wb') as f:
        pickle.dump({'grs_reg': grs_reg_results, 'grs_stat': grs_stat, 'p_value': p_val}, f)