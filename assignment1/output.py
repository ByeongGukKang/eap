import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from impl import G_FACTOR_NAMES, _check_folder_path


def generate_reg_latex_table(result_folder_path: str = '', output_folder_path: str = '') -> pd.DataFrame:
    """Generate a LaTeX table summarizing the regression results from the specified result folder.
    
    Args:
        result_folder_path (str): Path to the folder containing the regression result pickle files. 
                                  If empty, defaults to './result' relative to this script.
    """
    result_folder_path = _check_folder_path(result_folder_path, 'result')
    output_folder_path = _check_folder_path(output_folder_path, 'output')

    # 1. Load results
    with open(os.path.join(result_folder_path, 'ols_pooled_simple.pickle'), 'rb') as f:
        res_simple = pickle.load(f)
    with open(os.path.join(result_folder_path, 'ols_pooled_fe.pickle'), 'rb') as f:
        res_fe = pickle.load(f)
    with open(os.path.join(result_folder_path, 'ols_pooled_fe_clustered.pickle'), 'rb') as f:
        res_fe_clustered = pickle.load(f)
    with open(os.path.join(result_folder_path, 'fama_macbeth.pickle'), 'rb') as f:
        res_fm_dict = pickle.load(f)

    # 2. Extract Data Helper Function
    def get_stars(p):
        if p < 0.01: return '^{***}'
        if p < 0.05: return '^{**}'
        if p < 0.10: return '^{*}'
        return ''

    def format_val(val, se, p):
        star = get_stars(p)
        return f"{val:.4f}{star} \\\\ ({se:.4f})"

    # 3. Compile Model Statistics
    # For Fama-MacBeth, we usually report the mean gamma and NW t-stats
    # Note: Your code ran 7 individual regressions, so we consolidate them here
    factors = list(res_fm_dict.keys())
    all_rows = factors + ['const', 'N', 'R2', 'Fixed Effects', 'Clustering']
    
    table_data = pd.DataFrame(index=all_rows, columns=['(1)', '(2)', '(3)', '(4)'])

    # Fill Column (1): Simple Pooled
    for f in factors + ['const']:
        if f in res_simple.params:
            table_data.loc[f, '(1)'] = format_val(res_simple.params[f], res_simple.bse[f], res_simple.pvalues[f])
    table_data.loc['N', '(1)'] = f"{int(res_simple.nobs):,}"
    table_data.loc['R2', '(1)'] = f"{res_simple.rsquared:.3f}"
    table_data.loc['Fixed Effects', '(1)'] = 'No'
    table_data.loc['Clustering', '(1)'] = 'No'

    # Fill Column (2): Fama-MacBeth
    # Note: In Fama-MacBeth, coefficients are the means of the monthly gammas
    for f in factors:
        res = res_fm_dict[f]
        # Using the constant (gamma) from your specific factor-by-factor loop
        # Usually FM is run as a multivariate regression, but we follow your code logic
        val = res.params.iloc[1] # the factor coefficient
        se = res.bse.iloc[1]
        p = res.pvalues.iloc[1]
        table_data.loc[f, '(2)'] = format_val(val, se, p)
    table_data.loc['N', '(2)'] = "T-avg"
    table_data.loc['R2', '(2)'] = "-"
    table_data.loc['Fixed Effects', '(2)'] = 'No'
    table_data.loc['Clustering', '(2)'] = 'Newey-West'

    # Fill Column (3): Pooled FE
    for f in factors:
        table_data.loc[f, '(3)'] = format_val(res_fe.params[f], res_fe.std_errors[f], res_fe.pvalues[f])
    table_data.loc['N', '(3)'] = f"{int(res_fe.nobs):,}"
    table_data.loc['R2', '(3)'] = f"{res_fe.rsquared:.3f}"
    table_data.loc['Fixed Effects', '(3)'] = 'Two-Way'
    table_data.loc['Clustering', '(3)'] = 'No'

    # Fill Column (4): Pooled FE Clustered
    for f in factors:
        table_data.loc[f, '(4)'] = format_val(res_fe_clustered.params[f], res_fe_clustered.std_errors[f], res_fe_clustered.pvalues[f])
    table_data.loc['N', '(4)'] = f"{int(res_fe_clustered.nobs):,}"
    table_data.loc['R2', '(4)'] = f"{res_fe_clustered.rsquared:.3f}"
    table_data.loc['Fixed Effects', '(4)'] = 'Two-Way'
    table_data.loc['Clustering', '(4)'] = 'Two-Way'

    # 4. Generate LaTeX String
    latex = table_data.fillna('').to_latex(
        escape=False, 
        column_format='lcccc',
        caption='Cross-Sectional Stock Return Regressions',
        label='tab:regression_results'
    )
    
    # Replace the standard row endings to allow for the newline in format_val
    latex = latex.replace('\\\\', '\\\\[4pt]') # Adds some vertical spacing
    
    with open(os.path.join(output_folder_path, 'regression_table.tex'), 'w') as f:
        f.write(latex)
    
    print(f"SYS|LaTeX table saved to {output_folder_path}")
    return table_data

def generate_cat_summary_table(data_folder_path: str = '', result_folder_path: str = '', output_folder_path: str = '') -> pd.DataFrame:
    """Generate a summary table for the CAT factor, including mean, std, and correlations with other factors.
    
    Args:
        data_folder_path (str): The folder path where the data is saved. If empty, it will be read from the 'data' folder in the current directory.
        result_folder_path (str): The folder path to save the regression results. If empty, it will be saved in the 'result' folder in the current directory.
        output_folder_path (str): The folder path to save the output LaTeX table. If empty, it will be saved in the 'output' folder in the current directory.
    """
    data_folder_path = _check_folder_path(data_folder_path, 'data')
    result_folder_path = _check_folder_path(result_folder_path, 'result')
    output_folder_path = _check_folder_path(output_folder_path, 'output')

    # Ensure 'CAT' is included with the other factors
    ff_factors = pd.read_parquet(os.path.join(data_folder_path, 'rd_french_data.parquet'))
    cat_factor = pd.read_parquet(os.path.join(data_folder_path, 'rd_cat_factor.parquet'))
    agg_factors = ff_factors.merge(cat_factor, left_index=True, right_index=True)
    df = agg_factors[G_FACTOR_NAMES]

    # 2. Calculate Mean and Std (Annualized)
    # Assuming monthly data: Mean * 12, Std * sqrt(12)
    summary = pd.DataFrame(index=G_FACTOR_NAMES)
    summary['Mean (Ann.)'] = df.mean() * 12
    summary['Std. Dev. (Ann.)'] = df.std() * np.sqrt(12)

    # 3. Calculate Correlation Matrix
    corr_matrix = df.corr()

    # 4. Combine into a single Summary Table
    # We will focus the table on CAT's relationship with others as requested
    summary_table = pd.concat([summary, corr_matrix['CAT'].rename('Corr with CAT')], axis=1)

    # 5. Format for LaTeX
    # Rounding to 3 decimal places for academic standard
    latex_table = summary_table.to_latex(
        index=True,
        caption='Summary Statistics and Correlations for the CAT Factor',
        label='tab:cat_summary',
        column_format='lccc',
        float_format="%.3f"
    )

    # Save
    with open(os.path.join(output_folder_path, 'cat_summary_table.tex'), 'w') as f:
        f.write(latex_table)
    
    print("SYS|CAT summary table generated.")
    return summary_table

def generate_grs_output(result_folder_path: str = '', output_folder_path: str = ''):
    """Generate a summary of the GRS test results and create a heatmap of the intercepts (alphas) for the 25 portfolios.

    Args:
        result_folder_path (str): The folder path where the GRS test results are saved. If empty, it will be read from the 'result' folder in the current directory.
        output_folder_path (str): The folder path to save the output files (text summary and heatmap). If empty, it will be saved in the 'output' folder in the current directory.
    """
    result_folder_path = _check_folder_path(result_folder_path, 'result')
    output_folder_path = _check_folder_path(output_folder_path, 'output')

    for file, title in [('grs_all','all'), ('grs_no_cat','no_cat')]:
        with open(os.path.join(result_folder_path, f'{file}.pickle'), 'rb') as f:
            grs_data = pickle.load(f)
            grs_data_reg  = grs_data['grs_reg']
            grs_data_stat = grs_data['grs_stat']
            grs_data_pval = grs_data['p_value']

        consts = []
        consts_for_abs_mean = []
        for pf_nm, result in grs_data_reg.items():
            const = result.params['const']
            match = re.search(r'HML(\d)_CAT(\d)', pf_nm)
            if match:
                hml_q = int(match.group(1))
                cat_q = int(match.group(2))
                consts.append({'HML_Q': hml_q, 'CAT_Q': cat_q, 'alpha': const})
                consts_for_abs_mean.append(abs(const))

        const_df = pd.DataFrame(consts)
        const_matrix = const_df.pivot(index='CAT_Q', columns='HML_Q', values='alpha')
        const_matrix = const_matrix.sort_index(ascending=False)

        # Save results
        with open(os.path.join(output_folder_path, f'{file}_result.txt'), 'w') as f:
            f.write(f'GRS Test Results [{title}]:\n')
            f.write(f'F-statistic: {grs_data_stat:.4f}\n')
            f.write(f'P-value: {grs_data_pval:.4f}\n')
            f.write(f'Mean Absolute Intercept: {np.mean(consts_for_abs_mean):.4f}\n')

        # Plot
        plt.figure(figsize=(10, 8))
        sns.set_theme(style="white")
        ax = sns.heatmap(
            const_matrix, 
            annot=True, 
            fmt=".4f", 
            cmap="RdBu_r", 
            center=0,
            linewidths=.5,
            cbar_kws={'label': 'Intercept (Alpha)'},
            vmin=-0.02, vmax=0.005
        )
        plt.title(f'Intercepts for 25 Portfolios [{title}]', fontsize=14)
        plt.xlabel('HML', fontsize=12)
        plt.ylabel('CAT', fontsize=12)
        
        # Save
        plt.savefig(os.path.join(output_folder_path, f'intercept_heatmap_{title}.png'), bbox_inches='tight', dpi=300)
        plt.show()
        print(f"SYS|Intercept heatmap[{title}] saved to output folder.")
