from impl import (
    get_french_data, get_crsp_msf_data,
    create_cat_factor,
    estimate_factor_loadings,
    run_regressions,
    run_grs_test,
)
from output import (
    generate_reg_latex_table,
    generate_cat_summary_table,
    generate_grs_output,
)


if __name__ == '__main__':
    ### Get Data
    # Download data
    get_french_data()
    get_crsp_msf_data()
    # Create the CAT factor
    create_cat_factor() 
    
    ### Generate Results
    estimate_factor_loadings()
    run_regressions()
    run_grs_test()

    ### Generate Outputs
    generate_reg_latex_table()
    generate_cat_summary_table()
    generate_grs_output()