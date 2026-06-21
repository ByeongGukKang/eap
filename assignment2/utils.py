from typing import Literal

import polars as pl
import polars_readstat as pl_rs

ghz_identifiers: list[str] = [
    # Identifier, date, returns, market value, SIC code
    "permno",
    "gvkey",
    "date",
    "ret",
    "mve",
    "sic2",  # For SIC dummy, only has 73 unique values, EAP via ML has 74 dummy variables
]
characteristics_dict: dict[str, Literal["M", "Q", "A"]] = {
    # 1. Gu, Kelly, Xiu (2020) 94 Stock-level Characteristics (IA. Table A.6)
    "absacc": "A",
    "acc": "A",
    "aeavol": "Q",
    "age": "A",
    "agr": "A",
    "baspread": "M",
    "beta": "M",
    "betasq": "M",
    "bm": "A",
    "bm_ia": "A",
    "cash": "Q",
    "cashdebt": "A",
    "cashpr": "A",
    "cfp": "A",
    "cfp_ia": "A",
    "chatoia": "A",
    "chcsho": "A",
    "chempia": "A",
    "chinv": "A",
    "chmom": "M",
    "chpmia": "A",
    "chtx": "Q",
    "cinvest": "Q",
    "convind": "A",
    "currat": "A",
    "depr": "A",
    "divi": "A",
    "divo": "A",
    "dolvol": "M",
    "dy": "A",
    "ear": "Q",
    "egr": "A",
    "ep": "A",
    "gma": "A",
    "grcapx": "A",
    "grltnoa": "A",
    "herf": "A",
    "hire": "A",
    "idiovol": "M",
    "ill": "M",
    "indmom": "M",
    "invest": "A",
    "lev": "A",
    "lgr": "A",
    "maxret": "M",
    "mom12m": "M",
    "mom1m": "M",
    "mom36m": "M",
    "mom6m": "M",
    "ms": "Q",
    "mve_m": "M",  # EAP via ML paper IA has typo!!!!!, mvel1 -> mve_m
    "mve_ia": "A",
    "nincr": "Q",
    "operprof": "A",
    "orgcap": "A",
    "pchcapx_ia": "A",
    "pchcurrat": "A",
    "pchdepr": "A",
    "pchgm_pchsale": "A",
    "pchquick": "A",
    "pchsale_pchinvt": "A",
    "pchsale_pchrect": "A",
    "pchsale_pchxsga": "A",
    "pchsaleinv": "A",
    "pctacc": "A",
    "pricedelay": "M",
    "ps": "A",
    "quick": "A",
    "rd": "A",
    "rd_mve": "A",
    "rd_sale": "A",
    "realestate": "A",
    "retvol": "M",
    "roaq": "Q",
    "roavol": "Q",
    "roeq": "Q",
    "roic": "A",
    "rsup": "Q",
    "salecash": "A",
    "saleinv": "A",
    "salerec": "A",
    "secured": "A",
    "securedind": "A",
    "sgr": "A",
    "sin": "A",
    "sp": "A",
    "std_dolvol": "M",
    "std_turn": "M",
    "stdacc": "Q",
    "stdcf": "Q",
    "tang": "A",
    "tb": "A",
    "turn": "M",
    "zerotrade": "M",
    # Remove analyst forecasts
    # "disp",
    # "chfeps",
    # "fgr5yr",
    # "chrec",
    # "nanalyst",
    # "sfe",
    # "sue",
    # "ltg",
}


# SAS file to Polars DataFrame
# SAS file is manually downloaded using
# https://drive.google.com/file/d/0BwwEXkCgXEdRQWZreUpKOHBXOUU/view?resourcekey=0-1xjZ8fAc0sTybVC6RADDCA
def scan_ghz(path: str = "./data/ghz.sas7bdat") -> pl.LazyFrame:
    df = (
        pl_rs.scan_readstat(path)
        .rename(lambda col: col.lower())
        .select(ghz_identifiers + list(characteristics_dict.keys()))
        .cast({"permno": pl.Int32})
        .with_columns(
            pl.col("date").dt.month_end().alias("date"),
        )
        .sort(["permno", "date"])
    )
    chars = [
        col
        for col in df.collect_schema().names()
        if col not in ["permno", "gvkey", "date", "ret", "mve", "sic2"]
    ]

    lag_mapping = {"M": 1, "Q": 4, "A": 6}
    lag_exprs = []
    for char in chars:
        lag = lag_mapping[characteristics_dict[char]]
        lag_exprs.append(pl.col(char).shift(lag).over("permno").alias(char))
    df_lagged = df.with_columns(lag_exprs)

    norm_exprs = []
    for char in chars:
        nan_handled = (
            pl.col(char)
            .fill_nan(None)
            .fill_null(pl.col(char).median().over("date"))
            .fill_null(0.0)
        )

        N = nan_handled.count().over("date")
        rank = (
            pl.when(N > 1)
            .then((nan_handled.rank().over("date") - 1) / (N - 1) * 2 - 1)
            .otherwise(0.0)
        )
        norm_exprs.append(rank.alias(char))

    return df_lagged.with_columns(norm_exprs)


def scan_macro(
    path: str = "./data/PredictorData2025.xlsx - Monthly.csv",
) -> pl.LazyFrame:
    numberic_cols = [
        "D12",
        "Index",
        "E12",
        "b/m",
        "lty",
        "tbl",
        "BAA",
        "AAA",
        "ntis",
        "svar",
    ]
    return (
        pl.scan_csv(path, infer_schema=False)
        .select(
            pl.col("yyyymm")
            .cast(pl.String)
            .str.to_date("%Y%m")
            .dt.month_end()
            .alias("join_date"),
            *[
                pl.col(c)
                .cast(pl.String)
                .str.replace_all(",", "")
                .cast(pl.Float64)
                .alias(c)
                for c in numberic_cols
            ],
        )
        .with_columns(
            (pl.col("D12").log() - pl.col("Index").log()).alias(
                "dp"
            ),  # Dividend-Price ratio
            (pl.col("E12").log() - pl.col("Index").log()).alias(
                "ep"
            ),  # Earnings-Price ratio
            pl.col("b/m").alias("bm"),  # Book-to-Market ratio
            (pl.col("lty") - pl.col("tbl")).alias("tms"),  # Term Spread
            (pl.col("BAA") - pl.col("AAA")).alias("dfy"),  # Default Yield Spread
            pl.col("tbl"),  # Treasury-bill rate
            pl.col("ntis"),  # Net Equity Expansion
            pl.col("svar"),  # Stock Variance
        )
        .select(["join_date", "dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar"])
        .rename({"join_date": "date"})
    )
