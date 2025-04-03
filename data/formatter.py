# data/formatter.py

def to_wide_format(df: pd.DataFrame, value_cols=["Adj Close", "rsi", "sma"]):
    return df.pivot_table(index="Date", columns="ticker", values=value_cols)
