import pandas as pd


def to_dummy(df: pd.DataFrame, col_list: list[str]) -> pd.DataFrame:
    # drop_first=Trueで多重共線性を防ぐ
    return pd.get_dummies(
        df, columns=col_list, drop_first=True, dtype=int, dummy_na=True
    )
