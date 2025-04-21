# ruff: noqa: INP001

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s : %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)


TABLES = [
    "matrix_elements_d",
    "matrix_elements_q",
    "matrix_elements_o",
    "matrix_elements_q0",
    "matrix_elements_mu",
]


def main() -> None:
    new_path = Path("Rb_v1.1")
    old_path = Path("Rb_v1.0")

    compare_states_table(new_path / "states.parquet", old_path / "states.parquet")

    for table_name in TABLES:
        compare_matrix_elements_table(
            new_path / f"{table_name}.parquet",
            old_path / f"{table_name}.parquet",
        )


def compare_states_table(new_file: Path, old_file: Path, rtol: float = 1e-5, atol: float = 1e-5) -> None:
    new = pd.read_parquet(new_file)
    old = pd.read_parquet(old_file)

    # diff = new.compare(old, result_names=(new_file.parent.name, old_file.parent.name))  # noqa: ERA001
    diff = deep_compare(new, old, atol=atol, rtol=rtol)

    print("Differences in states table:")
    print(diff)


def compare_matrix_elements_table(new_file: Path, old_file: Path, rtol: float = 1e-5, atol: float = 1e-5) -> None:
    new = pd.read_parquet(new_file)
    old = pd.read_parquet(old_file)

    # diff = new.compare(old, result_names=(new_file.parent.name, old_file.parent.name))  # noqa: ERA001
    diff = deep_compare(new, old, atol=atol, rtol=rtol)

    print("Differences in matrix elements table:")
    print(diff)


def deep_compare(df1: pd.DataFrame, df2: pd.DataFrame, atol: float = 0, rtol: float = 0) -> pd.DataFrame:
    """Compare two pandas dataframes at a deep level.

    This will return a dataframe with the differences between the two frames
    explicitly shown.
    See also https://github.com/pandas-dev/pandas/issues/54677


    Args:
        df1: The left dataframe
        df2: The right dataframe
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        A dataframe with the differences between the two frames

    """
    diff_df = pd.DataFrame(index=df1.index, columns=df1.columns)
    for col in df1.columns:
        is_numeric = pd.api.types.is_any_real_numeric_dtype(df1[col]) and pd.api.types.is_any_real_numeric_dtype(
            df2[col]
        )
        if is_numeric:
            condition = np.abs(df1[col] - df2[col]) > atol
            condition &= np.abs(df1[col] - df2[col]) > rtol * np.abs(df2[col])
            diff_df[col] = np.where(condition, df1[col], np.nan)
        else:
            np.where(df1[col] != df2[col], df1[col], np.nan)

    # Remove all rows and columns that are all NaN
    diff_df = diff_df.dropna(how="all")
    diff_df = diff_df.dropna(axis=1, how="all")

    diff_colums = diff_df.columns
    right_df = df2[diff_colums]

    return diff_df.merge(right_df, left_index=True, right_index=True, suffixes=("_df1", "_df2"))


if __name__ == "__main__":
    main()
