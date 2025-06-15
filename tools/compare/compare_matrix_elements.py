from pathlib import Path

import pandas as pd
from generate_database_sqdt.main import MATRIX_ELEMENTS_OF_INTEREST


def main() -> None:
    # CHANGE THESE PATHS, TO THE FOLDERS YOU WANT TO COMPARE
    name = "Rb"
    new_path = Path(f"{name}_v1.2")
    old_path = Path(f"{name}_v1.1")

    for table_name in MATRIX_ELEMENTS_OF_INTEREST:
        compare_matrix_elements_table(table_name, new_path, old_path, verbose=False)


def compare_matrix_elements_table(
    table_name: str, new_path: Path, old_path: Path, rtol: float = 1e-4, atol: float = 1e-10, *, verbose: bool = False
) -> None:
    """Compare the matrix elements table of two versions of the database.

    Given the path to states and matrix elements parquet tables,
    this function will
    1) create a unique index mapping from a state (defined by n, exp_l, and exp_j) to a unique identifier
    2) replace the old id_initial and id_final column in the matrix elements table with this unique index
    3) compare the column "val" of the two matrix elements tables.

    """
    print(f"Comparing matrix elements table for {table_name}:\n  New: {new_path}\n  Old: {old_path}\n")

    states_dict = {
        "new": pd.read_parquet(new_path / "states.parquet"),
        "old": pd.read_parquet(old_path / "states.parquet"),
    }
    table_dict = {
        "new": pd.read_parquet(new_path / f"{table_name}.parquet"),
        "old": pd.read_parquet(old_path / f"{table_name}.parquet"),
    }

    old_id_columns = ["id_initial", "id_final"]
    new_id_columns = ["id_initial_unique", "id_final_unique"]

    for key, state in states_dict.items():
        # Create a unique index for each state based on quantum numbers
        state["unique_id"] = state.apply(lambda row: f"{row['n']}_{row['exp_l']}_{row['exp_j']}", axis=1)
        id_to_unique = dict(zip(state["id"], state["unique_id"], strict=False))

        # Map the ids in the matrix elements table to the unique identifiers
        table = table_dict[key]
        for old_id_col, new_id_col in zip(old_id_columns, new_id_columns, strict=True):
            table[new_id_col] = table[old_id_col].map(id_to_unique)

        # Index the matrix elements by the new unique id
        table_dict[key] = table.set_index(new_id_columns)

    # Compare val values within tolerance
    new, old = table_dict["new"], table_dict["old"]
    val_diff = (new["val"] - old["val"]).abs()
    tolerance = atol + rtol * old["val"].abs()
    val_mask = val_diff.gt(tolerance)

    print(f"Found {val_mask.sum()} val differences outside tolerance:")
    if verbose and val_mask.any():
        diff_uids = val_mask.loc[val_mask].index
        for uid in diff_uids:
            new_val = new.loc[uid, "val"]
            old_val = old.loc[uid, "val"]
            diff_val = val_diff[uid]
            print(f"  State initial: {uid[0]}; State final: {uid[1]}")
            print(f"    New val: {new_val:.12f}")
            print(f"    Old val: {old_val:.12f}")
            print(f"    Absolute difference: {diff_val:.2e}")
            print(f"    Relative difference: {diff_val / abs(old_val):.2e}")

    max_rdiff = (val_diff / old["val"].abs()).max()
    print(f"Maximum absolute val difference: {val_diff.max():.2e}")
    print(f"Maximum relative val difference: {max_rdiff:.2e}")


if __name__ == "__main__":
    main()
