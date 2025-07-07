from pandas import DataFrame


def find_duplicate_columns(df: DataFrame):
    """Returns dict of column names and their index"""
    duplicate_column_map = {
        col: i for i, col in enumerate(df.columns) if col in df.columns[:i]
    }
    return duplicate_column_map


def check_column_differences(df1: DataFrame, df2: DataFrame):
    """
    Check column differences between two dataframes
    returns the columns and index of that column that are missing in df2 compared to df1
    """

    mis_col = set(df1.columns) - set(df2.columns)

    missing_columns_dict = {col: df1.columns.get_loc(col) for col in mis_col}

    return missing_columns_dict


def align_columns(df1: DataFrame, df2: DataFrame):
    """
    Align the order of  columns of two dataframes if they have identical columns
    Aligns the columns of df1 to the order of df2
    """
    alligned_df = df1[df2.columns]

    return alligned_df
