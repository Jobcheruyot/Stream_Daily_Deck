import pandas as pd
import numpy as np


def format_and_display(dataframe):
    """Format the given dataframe for display."""
    # Replace empty strings with np.nan
    dataframe.replace('', np.nan, inplace=True)
    # Additional formatting code can go here
    return dataframe


def main():
    # Example use of the format_and_display function
    data = {
        'Column1': [1, 2, '', 4],
        'Column2': ['', 2.5, 3.5, 4.5]
    }
    df = pd.DataFrame(data)
    formatted_df = format_and_display(df)
    print(formatted_df)


if __name__ == '__main__':
    main()