import numpy as np

# Assuming the rest of the content from the original app (31).py file gets inserted here


def format_and_display(data):
    totals = {}  # Assume totals is defined properly in your application
    for col in data.columns:
        if data[col].dtype in ['float64', 'int64']:
            totals[col] = np.nan  # Change here for numeric columns
        else:
            totals[col] = ''
    return totals

# Example of how use_container_width was changed
# previous code was using 'use_container_width=True'
# changed to 'width="stretch"'

