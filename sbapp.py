# Updated sbapp.py

import numpy as np

# Original function definitions
...

# Update instances of use_container_width
# Replace `use_container_width=True` to `width='stretch'`
# Replace `use_container_width=False` to `width='content'`

# Example of what it might change to
# st.container(use_container_width=True) -> st.container(width='stretch')

# In the format_and_display function, modify how totals are set

def format_and_display(totals, col):
    if col in numeric_columns:
        totals[col] = np.nan  # Changed from '' to np.nan
    else:
        totals[col] = ''

# Continue with the rest of the code...