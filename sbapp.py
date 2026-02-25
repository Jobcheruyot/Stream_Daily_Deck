import numpy as np

def format_and_display(data):
    # Replace use_container_width=True with width='stretch'
    # Replace use_container_width=False with width='content'
    data = data.replace(use_container_width=True, width='stretch')
    data = data.replace(use_container_width=False, width='content')

    # Change empty string assignments to np.nan for numeric columns
    for column in data.select_dtypes(include='number').columns:
        data[column].replace('', np.nan, inplace=True)

    return data
