from typing import Dict
import pandas as pd
import re

def tidy_spinal_loads_tool(spinal_loads_data: dict) -> pd.DataFrame:
    """
    Transforms raw spinal loads data into a tidy DataFrame format.

    Parameters:
    spinal_loads_data (dict): A dictionary containing spinal loads data,
                              where keys are variable names and values are lists of measurements.

    Returns:
    pd.DataFrame: A tidy DataFrame with columns 'time', 'variable', and 'value'.

    Example Input:
    {'L5_S1_IVDjnt_on_lumbar5_in_lumbar5_fx': '-126.51181212',
     'L5_S1_IVDjnt_on_lumbar5_in_lumbar5_fy': '759.51954459',
     'L5_S1_IVDjnt_on_lumbar5_in_lumbar5_fz': '0.05621091',}

    Example Output:
        joint_level	    spinal_level	fx	        fy	        fz
    0	T1_T2	        thoracic1	    -91.135463	247.889518	0.634588
    1	T2_T3	        thoracic2	    -51.097970	257.084588	-0.955369
    2	T3_T4	        thoracic3	    -60.751215	304.082439	-0.862007
    3	T4_T5	        thoracic4	    -34.055614	335.512831	-1.330031
    """
    # Convert the input dictionary to a DataFrame
    # 1. Extraction logic
    extracted_data = {}

    for key, value in spinal_loads_data.items():
        # Regex captures: 1. Joint Level, 2. Level name, 3. Component (fx/fy/fz)
        match = re.search(r'^(.*)_IVDjnt_on_([^_]*)_in_([^_]*)_(f[xyz])', key)
        if match:
            joint = match.group(1)
            level = match.group(3)
            axis = match.group(4)
            
            if joint not in extracted_data:
                extracted_data[joint] = {'spinal_level': level}
            extracted_data[joint][axis] = float(value)

    # 2. Convert to DataFrame
    df = pd.DataFrame.from_dict(extracted_data, orient='index').reset_index()
    df.rename(columns={'index': 'joint_level'}, inplace=True)

    # 3. Sort anatomically
    def joint_sorter(joint):
        top_vertebra = joint.split('_')[0]
        prefix = 0 if 'T' in top_vertebra else 1
        number = int(re.search(r'\d+', top_vertebra).group())
        return prefix, number

    df = df.sort_values(by='joint_level', key=lambda x: x.map(joint_sorter)).reset_index(drop=True)

    # 4. Final selection
    df = df[['joint_level', 'spinal_level', 'fx', 'fy', 'fz']]



    return df