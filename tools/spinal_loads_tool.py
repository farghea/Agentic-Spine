import pandas as pd 

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
    Axis                                 Measurement     x_force     y_force   z_force
    0             L1_L2_IVDjnt_on_lumbar1_in_lumbar1   98.848172  501.312445 -0.277177
    1             L2_L3_IVDjnt_on_lumbar2_in_lumbar2   37.883546  524.940500 -0.188179
    2             L3_L4_IVDjnt_on_lumbar3_in_lumbar3   59.220554  517.651793 -0.072809
    3             L4_L5_IVDjnt_on_lumbar4_in_lumbar4  -21.318754  585.294823  0.092017
    """
    # Convert the input dictionary to a DataFrame
    df = pd.DataFrame.from_dict(spinal_loads_data, orient='index', columns=['Force'])
    
    # make sure all forces are numeric
    df['Force'] = pd.to_numeric(df['Force'])

    # 2. Extract the Joint Name and the Axis from the index
    # We split by the LAST underscore to separate '..._fx' into '...' and 'fx'
    df[['Measurement', 'Axis']] = df.index.to_series().str.rsplit('_', n=1, expand=True)

    # 3. Pivot the table so Axis (fx, fy, fz) become columns
    df_tidy = df.pivot(index='Measurement', columns='Axis', values='Force')

    # 4. Clean up column names to your specific request
    df_tidy = df_tidy.rename(columns={
        'fx': 'x_force',
        'fy': 'y_force',
        'fz': 'z_force'
    }).reset_index()

    # Optional: Reorder columns for better readability
    df_tidy = df_tidy[['Measurement', 'x_force', 'y_force', 'z_force']]


    
    return df_tidy


if __name__ == "__main__":
    # Example usage
    spinal_loads_data = {'L5_S1_IVDjnt_on_lumbar5_in_lumbar5_fx': '-126.51181212',
 'L5_S1_IVDjnt_on_lumbar5_in_lumbar5_fy': '759.51954459',
 'L5_S1_IVDjnt_on_lumbar5_in_lumbar5_fz': '0.05621091',
 'L4_L5_IVDjnt_on_lumbar4_in_lumbar4_fx': '-21.31875354',
 'L4_L5_IVDjnt_on_lumbar4_in_lumbar4_fy': '585.29482345',
 'L4_L5_IVDjnt_on_lumbar4_in_lumbar4_fz': '0.09201725',
 'L3_L4_IVDjnt_on_lumbar3_in_lumbar3_fx': '59.22055410',
 'L3_L4_IVDjnt_on_lumbar3_in_lumbar3_fy': '517.65179267',
 'L3_L4_IVDjnt_on_lumbar3_in_lumbar3_fz': '-0.07280889',
 'L2_L3_IVDjnt_on_lumbar2_in_lumbar2_fx': '37.88354610',
 'L2_L3_IVDjnt_on_lumbar2_in_lumbar2_fy': '524.94049969',
 'L2_L3_IVDjnt_on_lumbar2_in_lumbar2_fz': '-0.18817885',
 'L1_L2_IVDjnt_on_lumbar1_in_lumbar1_fx': '98.84817247',
 'L1_L2_IVDjnt_on_lumbar1_in_lumbar1_fy': '501.31244454',
 'L1_L2_IVDjnt_on_lumbar1_in_lumbar1_fz': '-0.27717733',
 'T12_L1_IVDjnt_on_thoracic12_in_thoracic12_fx': '116.49386404',
 'T12_L1_IVDjnt_on_thoracic12_in_thoracic12_fy': '527.41188373',
 'T12_L1_IVDjnt_on_thoracic12_in_thoracic12_fz': '-0.26993487',
 'T11_T12_IVDjnt_on_thoracic11_in_thoracic11_fx': '47.42859327',
 'T11_T12_IVDjnt_on_thoracic11_in_thoracic11_fy': '574.33139996',
 'T11_T12_IVDjnt_on_thoracic11_in_thoracic11_fz': '-0.38699718',
 'T10_T11_IVDjnt_on_thoracic10_in_thoracic10_fx': '12.78113448',
 'T10_T11_IVDjnt_on_thoracic10_in_thoracic10_fy': '554.78850405',
 'T10_T11_IVDjnt_on_thoracic10_in_thoracic10_fz': '-0.42090086',
 'T9_T10_IVDjnt_on_thoracic9_in_thoracic9_fx': '-22.37129414',
 'T9_T10_IVDjnt_on_thoracic9_in_thoracic9_fy': '444.72463540',
 'T9_T10_IVDjnt_on_thoracic9_in_thoracic9_fz': '-0.45577510',
 'T8_T9_IVDjnt_on_thoracic8_in_thoracic8_fx': '13.75812491',
 'T8_T9_IVDjnt_on_thoracic8_in_thoracic8_fy': '424.65935962',
 'T8_T9_IVDjnt_on_thoracic8_in_thoracic8_fz': '0.23484129',
 'T7_T8_IVDjnt_on_thoracic7_in_thoracic7_fx': '-12.10696102',
 'T7_T8_IVDjnt_on_thoracic7_in_thoracic7_fy': '413.04294505',
 'T7_T8_IVDjnt_on_thoracic7_in_thoracic7_fz': '-0.08231080',
 'T6_T7_IVDjnt_on_thoracic6_in_thoracic6_fx': '-51.91851344',
 'T6_T7_IVDjnt_on_thoracic6_in_thoracic6_fy': '452.48468960',
 'T6_T7_IVDjnt_on_thoracic6_in_thoracic6_fz': '1.90807211',
 'T5_T6_IVDjnt_on_thoracic5_in_thoracic5_fx': '-49.70190000',
 'T5_T6_IVDjnt_on_thoracic5_in_thoracic5_fy': '436.59560139',
 'T5_T6_IVDjnt_on_thoracic5_in_thoracic5_fz': '0.93090895',
 'T4_T5_IVDjnt_on_thoracic4_in_thoracic4_fx': '-34.05561386',
 'T4_T5_IVDjnt_on_thoracic4_in_thoracic4_fy': '335.51283077',
 'T4_T5_IVDjnt_on_thoracic4_in_thoracic4_fz': '-1.33003061',
 'T3_T4_IVDjnt_on_thoracic3_in_thoracic3_fx': '-60.75121496',
 'T3_T4_IVDjnt_on_thoracic3_in_thoracic3_fy': '304.08243935',
 'T3_T4_IVDjnt_on_thoracic3_in_thoracic3_fz': '-0.86200681',
 'T2_T3_IVDjnt_on_thoracic2_in_thoracic2_fx': '-51.09796994',
 'T2_T3_IVDjnt_on_thoracic2_in_thoracic2_fy': '257.08458819',
 'T2_T3_IVDjnt_on_thoracic2_in_thoracic2_fz': '-0.95536942',
 'T1_T2_IVDjnt_on_thoracic1_in_thoracic1_fx': '-91.13546337',
 'T1_T2_IVDjnt_on_thoracic1_in_thoracic1_fy': '247.88951804',
 'T1_T2_IVDjnt_on_thoracic1_in_thoracic1_fz': '0.63458795'}
    
    tidy_df = tidy_spinal_loads_tool(spinal_loads_data)
    print(tidy_df)