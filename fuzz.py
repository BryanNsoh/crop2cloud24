import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
import skfuzzy as fuzz
import sqlite3
from datetime import datetime, timedelta

def compute_scaled_sums(df, num_days):
    """
    Computes the scaled sums of each column in the given DataFrame, taking into account the number of missing values.
    """
    actual_sum_available = df.sum()
    num_available_days = df.count()
    
    if num_days == num_available_days.min():
        return actual_sum_available
    else:
        mean_values = df.mean()
        num_missing_days = num_days - num_available_days
        estimated_sum_missing = mean_values * num_missing_days
        final_sums = actual_sum_available + estimated_sum_missing
        return final_sums

def setup_fuzzy_system():
    # Define the range of values for each variable
    x_hsi = np.arange(0, 100, 1)  # Heat stress index
    x_et = np.arange(0, 20, 0.1)  # evaporative demand
    x_swsi = np.arange(0, 4, 0.01)  # Soil water stress index
    x_cwsi = np.arange(0, 3, 0.01)  # Crop water stress index
    x_irrigation = np.arange(0, 1, 0.01)  # Irrigation amount in inches

    # Define fuzzy sets for each variable
    hsi_low = fuzz.trapmf(x_hsi, [0, 0, 30, 50])
    hsi_med = fuzz.trapmf(x_hsi, [30, 50, 70, 90])
    hsi_high = fuzz.trapmf(x_hsi, [70, 90, 100, 100])

    et_low = fuzz.trapmf(x_et, [0, 0, 5, 10])
    et_med = fuzz.trapmf(x_et, [5, 10, 15, 20])
    et_high = fuzz.trapmf(x_et, [15, 20, 20, 20])

    swsi_wet = fuzz.trapmf(x_swsi, [0, 0, 0.8, 1.6])
    swsi_normal = fuzz.trapmf(x_swsi, [0.8, 1.6, 2.4, 3.2])
    swsi_dry = fuzz.trapmf(x_swsi, [2.4, 3.2, 4, 4])

    cwsi_low = fuzz.trapmf(x_cwsi, [0, 0, 0.5, 1])
    cwsi_med = fuzz.trapmf(x_cwsi, [0.5, 1, 1.5, 2])
    cwsi_high = fuzz.trapmf(x_cwsi, [1.5, 2, 3, 3])

    irrigation_zero = fuzz.trimf(x_irrigation, [0, 0, 0.05])
    irrigation_low = fuzz.trapmf(x_irrigation, [0, 0.05, 0.2, 0.3])
    irrigation_med = fuzz.trapmf(x_irrigation, [0.2, 0.3, 0.6, 0.7])
    irrigation_high = fuzz.trapmf(x_irrigation, [0.6, 0.7, 1, 1])

    # Create control variables
    hsi = ctrl.Antecedent(x_hsi, "hsi")
    et = ctrl.Antecedent(x_et, "et")
    swsi = ctrl.Antecedent(x_swsi, "swsi")
    cwsi = ctrl.Antecedent(x_cwsi, "cwsi")
    irrigation = ctrl.Consequent(x_irrigation, "irrigation")

    # Assign membership functions to variables
    hsi['low'], hsi['med'], hsi['high'] = hsi_low, hsi_med, hsi_high
    et['low'], et['med'], et['high'] = et_low, et_med, et_high
    swsi['dry'], swsi['normal'], swsi['wet'] = swsi_dry, swsi_normal, swsi_wet
    cwsi['low'], cwsi['med'], cwsi['high'] = cwsi_low, cwsi_med, cwsi_high
    irrigation['zero'], irrigation['low'], irrigation['med'], irrigation['high'] = irrigation_zero, irrigation_low, irrigation_med, irrigation_high

    # Define a comprehensive set of rules
    rules = [
        # HSI low
        ctrl.Rule(hsi['low'] & et['low'] & swsi['wet'] & cwsi['low'], irrigation['zero']),
        ctrl.Rule(hsi['low'] & et['low'] & swsi['wet'] & cwsi['med'], irrigation['low']),
        ctrl.Rule(hsi['low'] & et['low'] & swsi['wet'] & cwsi['high'], irrigation['low']),
        ctrl.Rule(hsi['low'] & et['low'] & swsi['normal'] & cwsi['low'], irrigation['zero']),
        ctrl.Rule(hsi['low'] & et['low'] & swsi['normal'] & cwsi['med'], irrigation['low']),
        ctrl.Rule(hsi['low'] & et['low'] & swsi['normal'] & cwsi['high'], irrigation['med']),
        ctrl.Rule(hsi['low'] & et['low'] & swsi['dry'] & cwsi['low'], irrigation['low']),
        ctrl.Rule(hsi['low'] & et['low'] & swsi['dry'] & cwsi['med'], irrigation['med']),
        ctrl.Rule(hsi['low'] & et['low'] & swsi['dry'] & cwsi['high'], irrigation['high']),
        
        ctrl.Rule(hsi['low'] & et['med'] & swsi['wet'], irrigation['low']),
        ctrl.Rule(hsi['low'] & et['med'] & swsi['normal'], irrigation['med']),
        ctrl.Rule(hsi['low'] & et['med'] & swsi['dry'], irrigation['high']),
        
        ctrl.Rule(hsi['low'] & et['high'] & swsi['wet'], irrigation['med']),
        ctrl.Rule(hsi['low'] & et['high'] & swsi['normal'], irrigation['high']),
        ctrl.Rule(hsi['low'] & et['high'] & swsi['dry'], irrigation['high']),
        
        # HSI medium
        ctrl.Rule(hsi['med'] & et['low'] & swsi['wet'], irrigation['low']),
        ctrl.Rule(hsi['med'] & et['low'] & swsi['normal'], irrigation['med']),
        ctrl.Rule(hsi['med'] & et['low'] & swsi['dry'], irrigation['high']),
        
        ctrl.Rule(hsi['med'] & et['med'] & swsi['wet'] & cwsi['low'], irrigation['low']),
        ctrl.Rule(hsi['med'] & et['med'] & swsi['wet'] & cwsi['med'], irrigation['med']),
        ctrl.Rule(hsi['med'] & et['med'] & swsi['wet'] & cwsi['high'], irrigation['med']),
        ctrl.Rule(hsi['med'] & et['med'] & swsi['normal'] & cwsi['low'], irrigation['med']),
        ctrl.Rule(hsi['med'] & et['med'] & swsi['normal'] & cwsi['med'], irrigation['med']),
        ctrl.Rule(hsi['med'] & et['med'] & swsi['normal'] & cwsi['high'], irrigation['high']),
        ctrl.Rule(hsi['med'] & et['med'] & swsi['dry'] & cwsi['low'], irrigation['med']),
        ctrl.Rule(hsi['med'] & et['med'] & swsi['dry'] & cwsi['med'], irrigation['high']),
        ctrl.Rule(hsi['med'] & et['med'] & swsi['dry'] & cwsi['high'], irrigation['high']),
        
        ctrl.Rule(hsi['med'] & et['high'] & swsi['wet'], irrigation['med']),
        ctrl.Rule(hsi['med'] & et['high'] & swsi['normal'], irrigation['high']),
        ctrl.Rule(hsi['med'] & et['high'] & swsi['dry'], irrigation['high']),
        
        # HSI high
        ctrl.Rule(hsi['high'] & et['low'], irrigation['med']),
        ctrl.Rule(hsi['high'] & et['med'], irrigation['high']),
        
        ctrl.Rule(hsi['high'] & et['high'] & swsi['wet'] & cwsi['low'], irrigation['med']),
        ctrl.Rule(hsi['high'] & et['high'] & swsi['wet'] & cwsi['med'], irrigation['med']),
        ctrl.Rule(hsi['high'] & et['high'] & swsi['wet'] & cwsi['high'], irrigation['high']),
        ctrl.Rule(hsi['high'] & et['high'] & swsi['normal'] & cwsi['low'], irrigation['med']),
        ctrl.Rule(hsi['high'] & et['high'] & swsi['normal'] & cwsi['med'], irrigation['high']),
        ctrl.Rule(hsi['high'] & et['high'] & swsi['normal'] & cwsi['high'], irrigation['high']),
        ctrl.Rule(hsi['high'] & et['high'] & swsi['dry'] & cwsi['low'], irrigation['high']),
        ctrl.Rule(hsi['high'] & et['high'] & swsi['dry'] & cwsi['med'], irrigation['high']),
        ctrl.Rule(hsi['high'] & et['high'] & swsi['dry'] & cwsi['high'], irrigation['high']),
    ]

    # Create and return the control system
    irrigation_system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(irrigation_system)

def get_plot_data(conn, plot_number, end_date):
    start_date = end_date - timedelta(days=3)
    query = f"""
    SELECT TIMESTAMP, HeatIndex_2m_Avg, et, "cwsi-eb2 " as cwsi, swsi
    FROM plot_{plot_number}
    WHERE TIMESTAMP BETWEEN ? AND ?
    AND is_actual = 1
    """
    df = pd.read_sql_query(query, conn, params=(start_date, end_date))
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    return df.set_index('TIMESTAMP')

def compute_irrigation_recommendation(irrigation_sim, plot_data):
    num_days = (plot_data.index.max() - plot_data.index.min()).days + 1
    final_sums = compute_scaled_sums(plot_data, num_days)
    
    hsi_sum = final_sums['HeatIndex_2m_Avg']
    et_sum = final_sums['et']
    swsi_sum = final_sums['swsi']
    cwsi_sum = final_sums['cwsi']

    irrigation_sim.input['hsi'] = hsi_sum
    irrigation_sim.input['et'] = et_sum
    irrigation_sim.input['swsi'] = swsi_sum
    irrigation_sim.input['cwsi'] = cwsi_sum

    try:
        irrigation_sim.compute()
        return irrigation_sim.output['irrigation'], final_sums
    except ValueError as e:
        print(f"Error in fuzzy computation: {e}")
        print(f"Input values: HSI={hsi_sum}, ET={et_sum}, SWSI={swsi_sum}, CWSI={cwsi_sum}")
        return None, final_sums

def main():
    print(f"Script started at: {datetime.now()}")
    
    # Connect to the database
    conn = sqlite3.connect('mpc_data.db')

    # Set up the fuzzy control system
    irrigation_sim = setup_fuzzy_system()

    # Get yesterday's date
    yesterday = datetime.now().date() - timedelta(days=1)
    print(f"Computing for date range: {yesterday - timedelta(days=3)} to {yesterday}")

    # List of plot numbers
    plot_numbers = [5006, 5010, 5023]

    for plot_number in plot_numbers:
        print(f"\nProcessing Plot {plot_number}")
        # Get data for the plot
        plot_data = get_plot_data(conn, plot_number, yesterday)

        if plot_data.empty:
            print(f"No data available for plot {plot_number}")
            continue

        print(f"Data range: {plot_data.index.min()} to {plot_data.index.max()}")
        
        # Resample to daily frequency
        plot_data_daily = plot_data.resample('D').mean()
        print("Daily resampled data:")
        print(plot_data_daily)

        # Compute irrigation recommendation
        irrigation_amount, final_sums = compute_irrigation_recommendation(irrigation_sim, plot_data_daily)

        if irrigation_amount is not None:
            print(f"Plot {plot_number} Irrigation Recommendation:")
            print(f"Recommended irrigation amount: {irrigation_amount:.2f} inches")
        else:
            print(f"Unable to compute irrigation recommendation for plot {plot_number}")

        print("Input values (after scaling):")
        print(f"HSI sum: {final_sums['HeatIndex_2m_Avg']:.2f}")
        print(f"ET sum: {final_sums['et']:.2f}")
        print(f"SWSI sum: {final_sums['swsi']:.2f}")
        print(f"CWSI sum: {final_sums['cwsi']:.2f}")

    conn.close()
    print(f"\nScript completed at: {datetime.now()}")

if __name__ == "__main__":
    main()