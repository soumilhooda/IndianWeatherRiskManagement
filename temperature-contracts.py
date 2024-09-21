# ==============================================================
# Weather Risk Model for Temperature Derivatives Pricing
# ==============================================================

import os
import pandas as pd
import numpy as np
import logging
from scipy.optimize import curve_fit
from statsmodels.tsa.ar_model import AutoReg
import datetime as dt
from scipy import interpolate

# --------------------------------------------------------------
# Set up logging
# --------------------------------------------------------------

logging.basicConfig(filename='weather_risk_model.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------------------------------------------
# Global Parameters and Constants
# --------------------------------------------------------------

WINTER_MONTHS = [1, 2, 10, 11, 12]
MONSOON_MONTHS = [3, 4, 5, 6, 7, 8, 9]
r = 0.05       # Risk-free rate
alpha = 25     # Scaling factor for option payoffs
no_sims = 1000 # Number of simulations for Monte Carlo

# Constants for Heatwave and Coldwave Definitions
HEATWAVE_PROBABILITY = 0.005  # Probability of a heatwave starting on any day
COLDWAVE_PROBABILITY = 0.005  # Probability of a coldwave starting on any day
SHOCK_DURATION = 5            # Number of days a shock lasts
SHOCK_INTENSITY_HEAT = 5      # Temperature increase per day during a heatwave
SHOCK_INTENSITY_COLD = -5     # Temperature decrease per day during a coldwave

# --------------------------------------------------------------
# Function Definitions
# --------------------------------------------------------------

def load_data():
    """
    Load and preprocess all necessary datasets.
    Aggregates data by state, storing minimum, maximum, and average values for each date.
    Returns:
        data (dict): Dictionary containing aggregated weather data indexed by state.
    """
    logging.info("Loading datasets...")

    # Load weather data
    folder_path = "/Users/soumilhooda/Desktop/IndianWeatherRiskManagement/Data/Weather" 
    data_raw = {}
    for file_name in os.listdir(folder_path):
        if file_name.startswith(("rainnn", "maxtmp", "mintmp")):
            year = file_name[-10:-6]
            data_type = file_name[:6]
            df = pd.read_csv(os.path.join(folder_path, file_name), delim_whitespace=True, header=None)
            num_days = df.shape[1] - 2
            date_range = pd.date_range(start=f'{year}-01-01', periods=num_days)
            date_columns = date_range.strftime('%Y-%m-%d').tolist()
            df.columns = ['Latitude', 'Longitude'] + date_columns
            data_raw[(data_type, year)] = df

    # Calculate average temperature
    for year in range(1951, 2024):
        year_str = str(year)
        maxtmp = data_raw.get(('maxtmp', year_str))
        mintmp = data_raw.get(('mintmp', year_str))
        if maxtmp is not None and mintmp is not None:
            avg_tmp = (maxtmp.iloc[:, 2:] + mintmp.iloc[:, 2:]) / 2
            data_raw[('avgtmp', year_str)] = pd.concat([maxtmp.iloc[:, :2], avg_tmp], axis=1)

    # Create dictionary with (latitude, longitude) pairs and rainfall/temperature data
    result = {}
    for year in range(1951, 2024):
        year_str = str(year)
        rainnn = data_raw.get(('rainnn', year_str))
        avgtmp = data_raw.get(('avgtmp', year_str))
        if rainnn is not None and avgtmp is not None:
            for index, row in rainnn.iterrows():
                lat = row['Latitude']
                lon = row['Longitude']
                key = (lat, lon)
                if key not in result:
                    result[key] = pd.DataFrame()
                rain = row.iloc[2:].values
                temp = avgtmp.iloc[index, 2:].values
                date_index = pd.date_range(start=f'{year}-01-01', periods=len(rain))
                df = pd.DataFrame({'Rainfall': rain, 'Temperature': temp}, index=date_index)
                result[key] = pd.concat([result[key], df])

    # Filter result to keep only key-value pairs with full data
    full_length = (dt.datetime(2023, 12, 31) - dt.datetime(1951, 1, 1)).days + 1
    filtered_result = {key: value for key, value in result.items() if len(value) == full_length}

    # Load GeoLocations data
    geo_df = pd.read_csv("/Users/soumilhooda/Desktop/IndianWeatherRiskManagement/Data/Geolocations/GeoLocations.csv") 

    # Combine weather and geolocation data
    state_data = {}
    for key, df in filtered_result.items():
        lat, lon = key
        state_df = geo_df[(geo_df['Latitude'] == lat) & (geo_df['Longitude'] == lon)]
        if not state_df.empty:
            state_name = state_df['State Name'].iloc[0]
            if state_name != 'MAHARASHTRA':
                if state_name not in state_data:
                    state_data[state_name] = []
                state_data[state_name].append(df)

    # Aggregate data by state
    aggregated_data = {}
    for state, df_list in state_data.items():
        try:
            # Concatenate all DataFrames for the state
            concatenated_df = pd.concat(df_list)
            # Group by date and calculate min, max, and average
            grouped = concatenated_df.groupby(concatenated_df.index).agg({
                'Rainfall': ['min', 'max', 'mean'],
                'Temperature': ['min', 'max', 'mean']
            })
            # Flatten MultiIndex columns
            grouped.columns = ['Rainfall_min', 'Rainfall_max', 'Rainfall_avg',
                               'Temperature_min', 'Temperature_max', 'Temperature_avg']
            aggregated_data[state] = grouped
        except Exception as e:
            logging.error(f"Error aggregating data for state {state}: {e}")
            continue

    logging.info("Data loading and aggregation completed.")
    return aggregated_data

def T_model(x, a, b, c, d, alpha_param, beta, theta):
    """
    Temperature model function based on a modified Ornstein-Uhlenbeck process.
    Parameters:
        x (array): Time in days.
        a, b, c, d, alpha_param, beta, theta: Model parameters.
    Returns:
        array: Modeled temperature values.
    """
    omega = 2 * np.pi / 365.25  # Annual frequency
    return a + b*x + c*x**2 + d*x**3 + alpha_param*np.sin(omega*x + theta) + beta*np.cos(omega*x + theta)

def dT_model(x, a, b, c, d, alpha_param, beta, theta):
    """
    Derivative of the temperature model function.
    Parameters:
        x (array): Time in days.
        a, b, c, d, alpha_param, beta, theta: Model parameters.
    Returns:
        array: Derivative of the modeled temperature values.
    """
    omega = 2 * np.pi / 365.25  # Annual frequency
    return b + 2*c*x + 3*d*x**2 + alpha_param*omega*np.cos(omega*x + theta) - beta*omega*np.sin(omega*x + theta)

def fit_temperature_model(data):
    """
    Fit temperature model to data and evaluate performance.
    Parameters:
        data (dict): Aggregated weather data indexed by state.
    Returns:
        T_models (dict), dT_models (dict), Tbar_params_list (dict), kappas (dict), performance_metrics (dict)
    """
    logging.info("Fitting temperature model and evaluating performance...")

    T_models = {}
    dT_models = {}
    Tbar_params_list = {}
    kappas = {}
    performance_metrics = {}

    for state, df in data.items():
        try:
            # Split data into train (1951-2020) and test (2021-2023) sets
            train_data = df.loc['1951-01-01':'2020-12-31']
            test_data = df.loc['2021-01-01':]

            # Fit model to training data using average temperature
            t_train = (train_data.index - train_data.index[0]).days.values
            temp_train = train_data['Temperature_avg'].values
            initial_guess = [np.mean(temp_train), 0, 0, 0, 10, 10, 0]
            params, _ = curve_fit(T_model, t_train, temp_train, p0=initial_guess, maxfev=10000)

            # Evaluate model on test data
            t_test = (test_data.index - train_data.index[0]).days.values
            temp_test = test_data['Temperature_avg'].values
            temp_pred = T_model(t_test, *params)

            # Calculate performance metrics
            rmse = np.sqrt(np.mean((temp_test - temp_pred) ** 2))
            rmse_percent = (rmse / np.mean(temp_test)) * 100
            std_dev = np.std(temp_test)

            # Fit AR(1) model to residuals
            residuals = temp_test - temp_pred
            ar_model = AutoReg(residuals, lags=1, trend='n').fit()
            gamma = ar_model.params[0]
            kappa = -np.log(gamma)

            # Store results
            T_models[state] = T_model
            dT_models[state] = dT_model
            Tbar_params_list[state] = params
            kappas[state] = kappa
            performance_metrics[state] = {
                'RMSE': rmse,
                'RMSE_percent': rmse_percent,
                'Std_Dev': std_dev
            }
        except Exception as e:
            logging.error(f"Error fitting model for {state}: {e}")
            continue

    logging.info("Temperature model fitting and evaluation completed.")
    return T_models, dT_models, Tbar_params_list, kappas, performance_metrics

def calculate_volatility(data):
    """
    Calculate temperature volatility for each day of the year.
    Parameters:
        data (dict): Aggregated weather data indexed by state.
    Returns:
        volatility (dict): Volatility values for each state.
    """
    logging.info("Calculating temperature volatility...")

    volatility = {}
    for state, df in data.items():
        try:
            temp_vol = df['Temperature_avg'].copy()
            temp_vol = temp_vol.to_frame()
            temp_vol['day'] = temp_vol.index.dayofyear
            vol = temp_vol.groupby(['day'])['Temperature_avg'].agg(['std'])
            days = np.array(vol.index)
            T_std = np.array(vol['std'].values)

            def spline(knots, x, y):
                # Ensure that x is sorted
                sorted_indices = np.argsort(x)
                x_sorted = x[sorted_indices]
                y_sorted = y[sorted_indices]
                # Normalize x for spline
                x_norm = (x_sorted - x_sorted.min()) / (x_sorted.max() - x_sorted.min())
                t, c, k = interpolate.splrep(x_norm, y_sorted, s=3)
                x_new = x_norm
                return interpolate.BSpline(t, c, k)(x_new)

            volatility[state] = spline(15, days, T_std)
        except Exception as e:
            logging.error(f"Error calculating volatility for {state}: {e}")
            continue

    logging.info("Volatility calculation completed.")
    return volatility

def calculate_tref(data):
    """
    Calculate reference temperatures for Heating Degree Days (HDD) and Cooling Degree Days (CDD) for each state.
    Assumes that higher electricity consumption correlates with extreme temperatures.
    Parameters:
        data (dict): Aggregated weather data indexed by state.
    Returns:
        tref (dict): Reference temperatures for each state.
    """
    logging.info("Calculating reference temperatures...")

    tref = {}
    states = ['PUNJAB', 'GUJARAT', 'HARYANA', 'HIMACHAL PRADESH', 'MADHYA PRADESH', 'BIHAR', 
              'UTTAR PRADESH', 'KARNATAKA', 'TELANGANA', 'ANDHRA PRADESH', 'RAJASTHAN', 'ORISSA']

    # Load electricity consumption data
    electricity = pd.read_csv('/Users/soumilhooda/Desktop/IndianWeatherRiskManagement/Data/Electricity/Electricity.csv') 
    electricity["Date"] = pd.to_datetime(electricity["Date"])
    electricity.set_index("Date", inplace=True)
    electricity = electricity.sort_index()

    # Calculate mean temperature for each state
    temperature = pd.DataFrame(index=pd.date_range(start='2018-01-01', end='2023-12-31'))
    for state in states:
        try:
            state_temp = data[state].loc['2018-01-01':'2023-12-31']['Temperature_avg']
            temperature[state] = state_temp
        except Exception as e:
            logging.error(f"Error calculating mean temperature for {state}: {e}")
            continue
    temperature = temperature.sort_index()

    # Define functions to find reference temperatures
    def find_tref_peak_correlation_HDD(state, electricity_data, temperature_data):
        winter_data = electricity_data[electricity_data.index.month.isin(WINTER_MONTHS)]
        winter_temp = temperature_data[electricity_data.index.month.isin(WINTER_MONTHS)]
        correlations = winter_data.rolling(window=90).corr(winter_temp)
        tref_hdd = temperature_data.loc[correlations.idxmin()].mean()
        return tref_hdd

    def find_tref_peak_correlation_CDD(state, electricity_data, temperature_data):
        monsoon_data = electricity_data[electricity_data.index.month.isin(MONSOON_MONTHS)]
        monsoon_temp = temperature_data[electricity_data.index.month.isin(MONSOON_MONTHS)]
        correlations = monsoon_data.rolling(window=90).corr(monsoon_temp)
        tref_cdd = temperature_data.loc[correlations.idxmax()].mean()
        return tref_cdd

    # Calculate reference temperatures for each state
    for state in states:
        try:
            electricity_state = electricity[state].replace('-', np.nan).astype(float)
            electricity_state = electricity_state.interpolate(method='linear')
            temp_state = temperature[state]
            tref[('HDD', state)] = find_tref_peak_correlation_HDD(state, electricity_state, temp_state)
            tref[('CDD', state)] = find_tref_peak_correlation_CDD(state, electricity_state, temp_state)
        except Exception as e:
            logging.error(f"Error calculating tref for {state}: {e}")
            continue

    # Save tref to a CSV file
    tref_df = pd.DataFrame.from_dict(tref, orient='index', columns=['Reference Temperature'])
    tref_df.to_csv('reference_temperatures.csv')
    logging.info("Reference temperatures saved to 'reference_temperatures.csv'.")

    logging.info("Reference temperature calculation completed.")
    return tref

def monte_carlo_temp_with_shocks(trading_dates, Tbar_params_list, volatility, first_ord, kappas, state, no_sims=no_sims, lamda=0,
                                 heatwave_prob=HEATWAVE_PROBABILITY, coldwave_prob=COLDWAVE_PROBABILITY,
                                 shock_duration=SHOCK_DURATION, shock_intensity_heat=SHOCK_INTENSITY_HEAT,
                                 shock_intensity_cold=SHOCK_INTENSITY_COLD):
    """
    Perform Monte Carlo simulation for temperature with shock events (heatwaves and coldwaves).
    Parameters:
        trading_dates (DatetimeIndex): Dates over which to simulate temperatures.
        Tbar_params_list (dict): Temperature model parameters for each state.
        volatility (dict): Volatility values for each state.
        first_ord (dict): First ordinal date for each state.
        kappas (dict): Mean reversion rates for each state.
        state (str): State name.
        no_sims (int): Number of simulations.
        lamda (float): Risk aversion parameter.
        heatwave_prob (float): Probability of a heatwave starting on any day.
        coldwave_prob (float): Probability of a coldwave starting on any day.
        shock_duration (int): Number of days a shock lasts.
        shock_intensity_heat (float): Temperature increase per day during a heatwave.
        shock_intensity_cold (float): Temperature decrease per day during a coldwave.
    Returns:
        mc_sims_df (DataFrame): Simulated temperature paths (days x simulations).
        shock_counts (dict): Counts of heatwaves and coldwaves per simulation.
    """
    kappa = kappas[state]
    Tbar_params = Tbar_params_list[state]
    vol_model = volatility[state]
    first_ord_value = first_ord[state]

    num_days = len(trading_dates)
    M = no_sims
    # Initialize arrays
    mc_sims = np.zeros((num_days, M))
    shock_counts = {'heatwave': np.zeros(M), 'coldwave': np.zeros(M)}
    
    # Initialize temperature
    T_prev = T_model(0, *Tbar_params)
    mc_sims[0, :] = T_prev + (kappa * (Tbar_params[0] - T_prev)) + vol_model[0] * np.random.randn(M) + (-lamda) * (vol_model[0] ** 2)
    
    # Initialize shock trackers
    shock_remaining = np.zeros(M, dtype=int)  # Days remaining in shock
    current_shock = np.array(['none'] * M)   # Current shock type
    
    for day in range(1, num_days):
        # Compute deterministic and mean-reversion components
        t = (trading_dates[day] - dt.datetime.fromordinal(first_ord_value)).days
        Tbar = T_model(t, *Tbar_params)
        dTbar = dT_model(t, *Tbar_params)
        T_det = T_prev + dTbar
        T_mrev = kappa * (Tbar - T_prev)
        
        # Initialize shocks for this day
        shocks = np.zeros(M)
        
        # Determine which simulations start a new shock today
        start_heatwave = np.random.rand(M) < heatwave_prob
        start_coldwave = np.random.rand(M) < coldwave_prob
        
        # Assign shocks
        for sim in range(M):
            if shock_remaining[sim] == 0:
                if start_heatwave[sim]:
                    shock_remaining[sim] = shock_duration
                    current_shock[sim] = 'heatwave'
                    shock_counts['heatwave'][sim] += 1
                elif start_coldwave[sim]:
                    shock_remaining[sim] = shock_duration
                    current_shock[sim] = 'coldwave'
                    shock_counts['coldwave'][sim] += 1
            
            if shock_remaining[sim] > 0:
                if current_shock[sim] == 'heatwave':
                    shocks[sim] += shock_intensity_heat
                elif current_shock[sim] == 'coldwave':
                    shocks[sim] += shock_intensity_cold
                shock_remaining[sim] -= 1
                if shock_remaining[sim] == 0:
                    current_shock[sim] = 'none'
        
        # Random shock component
        random_shock = vol_model[day] * np.random.randn(M)
        
        # Total temperature
        T_i = T_det + T_mrev + shocks + random_shock + (-lamda) * (vol_model[day] ** 2)
        mc_sims[day, :] = T_i
        
        # Update T_prev
        T_prev = T_i

    # Convert to DataFrame
    mc_sims_df = pd.DataFrame(mc_sims, index=trading_dates)
    
    return mc_sims_df, shock_counts

def temperature_option_extreme_analysis(trading_dates, Tbar_params_list, volatility, kappas, r, alpha_val, K, tau, 
                                        first_ord, option_type, tref, opt='c', no_sims=no_sims, lamda=0):
    """
    Calculate temperature option prices based on extreme weather events (heatwaves or coldwaves).
    Also collects data for analysis: number of events per simulation and option payoffs.
    Parameters:
        trading_dates (DatetimeIndex): Dates over which to simulate temperatures.
        Tbar_params_list (dict): Temperature model parameters for each state.
        volatility (dict): Volatility values for each state.
        kappas (dict): Mean reversion rates for each state.
        r (float): Risk-free rate.
        alpha_val (float): Scaling factor for option payoffs.
        K (int): Strike number of events.
        tau (float): Time to maturity in years.
        first_ord (dict): First ordinal date for each state.
        option_type (str): 'heatwave' or 'coldwave'.
        tref (dict): Reference temperatures for each state.
        opt (str): 'c' for call, 'p' for put.
        no_sims (int): Number of simulations.
        lamda (float): Risk aversion parameter.
    Returns:
        option_prices (dict): Option prices and standard errors for each state.
        analysis_data (dict): Per simulation data for analysis.
    """
    logging.info(f"Calculating {opt.upper()} option prices for {option_type} events...")
    
    option_prices = {}
    analysis_data = {}
    
    for state in Tbar_params_list.keys():
        try:
            # Simulate temperatures with shocks
            mc_sims_df, shock_counts = monte_carlo_temp_with_shocks(
                trading_dates, Tbar_params_list, volatility, first_ord, kappas, state, no_sims, lamda,
                heatwave_prob=HEATWAVE_PROBABILITY, coldwave_prob=COLDWAVE_PROBABILITY,
                shock_duration=SHOCK_DURATION, shock_intensity_heat=SHOCK_INTENSITY_HEAT,
                shock_intensity_cold=SHOCK_INTENSITY_COLD
            )
            
            # Depending on option type, get the relevant counts
            if option_type == 'heatwave':
                event_counts = shock_counts['heatwave']
            elif option_type == 'coldwave':
                event_counts = shock_counts['coldwave']
            else:
                logging.error("Invalid option type.")
                option_prices = None
                return None, None
            
            # Calculate option payoffs based on counts and option style
            if opt == 'c' and option_type == 'heatwave':
                # Call option: Payoff if event count > K
                payoffs = alpha_val * np.maximum(event_counts - K, 0)
            elif opt == 'p' and option_type == 'coldwave':
                # Put option: Payoff if event count > K
                payoffs = alpha_val * np.maximum(event_counts - K, 0)
            else:
                logging.error("Invalid combination of option type and option style.")
                option_prices = None
                return None, None
            
            # Discount payoffs
            discounted_payoffs = np.exp(-r * tau) * payoffs
            
            # Option price is average discounted payoff
            C0 = np.mean(discounted_payoffs)
            
            # Standard Error
            SE = np.std(discounted_payoffs) / np.sqrt(no_sims)
            
            # Store option price
            option_prices[state] = {'Price': C0, 'SE': SE}
            
            # Store analysis data
            analysis_data[state] = {
                'Event_Counts': event_counts,
                'Payoffs': discounted_payoffs
            }
        except Exception as e:
            logging.error(f"Error calculating option price for {state}: {e}")
            continue
    
    logging.info(f"{opt.upper()} option price calculation for {option_type} completed.")
    return option_prices, analysis_data

def save_results_to_dataframe(performance_metrics, option_prices_list, summary_stats):
    """
    Combine all results into a single DataFrame.
    Parameters:
        performance_metrics (dict): Temperature model performance metrics.
        option_prices_list (list of dict): List containing option prices dictionaries.
        summary_stats (dict): Summary statistics for option prices.
    Returns:
        combined_df (DataFrame): Combined DataFrame with all results.
    """
    # Initialize list to collect DataFrames
    df_list = []
    
    # Temperature Model Performance Metrics
    perf_df = pd.DataFrame.from_dict(performance_metrics, orient='index')
    perf_df.reset_index(inplace=True)
    perf_df.rename(columns={'index': 'State'}, inplace=True)
    perf_df['Category'] = 'Temperature_Model_Performance'
    df_list.append(perf_df)
    
    # Option Prices
    for option_type_entry in option_prices_list:
        option_df = pd.DataFrame.from_dict(option_type_entry['prices'], orient='index')
        option_df.reset_index(inplace=True)
        option_df.rename(columns={'index': 'State'}, inplace=True)
        option_df['Category'] = 'Option_Price'
        option_df['Season'] = option_type_entry['season']
        option_df['Option_Type'] = option_type_entry['option_type']
        option_df['Option_Style'] = option_type_entry['opt_style']
        option_df['Strike_K'] = option_type_entry['K']
        option_df['Price'] = option_df['Price']
        option_df['SE'] = option_df['SE']
        df_list.append(option_df)
    
    # Summary Statistics
    summary_df = pd.DataFrame.from_dict(summary_stats, orient='index')
    summary_df.reset_index(inplace=True)
    summary_df.rename(columns={'index': 'State'}, inplace=True)
    summary_df['Category'] = 'Summary_Statistics'
    df_list.append(summary_df)
    
    # Combine all DataFrames
    combined_df = pd.concat(df_list, ignore_index=True)
    
    return combined_df

def temperature_option_summary(option_prices):
    """
    Calculate summary statistics (max, min, average) for option prices per state.
    Parameters:
        option_prices (dict): Option prices and standard errors for each state.
    Returns:
        summary (dict): Summary statistics for each state.
    """
    summary = {}
    prices = {state: details['Price'] for state, details in option_prices.items()}

    for state, price in prices.items():
        summary[state] = {
            'Max_Price': price,    # Placeholder, as only one price per state
            'Min_Price': price,    # Placeholder
            'Average_Price': price # Placeholder
        }
    
    return summary

def temperature_option_extreme_summary(option_prices):
    """
    Calculate summary statistics (max, min, average) for option prices per state.
    Parameters:
        option_prices (dict): Option prices and standard errors for each state.
    Returns:
        summary (dict): Summary statistics for each state.
    """
    summary = {}
    prices = [details['Price'] for details in option_prices.values()]
    for state, details in option_prices.items():
        summary[state] = {
            'Max_Price': details['Price'],    # Placeholder, as only one price per state
            'Min_Price': details['Price'],    # Placeholder
            'Average_Price': details['Price'] # Placeholder
        }
    return summary

def save_results(combined_df, filename):
    """
    Save combined results to a CSV file.
    Parameters:
        combined_df (DataFrame): Combined DataFrame with all results.
        filename (str): Name of the output CSV file.
    """
    logging.info(f"Saving combined results to {filename}...")
    combined_df.to_csv(filename, index=False)
    logging.info("Results saved successfully.")

# --------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------

def main():
    logging.info("Starting Weather Risk Model analysis...")

    # Initialize a list to collect all option prices for summary
    all_option_prices = []

    # Initialize a dictionary to collect summary statistics
    summary_statistics = {}

    # Load data
    data = load_data()

    # Fit temperature model and evaluate performance
    T_models, dT_models, Tbar_params_list, kappas, performance_metrics = fit_temperature_model(data)
    
    # Calculate volatility
    volatility = calculate_volatility(data)

    # Calculate reference temperatures
    tref = calculate_tref(data)

    # Define parameters for option pricing
    start_date = dt.datetime(1951, 1, 1)
    first_ord = {state: start_date.toordinal() for state in data.keys()}

    # Define seasons with their respective trading periods
    seasons = {
        'monsoon': {
            'start_date': "2024-05-05",
            'end_date': "2024-07-20",
            'K_values': [1, 2, 3]  # Number of events
        },
        'winter': {
            'start_date': "2024-12-01",
            'end_date': "2025-02-28",
            'K_values': [1, 2, 3]  # Number of events
        }
    }

    # Iterate over each season
    for season, params in seasons.items():
        try:
            season_start = params['start_date']
            season_end = params['end_date']
            trading_dates = pd.date_range(start=season_start, end=season_end, freq='D')
            tau = (dt.datetime.strptime(season_end, "%Y-%m-%d") - dt.datetime.strptime(season_start, "%Y-%m-%d")).days / 365.25
            K_values = params['K_values']

            # Define option types based on season
            if season == 'monsoon':
                option_type = 'coldwave'
                opt_style = 'p'  # Put option for coldwave
            else:
                option_type = 'heatwave'
                opt_style = 'c'  # Call option for heatwave

            # Loop through multiple strike prices
            for K in K_values:
                option_prices, analysis_data = temperature_option_extreme_analysis(
                    trading_dates, Tbar_params_list, volatility, kappas, r, alpha, K, tau, 
                    first_ord, option_type, tref, opt=opt_style, no_sims=no_sims, lamda=0
                )
                
                # Append to all_option_prices for summary
                all_option_prices.append({
                    'season': season,
                    'option_type': option_type,
                    'opt_style': opt_style,
                    'K': K,
                    'prices': option_prices
                })

                # Perform Option Prices vs. Simulated Events Correlation
                correlation_results = {}
                for state in option_prices.keys():
                    if state in analysis_data:
                        event_counts = analysis_data[state]['Event_Counts']
                        payoffs = analysis_data[state]['Payoffs']
                        if len(event_counts) > 1:
                            correlation = np.corrcoef(event_counts, payoffs)[0,1]
                        else:
                            correlation = np.nan
                        correlation_results[state] = {'Correlation': correlation}
                # Store correlation results
                correlation_df = pd.DataFrame.from_dict(correlation_results, orient='index')
                correlation_df.reset_index(inplace=True)
                correlation_df.rename(columns={'index': 'State'}, inplace=True)
                correlation_df['Category'] = f'{season}_{opt_style}_K{K}_Correlation'
                correlation_df.to_csv(f"{season}_{opt_style}_K{K}_correlation.csv", index=False)
                logging.info(f"Saved correlation results for {season} season, K={K}.")

                # Perform Temperature Volatility vs. Option Prices
                # Vary volatility by scaling
                volatility_scalings = [0.8, 1.0, 1.2]  # Example scalings
                for scale in volatility_scalings:
                    scaled_volatility = {state: volatility[state] * scale for state in volatility.keys()}
                    option_prices_scaled, _ = temperature_option_extreme_analysis(
                        trading_dates, Tbar_params_list, scaled_volatility, kappas, r, alpha, K, tau, 
                        first_ord, option_type, tref, opt=opt_style, no_sims=no_sims, lamda=0
                    )
                    # Append to all_option_prices for summary
                    all_option_prices.append({
                        'season': season,
                        'option_type': option_type,
                        'opt_style': opt_style,
                        'K': K,
                        'volatility_scaling': scale,
                        'prices': option_prices_scaled
                    })
                    # Save scaled option prices
                    scaled_option_df = pd.DataFrame.from_dict(option_prices_scaled, orient='index')
                    scaled_option_df.reset_index(inplace=True)
                    scaled_option_df.rename(columns={'index': 'State'}, inplace=True)
                    scaled_option_df['Category'] = 'Option_Price_Scaled_Volatility'
                    scaled_option_df['Season'] = season
                    scaled_option_df['Option_Type'] = option_type
                    scaled_option_df['Option_Style'] = opt_style
                    scaled_option_df['Strike_K'] = K
                    scaled_option_df['Volatility_Scaling'] = scale
                    scaled_option_df.to_csv(f"{season}_{opt_style}_K{K}_volatility_scaling_{scale}.csv", index=False)
                    logging.info(f"Saved scaled option prices for {season} season, K={K}, scaling={scale}.")

            # Extreme Scenarios: Simulate unusually high number of shock events
            extreme_K = max(K_values) + 2  # Example: K increased by 2
            option_prices_extreme, analysis_data_extreme = temperature_option_extreme_analysis(
                trading_dates, Tbar_params_list, volatility, kappas, r, alpha, extreme_K, tau, 
                first_ord, option_type, tref, opt=opt_style, no_sims=no_sims, lamda=0
            )
            all_option_prices.append({
                'season': season,
                'option_type': option_type,
                'opt_style': opt_style,
                'K': extreme_K,
                'prices': option_prices_extreme
            })
            # Save extreme option prices
            extreme_option_df = pd.DataFrame.from_dict(option_prices_extreme, orient='index')
            extreme_option_df.reset_index(inplace=True)
            extreme_option_df.rename(columns={'index': 'State'}, inplace=True)
            extreme_option_df['Category'] = 'Option_Price_Extreme_Scenario'
            extreme_option_df['Season'] = season
            extreme_option_df['Option_Type'] = option_type
            extreme_option_df['Option_Style'] = opt_style
            extreme_option_df['Strike_K'] = extreme_K
            extreme_option_df.to_csv(f"{season}_{opt_style}_K{extreme_K}_extreme_scenario.csv", index=False)
            logging.info(f"Saved extreme scenario option prices for {season} season, K={extreme_K}.")

            # Economic Shocks: Simulate anomalies in electricity consumption data
            # Since no real-world data is present, we'll simulate by modifying reference temperatures
            # For example, reduce HDD and increase CDD to simulate economic downturn
            economic_tref = tref.copy()
            for state in economic_tref.keys():
                if 'HDD' in state:
                    economic_tref[state] *= 0.9  # Reduce HDD reference
                elif 'CDD' in state:
                    economic_tref[state] *= 1.1  # Increase CDD reference

            # Price options with economic shocks
            for K in K_values:
                option_prices_econ, analysis_data_econ = temperature_option_extreme_analysis(
                    trading_dates, Tbar_params_list, volatility, kappas, r, alpha, K, tau, 
                    first_ord, option_type, economic_tref, opt=opt_style, no_sims=no_sims, lamda=0
                )
                all_option_prices.append({
                    'season': season,
                    'option_type': option_type,
                    'opt_style': opt_style,
                    'K': K,
                    'economic_shock': True,
                    'prices': option_prices_econ
                })
                # Save economic shock option prices
                econ_option_df = pd.DataFrame.from_dict(option_prices_econ, orient='index')
                econ_option_df.reset_index(inplace=True)
                econ_option_df.rename(columns={'index': 'State'}, inplace=True)
                econ_option_df['Category'] = 'Option_Price_Economic_Shock'
                econ_option_df['Season'] = season
                econ_option_df['Option_Type'] = option_type
                econ_option_df['Option_Style'] = opt_style
                econ_option_df['Strike_K'] = K
                econ_option_df['Economic_Shock'] = True
                econ_option_df.to_csv(f"{season}_{opt_style}_K{K}_economic_shock.csv", index=False)
                logging.info(f"Saved economic shock option prices for {season} season, K={K}.")

        except Exception as e:
            logging.error(f"Error processing season {season}: {e}")
            continue

    # Prepare summary statistics
    # Initialize a dictionary to hold max, min, average prices
    summary_statistics = {}
    
    for option_entry in all_option_prices:
        season = option_entry['season']
        option_type = option_entry['option_type']
        opt_style = option_entry['opt_style']
        K = option_entry['K']
        prices = option_entry['prices']
        economic_shock = option_entry.get('economic_shock', False)
        volatility_scaling = option_entry.get('volatility_scaling', None)
        
        for state, details in prices.items():
            if state not in summary_statistics:
                summary_statistics[state] = {
                    'Call_Max': None, 'Call_Min': None, 'Call_Avg': [],
                    'Put_Max': None, 'Put_Min': None, 'Put_Avg': []
                }
            if opt_style == 'c' and option_type == 'heatwave':
                # Update call option stats
                if summary_statistics[state]['Call_Max'] is None or details['Price'] > summary_statistics[state]['Call_Max']:
                    summary_statistics[state]['Call_Max'] = details['Price']
                if summary_statistics[state]['Call_Min'] is None or details['Price'] < summary_statistics[state]['Call_Min']:
                    summary_statistics[state]['Call_Min'] = details['Price']
                summary_statistics[state]['Call_Avg'].append(details['Price'])
            elif opt_style == 'p' and option_type == 'coldwave':
                # Update put option stats
                if summary_statistics[state]['Put_Max'] is None or details['Price'] > summary_statistics[state]['Put_Max']:
                    summary_statistics[state]['Put_Max'] = details['Price']
                if summary_statistics[state]['Put_Min'] is None or details['Price'] < summary_statistics[state]['Put_Min']:
                    summary_statistics[state]['Put_Min'] = details['Price']
                summary_statistics[state]['Put_Avg'].append(details['Price'])

    # Calculate average prices
    for state, stats in summary_statistics.items():
        if stats['Call_Avg']:
            stats['Call_Avg'] = np.mean(stats['Call_Avg'])
        else:
            stats['Call_Avg'] = np.nan
        if stats['Put_Avg']:
            stats['Put_Avg'] = np.mean(stats['Put_Avg'])
        else:
            stats['Put_Avg'] = np.nan

    # Convert summary_statistics to a dictionary suitable for DataFrame
    summary_stats = {}
    for state, stats in summary_statistics.items():
        summary_stats[state] = stats

    # Combine all results into a single DataFrame
    combined_df = save_results_to_dataframe(performance_metrics, all_option_prices, summary_stats)

    # Add summary statistics to combined_df
    # Convert summary_stats to DataFrame
    summary_df = pd.DataFrame.from_dict(summary_stats, orient='index')
    summary_df.reset_index(inplace=True)
    summary_df.rename(columns={'index': 'State'}, inplace=True)
    summary_df['Category'] = 'Summary_Statistics'
    combined_df = pd.concat([combined_df, summary_df], ignore_index=True)

    # Save all results to a single CSV file
    save_results(combined_df, 'weather_risk_model_results.csv')

    logging.info("Weather Risk Model analysis completed.")

if __name__ == "__main__":
    main()
