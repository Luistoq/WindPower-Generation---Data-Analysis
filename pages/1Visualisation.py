import streamlit as st
import pandas as pd
import plotly.express as px

# Function to load data
@st.cache_data  # Corrected decorator here
def load_data(location):
    df = pd.read_csv(f'dataset/{location}.csv')
    df['Time'] = pd.to_datetime(df['Time'])  # Ensure Time is in datetime format
    return df

# Generalized function for plotting
def plot_data_production(df, variable, frequency):
    resample_map = {'Hourly': 'H', 'Weekly': 'W', 'Monthly': 'M', 'Quarterly': 'Q', 'Yearly': 'A'}
    df_resampled = df.set_index('Time').resample(resample_map[frequency]).mean().reset_index()

    fig = px.line(df_resampled, x='Time', y=variable, title=f'{variable} Over Time - {frequency} Average')
    return fig

variable_descriptions = {
    "Time": "Hour of the day when readings occurred",
    "temperature_2m": "Temperature in degrees Fahrenheit at 2 meters above the surface",
    "relativehumidity_2m": "Relative humidity (as a percentage) at 2 meters above the surface",
    "dewpoint_2m": "Dew point in degrees Fahrenheit at 2 meters above the surface",
    "windspeed_10m": "Wind speed in meters per second at 10 meters above the surface",
    "windspeed_100m": "Wind speed in meters per second at 100 meters above the surface",
    "winddirection_10m": "Wind direction in degrees (0-360) at 10 meters above the surface (see notes)",
    "winddirection_100m": "Wind direction in degrees (0-360) at 100 meters above the surface (see notes)",
    "windgusts_10m": "Wind gusts in meters per second at 10 meters above the surface",
    "Power": "Turbine output, normalized to be between 0 and 1 (i.e., a percentage of maximum potential output)"
}

# Sidebar for user inputs
location = st.sidebar.selectbox('Select Location', ('Location1', 'Location2', 'Location3', 'Location4'))
data = load_data(location)

# Dropdown for selecting the variable
selected_variable = st.sidebar.selectbox('Select Variable', [col for col in data.columns if col != 'Time'])

#Map description with variable selected by user
df_info = pd.DataFrame(list(variable_descriptions.items()), columns=['Variable', 'Description'])

# Dropdown for selecting the frequency
frequency = st.sidebar.selectbox('Select Frequency', ('Hourly', 'Weekly', 'Monthly', 'Quarterly', 'Yearly'))

# Main content
st.header(f'Wind Power Generation Analysis: {location}')

# Plotting the data based on the selected frequency and variable
fig = plot_data_production(data, selected_variable, frequency)
st.plotly_chart(fig)

# Displaying variable description
variable_description = df_info[df_info['Variable'] == selected_variable]
if not variable_description.empty:
    st.dataframe(variable_description, hide_index=True, use_container_width=True)