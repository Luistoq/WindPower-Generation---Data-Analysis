import streamlit as st

st.set_page_config(
    page_title="WindPower Analysis",
    page_icon="🧊",
    layout="wide",
)

import streamlit as st

def app():
    # Adding a sidebar with your picture, title, and contact information
    st.sidebar.subheader("For collaboration on similar projects, contact me.")
    st.sidebar.image("screenshots/Designer.png", width=250)  # Adjust width as needed
    st.sidebar.write("Senior Data Science & Digital Transformation Lead")
    st.sidebar.markdown("Email: [luistoq@outlook.com](mailto:luistoq@outlook.com)")
    st.sidebar.markdown("[LinkedIn Profile](https://www.linkedin.com/in/luis-toral-251007/)")

    st.title("🌬️ Wind Power Generation Analysis App")

    st.markdown("""
    ## Welcome to the Wind Power Generation Analysis App!
    This interactive application allows you to explore and analyze wind power generation data across four different locations. Dive into the specifics of how various meteorological factors influence wind power generation.

    ### 📊 Dataset Information
    The dataset, derived from wind turbines at four distinct locations, encompasses comprehensive measurements, providing insights into the dynamics of wind energy production. The information includes:

    - **Temperature**: Recorded at 2 meters above the surface.
    - **Relative Humidity**: Measured at 2 meters above the surface.
    - **Dew Point**: Noted at 2 meters above the surface.
    - **Wind Speeds**: Observed at both 10 meters and 100 meters above the surface.
    - **Wind Directions**: Tracked at 10 meters and 100 meters above the surface.
    - **Wind Gusts**: Monitored at 10 meters above the surface.
    - **Power Generation**: Data on the output from the wind turbines.

    ### 🌐 Source of the Dataset
    This dataset is sourced from [Kaggle: Wind Power Generation Data](https://www.kaggle.com/datasets/mubashirrahim/wind-power-generation-data-forecasting), where you can find more details and download the dataset for personal use or analysis.

    ### 🚀 Using the App
    To start exploring the data:
    
    1. **Select a Location**: Choose from the four locations available in the sidebar.
    2. **Pick a Variable**: Select a meteorological variable or power generation data to analyze.
    3. **Choose Timeframe**: Opt for the timeframe for your analysis - Hourly, Weekly, Monthly, etc.
    4. **Analyze the Trends**: The main area will display a line plot of your selected variable over time, reflecting your specified choices.

    Enjoy exploring the data, and uncover interesting patterns and insights into wind power generation!

    > To begin analysis, select *'Wind Power Analysis'* from the sidebar menu.
            
    """)


if __name__ == "__main__":
    app()
