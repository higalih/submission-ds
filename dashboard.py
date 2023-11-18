import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

sns.set(style="dark")
df = pd.read_csv("main_data.csv")

# Set the title
st.title("Bike Rental Dashboard")


# Helper function to plot seasonal demand
def plot_seasonal_demand(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="season_name", y="cnt", data=data, palette="viridis", ax=ax)
    plt.title("Permintaan Rental Sepeda tiap Musimnya")
    plt.xlabel("Musim")
    plt.ylabel("Total Rental Sepeda")
    st.pyplot(fig)


# Helper function to plot scatter plot matrix for weather variables
def plot_weather_scatter_matrix(data):
    weather_vars = ["temp", "hum", "windspeed", "weathersit", "cnt"]
    sns.pairplot(data[weather_vars], hue="weathersit", palette="viridis")
    plt.suptitle("Scatter Plot Matrix: Kondisi Cuaca vs Rental Sepeda", y=1.02)
    st.pyplot()


# Helper function to plot box plot for bike rental patterns between weekdays and weekends
def plot_bike_rental_patterns(data, weekday_mapping):
    data["day_type"] = data["weekday"].map(weekday_mapping)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x="day_type", y="cnt", data=data, ax=ax)
    ax.set_title("Pola rental sepeda: Weekdays vs Weekends")
    ax.set_xlabel("Type Hari")
    ax.set_ylabel("Jumlah rental sepeda")
    st.pyplot(fig)


# Map season to human-readable labels
season_mapping = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
df["season_name"] = df["season"].map(season_mapping)

# Map weathersit to human-readable labels
weather_mapping = {
    1: "Clear/Few clouds",
    2: "Mist/Cloudy",
    3: "Light Snow/Rain",
    4: "Heavy Rain/Ice Pallets",
}

# Weekday Mapping
weekday_mapping = {
    0: "Weekend",
    1: "Weekday",
    2: "Weekday",
    3: "Weekday",
    4: "Weekday",
    5: "Weekday",
    6: "Weekend",
}

# Question 1: Demand Across Different Seasons
st.subheader("Question 1: Demand Across Different Seasons")
plot_seasonal_demand(df.groupby("season_name")["cnt"].sum().reset_index())


# Question 2: Correlation Between Weather Conditions and Bike Rentals
st.subheader("Question 2: Correlation Between Weather Conditions and Bike Rentals")
plot_weather_scatter_matrix(df)

df["weathersit_label"] = df["weathersit"].map(weather_mapping)

# Question 3: Bike Rental Patterns Between Weekdays and Weekends
st.subheader("Question 3: Bike Rental Patterns Between Weekdays VS Weekends")
plot_bike_rental_patterns(df, weekday_mapping)

# Display the DataFrame with a checkbox for users to view raw data
if st.checkbox("Show Raw Data"):
    st.write(df)
