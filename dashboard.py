import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency

sns.set(style="dark")
# df = pd.read_csv("file:dashboard/main_data.csv")
df = pd.read_csv("main_data.csv")

# bike_rental = pd.read_csv("file:data/day.csv")

# Set the title
st.title("Bike Rental EDA Dashboard")

# Display some EDA results
st.subheader("Demand Across Different Seasons")
st.bar_chart(df.groupby("season")["cnt"].mean())

st.subheader("Correlation with Weather Conditions")
st.line_chart(df[["temp", "hum", "windspeed", "cnt"]])

st.subheader("Bike Rental Patterns on Weekdays vs. Weekends")
st.bar_chart(df.groupby("weekday")["cnt"].mean())
