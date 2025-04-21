# filepath: /Users/mac/Desktop/devprojects/playground/car-tracker/utils.py
import gspread
from google.oauth2.service_account import Credentials
import datetime
import pandas as pd
import streamlit as st  # needed to access st.secrets


def get_google_sheet_worksheet():
    SCOPE = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    # Load credentials from Streamlit secrets
    credentials = Credentials.from_service_account_info(st.secrets["gcp"], scopes=SCOPE)
    gc = gspread.authorize(credentials)
    sh = gc.open("VehicleData")
    worksheet = sh.sheet1
    return worksheet


def store_today_data(car_count, bus_count, truck_count):
    worksheet = get_google_sheet_worksheet()
    today = datetime.date.today().strftime("%Y-%m-%d")
    worksheet.append_row([today, car_count, truck_count, bus_count])


def load_sheet_data():
    worksheet = get_google_sheet_worksheet()
    records = worksheet.get_all_records()
    if not records:
        return pd.DataFrame(
            columns=["Date", "Number of Cars", "Number of Truck", "Number of Bus"]
        )
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    return df
