#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 15:02:50 2024

@author: irlahanum
"""

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

class StockAnalysis:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.model = LinearRegression()

    def download_data(self):
        """Download stock data from Yahoo Finance."""
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)

    def display_data(self):
        """Display the first few rows of the data."""
        print(self.data.head())

    def plot_prices(self):
        """Plot the stock prices."""
        plt.figure(figsize=(15, 6))
        plt.plot(self.data.index, self.data["Adj Close"], label="Adj Close")
        plt.plot(self.data.index, self.data["High"], label="High")
        plt.plot(self.data.index, self.data["Low"], label="Low")
        plt.grid(linestyle=":")
        plt.ylabel("Price ($)")
        plt.title(f"{self.ticker} stock price from {self.start_date} to {self.end_date}")
        plt.legend()
        plt.show()

    def perform_linear_regression(self):
        """Perform linear regression on the adjusted close prices."""
        self.data['dates_numeric'] = self.data.index.map(pd.Timestamp.timestamp)
        x = self.data['dates_numeric'].values.reshape(-1, 1)  # Reshape for sklearn
        y = self.data['Adj Close']
        self.model.fit(x, y)
        y_pred = self.model.predict(x)
        return y_pred

    def plot_with_prediction(self, y_pred):
        """Plot the stock prices along with the linear regression prediction."""
        plt.figure(figsize=(15, 6))
        plt.plot(self.data.index, self.data["Adj Close"], label="Adj Close")
        plt.plot(self.data.index, self.data["High"], label="High")
        plt.plot(self.data.index, self.data["Low"], label="Low")
        plt.plot(self.data.index, y_pred, label="Extrapolation", color="navy", linestyle="-")
        plt.grid(linestyle=":")
        plt.ylabel("Price ($)")
        plt.title(f"{self.ticker} stock price from {self.start_date} to {self.end_date} and prediction based on its trend")
        plt.legend()
        plt.savefig("Apple_Stock_Price.png", dpi=300)
        plt.show()

# Usage
if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2024-01-01"

    stock_analysis = StockAnalysis(ticker, start_date, end_date)
    stock_analysis.download_data()
    stock_analysis.display_data()
    stock_analysis.plot_prices()
    y_pred = stock_analysis.perform_linear_regression()
    stock_analysis.plot_with_prediction(y_pred)
