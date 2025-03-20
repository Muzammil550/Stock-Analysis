import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from psx import stocks
import requests
from datetime import datetime
from prophet import Prophet
#plt.style.use('ggplot')
#%%
# Part 1: Data Retrieval and Preprocessing
# Date rannges
start = datetime(2018,1,1)
end = datetime.today()
hbl = stocks("HBL", start,end)
hbl.reset_index(inplace=True)
ubl = stocks("UBL",start,end)
ubl.reset_index(inplace=True)

hbl.isna().sum() # checking for missing data
ubl.isna().sum() #checking for missing data
# Calculate daily simple returns
hbl['Simple_Return'] = hbl['Close'].pct_change()
ubl['Simple_Return'] = ubl['Close'].pct_change()

# Calculate daily logarithmic returns
hbl['Log_Return'] = np.log(hbl['Close'] / hbl['Close'].shift(1))
ubl['Log_Return'] = np.log(ubl['Close'] / ubl['Close'].shift(1))

# Calculate cumulative returns
hbl['Cumulative_Return'] = (1 + hbl['Simple_Return']).cumprod()
ubl['Cumulative_Return'] = (1 + ubl['Simple_Return']).cumprod()


#%%
# Plot cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(hbl['Date'], hbl['Cumulative_Return'], label='HBL', color='blue')
plt.plot(ubl['Date'], ubl['Cumulative_Return'], label='UBL', color='red')

# Graph formatting
plt.title("Cumulative Returns of HBL and UBL")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid()
plt.show()
#%%

# Part 2: Data Visualization

plt.figure(figsize=(12, 6))
plt.plot(hbl['Date'], hbl['Close'], label='HBL', color='blue')
plt.plot(ubl['Date'], ubl['Close'], label='UBL', color='red')
plt.title("Stock Closing Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(hbl['Date'], hbl['Simple_Return'], label='HBL', color='blue')
axes[0].plot(ubl['Date'], ubl['Simple_Return'], label='UBL', color='red')
axes[0].set_title("Daily Simple Returns")
axes[0].legend()

axes[1].plot(hbl['Date'], hbl['Log_Return'], label='HBL', color='blue')
axes[1].plot(ubl['Date'], ubl['Log_Return'], label='UBL', color='red')
axes[1].set_title("Daily Logarithmic Returns")
axes[1].legend()
plt.show()

# Part 3: Time Series Forecasting with Prophet
# Prepare data for Prophet
prophet_data = ubl[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

# Train Prophet model
model = Prophet()
model.fit(prophet_data)

# Create future dates & forecast
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.title("Stock Price Forecast (UBL)")
plt.show()


# Set the path to Desktop
save_path = "C:/Users/WAJAHAT TRADERS/Desktop/Downloaded_Files/"

# Create directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# Download Image
image_url = "https://d2jdgazzki9vjm.cloudfront.net/tutorial/data-mining/images/data-mining-world-wide-web2.png"
img_response = requests.get(image_url)
if img_response.status_code == 200:
    with open(os.path.join(save_path, "data_mining.png"), "wb") as img_file:
        img_file.write(img_response.content)
    print("Image saved to:", save_path)
else:
    print("Failed to download image.")

# Download Credit Report
credit_report_url = "https://www.pacra.com/summary_report/RR_1244_13537_13-Sep-24.pdf"
pdf_response = requests.get(credit_report_url)
if pdf_response.status_code == 200:
    with open(os.path.join(save_path, "credit_report.pdf"), "wb") as pdf_file:
        pdf_file.write(pdf_response.content)
    print("Credit report saved to:", save_path)
else:
    print("Failed to download credit report.")


print(os.listdir("C:/Users/WAJAHAT TRADERS/Desktop/Downloaded_Files/"))

