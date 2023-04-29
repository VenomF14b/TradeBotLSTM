import pyodbc
import logging
import configparser
import MetaTrader5 as mt5
import numpy as np
import datetime as dt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create logger for EURUSDmodel
EURUSDmodellog = logging.getLogger('EURUSDmodel')
file_handler1 = logging.FileHandler('EURUSD\EURUSDmodel.log')
file_handler1.setFormatter(formatter)
EURUSDmodellog.addHandler(file_handler1)
EURUSDmodellog.setLevel(logging.DEBUG)
logging.info('\n''\n')

# Create logger for EURUSDtdata
EURUSDtdatalog = logging.getLogger('EURUSDtdata')
file_handler2 = logging.FileHandler('EURUSD\EURUSDtdata.log')
file_handler2.setFormatter(formatter)
EURUSDtdatalog.addHandler(file_handler2)
EURUSDtdatalog.setLevel(logging.DEBUG)

# Create logger for EURUSDtparam
EURUSDtparamlog = logging.getLogger('EURUSDtparam')
file_handler3 = logging.FileHandler('EURUSD\EURUSDtparam.log')
file_handler3.setFormatter(formatter)
EURUSDtparamlog.addHandler(file_handler3)
EURUSDtparamlog.setLevel(logging.DEBUG)

# Create logger for EURUSDconparam
EURUSDconparamlog = logging.getLogger('EURUSDconparam')
file_handler4 = logging.FileHandler('EURUSD\EURUSDconparam.log')
file_handler4.setFormatter(formatter)
EURUSDconparamlog.addHandler(file_handler4)
EURUSDconparamlog.setLevel(logging.DEBUG)





# Read the configuration file, Script variables************************************************************************************
#**********************************************************************************************************************************
config = configparser.ConfigParser()
config.read('EURUSD/configEURUSD.ini')
training_params = config['Training Parameters']

#Update adata
symbol = "EURUSDm" #Symbol selector
passedtime = days= + int(training_params.get('passedtime', '7')) #Historical data time adjustor in days
#Trainerai
traineraiRowselector = int(training_params.get('trainerairowselector', '10080'))
traineraiEpochs = int(training_params.get('traineraiepochs', '100'))
traineraiBatchsize = int(training_params.get('traineraibatchsize', '5'))
TraineraiSplit = 0.90
TaimodelS = "EURUSD/EURUSD.h5" #Model to save
#Program
TotalRuntime = 21600  # seconds
ScriptInterval = 60  # seconds
MaxTrades = 1
#**********************************************************************************************************************************

# Training the new model code *****************************************************************************************************
#**********************************************************************************************************************************
# MT5 Initialize, Logging infomation **********************************************************************************************
#**********************************************************************************************************************************
if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()
symbol = symbol
timeframe = mt5.TIMEFRAME_M1
EURUSDtdatalog.debug("Connection to MetaTrader5 successful")
#**********************************************************************************************************************************

# Calculate the start and end times, Logging information, Update start and end times **********************************************
#**********************************************************************************************************************************
end_time = dt.datetime.now()
end_time += dt.timedelta()
start_time = end_time - dt.timedelta(passedtime)
EURUSDtdatalog.debug("Data Time Start = " + str(start_time))
EURUSDtdatalog.debug("Data Time End = " + str(end_time))
EURUSDtdatalog.debug("Getting historical data")
rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
rates = np.array(rates)
start_time = end_time
end_time = dt.datetime.now()
#**********************************************************************************************************************************

# Establish a connection to the database, Check if timestamp already exists, Writes the data to the database, Cmd information *****
#**********************************************************************************************************************************
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=VENOM-CLIENT\SQLEXPRESS;'
                      'Database=TRADEBOT;'
                      'Trusted_Connection=yes;')
cursor = conn.cursor()
for rate in rates:
    timestamp = int(rate[0])
    cursor.execute("SELECT COUNT(*) FROM EURUSDAdata WHERE timestamp = ?", (timestamp,))
    count = cursor.fetchone()[0]
    if count == 0:
         # Write the data to the database
         values = [timestamp, float(rate[1]), float(rate[2]), float(rate[3]), float(rate[4]), float(rate[5]), float(rate[6]), float(rate[7])]
         cursor.execute("INSERT INTO EURUSDAdata (timestamp, [open], high, low, [close], tick_volume, spread, real_volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", tuple(values))
         print(values)
conn.commit()
cursor.close()
conn.close()
print("MT data is up to date")
#**********************************************************************************************************************************

# Connect to the SQL Express database, Selects data to use for training ***********************************************************
#**********************************************************************************************************************************
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=VENOM-CLIENT\SQLEXPRESS;'
                      'Database=TRADEBOT;'
                      'Trusted_Connection=yes;')
query = f"SELECT TOP ({traineraiRowselector}) timestamp, [open], high, low, [close], tick_volume, spread, real_volume FROM EURUSDAdata ORDER BY timestamp DESC"
data = []
cursor = conn.cursor()
cursor.execute(query)
for row in cursor:
    converted_row = []
    for item in row:
        if isinstance(item, str):
            try:
                converted_item = int(item)
            except ValueError:
                converted_item = item
        else:
            converted_item = item
        converted_row.append(converted_item)
    data.append(converted_row)
cursor.close()
conn.close()
#**********************************************************************************************************************************

# Preprocess the data
# Perform feature engineering and normalization as required

# Split the data into training and testing sets
train_data = data[:5000]
test_data = data[5000:]

# Define the neural network model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Evaluate the model on the test set
score = model.evaluate(X_test, y_test, batch_size=32)

# Make predictions using the model
predictions = model.predict(X_test)

# Print the accuracy metrics
print('Test loss:', score)

