# import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# load data from CSV file
df = pd.read_csv('data/KF_Measurement.csv', names = ['time', 'x', 'y', 'x_truth', 'y_truth'])

# reset index and drop unnecessary columns
df = df.reset_index()
df = df.drop('y_truth', axis = 'columns')
df = df.rename(columns={"index": "time", "time": "x", "x" : 'y', "y" : 'x_truth', "x_truth" : 'y_truth'})

# print the first few rows of the dataframe
print(df)

# plot the ground truth trajectory
plt.plot(df['x_truth'], df['y_truth'])
plt.show()

# get the initial state from the dataframe
x_i = df.iloc[0][3:5].values[0]
y_i = df.iloc[0][3:5].values[1]

# define constants and variables
t = 1 # time step
std = 10 # standard deviation of measurement
var = std**2 # variance of measurement
v_max = 250 # max velocity of plane
Q_o = 0.1 # process noise covariance

# define initial state
x = np.array([x_i, y_i, 0, 0])

# define state transition matrix
A = np.array([[1, 0, t, 0],
              [0, 1, 0, t],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# define process covariance matrix
P = np.array([[var, 0, 0, 0],
              [0, var, 0, 0],
              [0, 0, (v_max/3)**2, 0],
              [0, 0, 0, (v_max/3)**2]])

# define process noise covariance matrix
Q = Q_o * np.eye(4)

# define measurement noise covariance matrix
R = np.array([[var, 0],
              [0, var]])

# define measurement matrix
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

# initialize list to store estimated states
x_estimate = [np.array([x_i, y_i])]

# iterate over the dataframe and perform Kalman filter steps
for dt in range(len(df)):
  # predict step
  # x(k) = A*x_k-1 + B*u(k)
  # Here u(k) = 0
  current_x = A @ x

  # find Kalman Gain K:
  # 1. find P(k+1|k) = A * P(k|k)AT + Q*R
  current_P = A @ P @ A.T + Q

  # 2. find K = (P(k+1|k) * HT)/(H*P(k+1|k)*HT + R)
  numerator = (current_P @ H.T)
  denominator = (H @ current_P @ H.T + R)
  K = numerator @ np.linalg.inv(denominator)

  # 3. find x(k+1|k) = x(k+1|k-1) + K*(z(k) - H*x(k+1|k-1))
  z = df.iloc[dt][1:3].values # get x and y from dataset
  new_x = current_x + K @ (z - H @ current_x)

  # update step
  # Update P = (I - K*H)*P_k-1
  P = (np.eye(4) - K @ H) @ current_P

  # store the estimated state
  x_estimate.append(x)

  # update the state for the next iteration
  x = new_x

# extract the x and y coordinates from the estimated states
x_predict = [pt[0] for pt in x_estimate]
y_predict = [pt[1] for pt in x_estimate]

# calculate the mean squared error between the predicted and true states
x_mse = mean_squared_error(df['x_truth'], x_predict[1:])
y_mse = mean_squared_error(df['y_truth'], y_predict[1:])

print(f'X MSE: {x_mse}')
print(f'Y MSE: {y_mse}')

# plot the predicted trajectory
plt.plot(df['x_truth'], df['y_truth'], marker='o')
plt.plot(x_predict, y_predict, marker='o')
plt.legend(['Truth', 'Prediction'])
plt.show()

