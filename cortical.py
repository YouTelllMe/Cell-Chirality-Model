import pandas as pd
import scipy
import os
import numpy as np
import matplotlib.pyplot as plt

def squared(x, time, data):
    alpha, lam = x
    time = np.array(time)
    output = np.multiply(alpha*time**2,np.e**(-lam * time))
    least_squares = 0 
    for column_index in range(10):
        least_squares += np.linalg.norm(data.iloc[:, column_index] - output)
    
    return least_squares

def linear(x, time, data):
    alpha, lam = x
    time = np.array(time)
    output = np.multiply(alpha*time,np.e**(-lam * time))
    least_squares = 0 
    for column_index in range(10):
        least_squares += np.linalg.norm(data.iloc[:, column_index] - output)
    
    return least_squares


if __name__ == "__main__":
    current_dir = os.getcwd()
    # read xlsx files
    corticalflow_xls = pd.ExcelFile(os.path.join(current_dir, "corticalflow.xlsx"))
    # remove first row
    corticalflow = pd.read_excel(corticalflow_xls, "corticalflow").drop(index=[0]).reset_index(drop=True)

    time = corticalflow["Time (s)"]
    corticalflow.drop(columns=["Time (s)"], inplace=True)
    corticalflow_right = corticalflow.iloc[:, 0:10]
    corticalflow_left = corticalflow.iloc[:, 10:20]
    
    alpha_r_sq, lambda_r_sq = scipy.optimize.fmin(squared, [1, 1], args=(time, corticalflow_right))
    alpha_r, lambda_r = scipy.optimize.fmin(linear, [1, 1], args=(time, corticalflow_right))
    alpha_l_sq, lambda_l_sq = scipy.optimize.fmin(squared, [1, 1], args=(time, corticalflow_left))
    alpha_l, lambda_l = scipy.optimize.fmin(linear, [1, 1], args=(time, corticalflow_left))

    # average angles for plotting
    cortical_sum_right = corticalflow_right.iloc[:,0].to_numpy()
    cortical_sum_left = corticalflow_left.iloc[:,0].to_numpy()
    for column in range(1,10):
        cortical_sum_right += corticalflow_right.iloc[:,column].to_numpy()
        cortical_sum_left += corticalflow_left.iloc[:,column].to_numpy()
    cortical_average_right = cortical_sum_right / 10
    cortical_average_left = cortical_sum_left / 10

    fig, ((axLeft, axRight), (axLeftSq, axRightSq)) = plt.subplots(2, 2)

    axLeftSq.plot(time, cortical_average_left)
    axLeftSq.plot(time, np.multiply(alpha_l_sq*time**2,np.e**(-lambda_l_sq * time)))
    axRightSq.plot(time, cortical_average_right)
    axRightSq.plot(time, np.multiply(alpha_r_sq*time**2,np.e**(-lambda_r_sq * time)))

    axLeft.plot(time, cortical_average_left)
    axLeft.plot(time, np.multiply(alpha_l*time,np.e**(-lambda_l * time)))
    axRight.plot(time, cortical_average_right)
    axRight.plot(time, np.multiply(alpha_r*time,np.e**(-lambda_r * time)))

    print(alpha_r_sq, lambda_r_sq)
    plt.savefig("cortical_fit.png")