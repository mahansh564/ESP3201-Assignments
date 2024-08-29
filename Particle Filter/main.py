import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from read_data import *
import time

from particle_filter import initialize_particles, mean_pose, sample_motion_model, eval_sensor_model, resample_particles

# Add random seed for generating comparable pseudo random numbers
np.random.seed(1)

# Plot preferences, interactive plotting mode
plt.ion()  # Enable interactive mode for live updating
plt.show()

def plot_state(ax, particles, landmarks, actual_robot_pos, map_limits):
    # Visualizes the state of the particle filter.
    
    xs = [particle['x'] for particle in particles]
    ys = [particle['y'] for particle in particles]

    lx = [landmarks[i + 1][0] for i in range(len(landmarks))]
    ly = [landmarks[i + 1][1] for i in range(len(landmarks))]

    # Mean pose as current estimate
    estimated_pose = mean_pose(particles)

    # Plot filter state
    ax.clear()  # Clear current axes
    ax.plot(xs, ys, 'r.')
    ax.plot(actual_robot_pos[0], actual_robot_pos[1], 'ko', markersize=5)
    ax.plot(lx, ly, 'bo', markersize=10)
    ax.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy', scale_units='xy', color='g')
    ax.quiver(actual_robot_pos[0], actual_robot_pos[1], np.cos(actual_robot_pos[2]), np.sin(actual_robot_pos[2]), angles='xy', scale_units='xy')
    ax.set_xlim(map_limits[0], map_limits[1])
    ax.set_ylim(map_limits[2], map_limits[3])
    ax.set_title('Particle Filter State')

def update_robot_pos(actual_robot_pos, odometry, map_limits):
    # Move robot to new position, based on old positions, the odometry, the motion noise 
    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # The motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    noise = [0.1, 0.1, 0.05, 0.05]
    
    # Standard deviations of motion noise
    sigma_delta_rot1 = noise[0] * abs(delta_rot1) + noise[1] * delta_trans
    sigma_delta_trans = noise[2] * delta_trans + noise[3] * (abs(delta_rot1) + abs(delta_rot2))
    sigma_delta_rot2 = noise[0] * abs(delta_rot2) + noise[1] * delta_trans
        
    noisy_delta_rot1 = delta_rot1 + np.random.normal(0, sigma_delta_rot1)
    noisy_delta_trans = delta_trans + np.random.normal(0, sigma_delta_trans)
    noisy_delta_rot2 = delta_rot2 + np.random.normal(0, sigma_delta_rot2)

    # Calculate new particle pose
    x = actual_robot_pos[0] + noisy_delta_trans * np.cos(actual_robot_pos[2] + noisy_delta_rot1)
    y = actual_robot_pos[1] + noisy_delta_trans * np.sin(actual_robot_pos[2] + noisy_delta_rot1)
    theta = actual_robot_pos[2] + noisy_delta_rot1 + noisy_delta_rot2
 
    new_robot_pos = [x, y, theta]    
    
    return new_robot_pos

def update_sensor_readings(actual_robot_pos, landmarks):
    # Computes the sensor readings
    # The employed sensor model is range only.
    sigma_r = 0.2
    
    lm_ids = []
    ranges = []
    bearings = []
    
    for lm_id in landmarks.keys():

        lx = landmarks[lm_id][0]
        ly = landmarks[lm_id][1]
        px = actual_robot_pos[0]
        py = actual_robot_pos[1]
        
        # Calculate range measurement with added noise
        meas_range = np.sqrt((lx - px) ** 2 + (ly - py) ** 2) + np.random.normal(loc=0.0, scale=sigma_r)
        meas_bearing = 0  # Bearing is not computed
        
        lm_ids.append(int(lm_id))    
        ranges.append(meas_range)
        bearings.append(meas_bearing)
        
        sensor_readings = {'id': lm_ids, 'range': ranges, 'bearing': bearings} 
        
    return sensor_readings

def main():
    # Implementation of a particle filter for robot pose estimation
    print("Reading landmark positions")
    landmarks = read_world('data/world.dat')

    # Initialize the particles
    map_limits = [0, 10, 0, 10]
    particles = initialize_particles(1000, map_limits)

    actual_robot_pos = read_pos("data/pos.dat")

    # To store the localization error and computational time for each iteration
    localization_errors = []
    computational_times = []

    # Create a figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Run particle filter
    for iteration in range(100):
        start_time = time.time()  # Start measuring time for this iteration
        
        sensor_readings = 1

        # Plot the current state in the first subplot
        plot_state(axs[0], particles, landmarks, actual_robot_pos, map_limits)

        # Move actual robot and sense state
        odometry = {'r1': 0.1, 't': 0.6, 'r2': 0.15}  # Constant motion
        actual_robot_pos = update_robot_pos(actual_robot_pos, odometry, map_limits)
        sensor_readings = update_sensor_readings(actual_robot_pos, landmarks)

        # Predict particles by sampling from motion model with odometry info
        new_particles = sample_motion_model(odometry, particles)

        # Calculate importance weights according to sensor model
        weights = eval_sensor_model(sensor_readings, new_particles, landmarks)

        # Resample new particle set according to their importance weights
        particles = resample_particles(new_particles, weights)

        predicted_pos = mean_pose(particles)
        pred_error = np.sqrt((predicted_pos[0] - actual_robot_pos[0]) ** 2 + 
                             (predicted_pos[1] - actual_robot_pos[1]) ** 2)

        localization_errors.append(pred_error)
        computational_times.append(time.time() - start_time)  # Calculate time for this iteration
        
        print('iter: %d, localization error: %.3f' % (iteration, pred_error))
        
        # Live update for localization error vs iteration in the second subplot
        axs[1].clear()
        axs[1].plot(localization_errors, label='Localization Error', color='b')
        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel('Localization Error')
        axs[1].set_title('Localization Error vs Iteration')
        axs[1].legend()
        axs[1].grid(True)

        # Live update for computational complexity vs iteration in the third subplot
        axs[2].clear()
        axs[2].plot(computational_times, label='Computational Time', color='r')
        axs[2].set_xlabel('Iteration')
        axs[2].set_ylabel('Time (seconds)')
        axs[2].set_title('Computational Complexity vs Iteration')
        axs[2].legend()
        axs[2].grid(True)

        plt.pause(0.01)  # Pause for a short period to update the subplots
    
    plt.show()
    plt.pause(10)

if __name__ == "__main__":
    main()