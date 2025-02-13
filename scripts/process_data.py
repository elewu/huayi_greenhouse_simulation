import rldev

import os
import pickle
import multiprocessing
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import argparse

def process_time_step_data(data):
    # Convert incoming data dict to rldev.Data type, if applicable
    processed_data = data
    return processed_data

def visualize_data(axs, data):
    # Extract data components
    front_img = data.front_camera_image
    back_img = data.back_camera_image
    laser1 = data.laser1_scan
    laser2 = data.laser2_scan

    # Visualizing using matplotlib
    # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    # Show front camera image
    axs[0, 0].imshow(front_img)
    axs[0, 0].set_title("Front Camera Image")
    axs[0, 0].axis('off')

    # Show back camera image
    axs[0, 1].imshow(back_img)
    axs[0, 1].set_title("Back Camera Image")
    axs[0, 1].axis('off')

    # Plot laser1 scan
    axs[1, 0].plot(laser1)
    axs[1, 0].set_title("Laser1 Scan")

    # Plot laser2 scan
    axs[1, 1].plot(laser2)
    axs[1, 1].set_title("Laser2 Scan")

    plt.tight_layout()
    
    # Pausing for visualization
    # plt.ion()  # Turn on interactive mode
    # plt.show()
    plt.pause(0.5)  # Pause for 0.5 seconds

def save_processed_data(output_dir, time_step, processed_data):
    file_path = os.path.join(output_dir, f'time_step_{time_step}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(processed_data, f)

def process_worker(file_path, output_dir):
    with open(file_path, 'rb') as f:
        sequential_data = pickle.load(f)
    
    import pdb; pdb.set_trace()
    # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    # for time_step, data in sequential_data.items():
    #     processed_data = process_time_step_data(data)

    #     for ax in axs.flatten():
    #         ax.clear()
    #     visualize_data(axs, rldev.Data(**processed_data))
    #     # Uncomment this line if you want to save each processed time step
    #     # save_processed_data(output_dir, time_step, processed_data)
    # plt.close(fig)


def process_data(args):
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    
    pickle_files = [os.path.join(args.input_directory, f) for f in os.listdir(args.input_directory) if f.endswith('.pkl')]
    
    with open(pickle_files[0], 'rb') as f:
        sequential_data1 = pickle.load(f)
    with open(pickle_files[1], 'rb') as f:
        sequential_data2 = pickle.load(f)
    import pdb; pdb.set_trace()


    if args.num_workers > 0:
        with Pool(args.num_workers) as pool:
            pool.starmap(process_worker, [(file, args.output_directory) for file in pickle_files])
    else:
        for file in pickle_files:
            process_worker(file, args.output_directory)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and save data at each time step individually.')
    
    parser.add_argument('--input_directory', type=str, default=os.path.expanduser('~/codespace/agri/huayi_greenhouse_simulation/results/GazeboCollectData/2025-02-12-19:13:17----Nothing/output'), help='Path to the input directory containing pickle files.')
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~/codespace/agri/huayi_greenhouse_simulation/results/GazeboCollectData/2025-02-12-19:13:17----Nothing/processed'), help='Path to the output directory where processed files will be saved.')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of worker processes for parallel processing (set to 0 for debugging).')
    
    args = parser.parse_args()
    
    process_data(args)