import rldev

import os
import pickle
import tqdm
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import argparse


def save_processed_data(output_dir, time_step, processed_data):
    file_path = os.path.join(output_dir, f'time_step_{time_step}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(processed_data, f)


def visualize_data(axs, data):
    # Extract data components
    front_img = data.front_camera_image
    back_img = data.back_camera_image
    laser1 = data.laser1_scan
    laser2 = data.laser2_scan

    # Visualizing using matplotlib
    # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    # Show front camera image
    try:
        axs[0, 0].imshow(front_img)
    except:
        import pdb; pdb.set_trace()
        pass
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


def get_sub_dict(original_dict, keys):
    """Extracts a sub-dictionary containing only the specified keys."""
    return {key: original_dict[key] for key in keys if key in original_dict}


def process_worker(file_path, output_dir, vis=False):
    base_dir, data_type = os.path.split(output_dir)
    extra_dir = os.path.join(base_dir, 'extra/sensors', data_type)
    os.makedirs(extra_dir, exist_ok=True)

    with open(file_path, 'rb') as f:
        sequential_data = pickle.load(f)
    
    if vis:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    for time_step, data in sequential_data.items():
        if any([v is None for v in data.values()]):
            continue
        save_path = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(file_path))[0]}_step_{time_step}.pkl')
        sensor_save_path =os.path.join(extra_dir, f'{os.path.splitext(os.path.basename(file_path))[0]}_step_{time_step}.pkl')
        data_pose = get_sub_dict(data, ['pose', 'linear_velocity', 'angular_velocity'])
        data_sensor = get_sub_dict(data, ['front_camera_image', 'back_camera_image', 'laser1_scan', 'laser2_scan', 'laser3_scan'])

        with open(save_path, 'wb') as f:
            pickle.dump(data_pose, f)
        with open(sensor_save_path, 'wb') as f:
            pickle.dump(data_sensor, f)

        if vis:
            for ax in axs.flatten():
                ax.clear()
            visualize_data(axs, rldev.Data(**data))
    if vis:
        plt.close(fig)
    return


def process_data(pickle_files, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    if args.num_workers > 0:
        with Pool(args.num_workers) as pool:
            # Use tqdm to wrap the iterable for displaying a progress bar
            list(tqdm.tqdm(pool.starmap(process_worker, [(file, output_directory) for file in pickle_files]), total=len(pickle_files)))
    else:
        for file in pickle_files:
            process_worker(file, output_directory)
    return




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and save data at each time step individually.')
    parser.add_argument('--input_directory', type=str, default=os.path.expanduser('~/codespace/agri/huayi_greenhouse_simulation/results/GazeboCollectData/2025-02-13-12:43:35----Nothing/output'), help='Path to the input directory containing pickle files.')
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~/codespace/agri/huayi_greenhouse_simulation/results/GazeboCollectData/2025-02-13-12:43:35----Nothing/processed'), help='Path to the output directory where processed files will be saved.')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of worker processes for parallel processing (set to 0 for debugging).')
    args = parser.parse_args()
    
    pickle_files = [os.path.join(args.input_directory, f) for f in os.listdir(args.input_directory) if f.endswith('.pkl')]
    pickle_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))

    train_files = pickle_files[:-10]
    val_files = pickle_files[-10:]

    process_data(train_files, os.path.join(args.output_directory, 'train'))
    process_data(val_files, os.path.join(args.output_directory, 'val'))
