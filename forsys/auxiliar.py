import os
import json
def create_folders(resFolder):
    upperFolder = os.path.join(resFolder, "complete")
    directory = os.path.join(upperFolder, "accelerations")
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.join(upperFolder, "accelerations_cutoff")
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.join(upperFolder, "velocities")
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.join(upperFolder, "velocities_cutoff")
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.join(upperFolder, "heatmaps")
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.join(upperFolder, "connections")
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.join(upperFolder, "distributions")
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.join(upperFolder, "forces_accel")
    if not os.path.exists(directory):
        os.makedirs(directory)    
    create_directory("forces_static", upperFolder)
    directory = os.path.join(upperFolder, "forces_vel")
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.join(upperFolder, "hist_accel")
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.join(upperFolder, "hist_vel")
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = os.path.join(upperFolder, "csvs")
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_directory(name, upperFolder):
    directory = os.path.join(upperFolder, name)
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_initial_guess(guess_file, min_time, max_time):
    try:
        with open(guess_file) as jfile:
            initial_guess = json.load(jfile)
            initial_guess = {int(k): {int(kin): vin for kin, vin in v.items()} for k, v in initial_guess.items()}
    except FileNotFoundError:
        initial_guess = {}
        print("No guess file, using zero guess")
    
    number_of_frames = max_time - min_time

    initial_guess = {k: {} for k in range(number_of_frames)
                    if k not in initial_guess.keys()} | initial_guess
    return initial_guess