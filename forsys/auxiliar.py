import os
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