Generalities
############

Folder structure
================
ForSys expects the input folder to have a specific structure. The input folder should contain subfolders for each image, and optionally a user-generated JSON file with connections. The expected structure is as follows:

.. code-block:: none

    /path/to/input_folder
        ├── 0 
        │   └── handCorrection.tif
        ├── 1
        │   └── handCorrection.tif
        ├── 2
        │   └── handCorrection.tif
        ├── 0.tif
        ├── 1.tif
        ├── 2.tif
        └── connections.json

The images can be in any format supported by ForSys libraries: TIFF or NPY.
The connections JSON file is optional and can also be placed in the parent folder. If it is not present, ForSys will use automatically generate connections between frames.

Connections file
================
The connections file is a JSON file that defines the connections between vertices in subsequent frames. It is optional, and if it is not present, ForSys will automatically create connections based on the microsopies. 
The JSON file should have the following structure:

.. code-block:: json

    {
        "0": {
                "0": 5,
                "1": 10},
        "1": {
                "5": 15,
                "10": 20
        }
    }

This example defines connections between vertices at time 0 to time 1, and time 1 to time 2. 
Each outer key corresponds to the starting time point, and defines connections to the following time point in a dictionary. 
Keys of this inner dictionary indicate the ID of a vertex at the original time, and their corresponding value is the new ID (of the identical vertex) at the next time point.