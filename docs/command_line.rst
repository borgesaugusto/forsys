Command line interface
======================

Generalities
------------
ForSys can be run using its Command Line Interface, by invoking:

.. code-block:: bash
     
 python -m forsys

Using the -h option allows for help to be displayed. This is only possible from version 1.1.0 onwards.
To check which version you are running, you can use:

.. code-block:: bash

  python -m forsys -v

Example
-------
Static inference
^^^^^^^^^^^^^^^^^
The following command would execute ForSys in "/path/to/input_folder" using the "nnls" method,
using the static modality, in only the first microscopy. Saving the results in "/path/to/save_folder"

.. code-block:: bash

  python -m forsys -f /path/to/input_folder -m nnls -mt 0 -sf /path/to/save_folder

Dynamic inference
^^^^^^^^^^^^^^^^^
The following command would execute ForSys in "/path/to/input_folder" using the "nnls" method,
in the dynamic modality, for the first 5 timepoints. The connections files is located in "/path/to/connections.json"
The results are saved in "/path/to/save_folder"

.. code-block:: bash

  python -m forsys -f /path/to/input_folder -m nnls -mt 5 -sf /path/to/save_folder --dynamic -c /path/to/connections.json


Current flags
-------------
.. list-table:: Flags
  :widths: 30 70
  :header-rows: 1
  :align: center

  * - **Flag** 
    - **Description**
  * - -h, -\-help
    - Shows help message.
  * - -f, -\-folder <FOLDER>
    - Path to the folder containing the input images.
  * - -d, -\-dynamic
    - Whether to use dynamic inference. Default is static.
  * - -m, -\-method <METHOD>
    - Method to use for inference. Options are: 'nnls', 'lsq' and 'lsq_linear. Default is 'nnls'.
  * - -o, -\-output
    - Generates the CSV outputs for cells and membranes.
  * - -vn, -\-velocity_normalization <NORMALIZATION>
    - Velocity normalization factor (only used in dynamic mode). Default is 0.1.
  * - -mt, -\-max_time <MAX_TIME>
    - Maximum time step to infer. Default is 0.
  * - -sf, -\-save_folder <SAVE_FOLDER>
    - Path to the folder in which to save the output. Default is the same as the input folder.
  * - -cc, -\-composite
    - Whether to create composite image with the original microscopies.
  * - -ms, -/-max_size <MAX_SIZE>
    - Maximum multiplier for cellular size. Used to filter out cells that are 
      too big. default is no limit",
  * - -p, -\-pngs
    - Whether to create forsys output of each frame.
  * - -pc, -\-plot_connections
    - Whether to plot the connections based on the connections file.
  * - -c, -\-connections <CONNECTIONS_FILE>
    - Path to the connections file.
  * - -ar, -\-aspect_ratio <ASPECT_RATIO>
    - Aspect ratio of the PNG images. Default is 1.
  * - -y, -\-pressure
    - Plot pressure in the PNG images.
  * - -cb, -\-colorbar
    - Plot colorbar in the PNG images.
  * - -v, -\-version
    - Show the version of the package.

  


