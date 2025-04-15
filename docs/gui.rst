GUI Reference
=============

Installation in Fiji
--------------------
...

Usage
-----
1. In fiji Open the plugin from the menu Plugins > ForSys, or using the search bar.
2. Select the folder containing the images to be analyzed.

   .. image:: images/select_input.png
    :width: 400

3. The Results folder will be automatically placed in the Parent directory.
   If there is a connections JSON file, it will be used to create the connections.
4. Make sure that the ForSys package is installed in the appropiate Conda enviroment, 
   given by the Conda Env.
5. Now the inference can be run.


Considerations
----------------
ForSys expects the Input Folder to have a structure like

.. code-block:: none

    /path/to/input_folder
        ├── 1 
        │   └── handCorrection.tif
        ├── 2
        │   └── handCorrection.tif
        ├── 3
        │   └── handCorrection.tif
        ├── 1.tif
        ├── 2.tif
        ├── 3.tif
        └── connections.json

The JSON file with the connections is optional, it can also be in the parent folder.


GUI Options
-----------

  .. image:: images/gui_reference.png
     :width: 800

