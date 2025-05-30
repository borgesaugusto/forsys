GUI Reference
=============

Installation in Fiji
--------------------
To install the Fiji plugin, inside Fiji go to Help > Update > Manage update sites.
Then select the option "Add unlisted site" and add the following URL: 

`https://sites.imagej.net/ForSys`

Then select the ForSys update site and click on "Apply changes". For more information, you can
read more about `Plugins <https://imagej.net/plugins/>`_ in the Fiji documentation.

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
- ForSys expects the Input Folder to have a certain structure.
- The JSON file with the connections is optional, and it can also be in the parent folder.
Information about this can be found in the :doc:`generalities` section.


GUI Options
-----------

  .. image:: images/gui_reference.png
     :width: 800

