##################
Installation Guide
##################

Installing ForSys package
=========================

We recommend using Conda to create a virtual environment for ForSys. This will help you manage dependencies. 
For instance, using `Miniconda <https://www.anaconda.com/docs/getting-started/miniconda/main>`_ you can create a new environment with Python 3.8 or higher:

.. code-block:: bash

  conda create -n forsys python=3.11

Then, activate the environment:

.. code-block:: bash

  conda activate forsys

In this example, ``forsys`` is the name of the environment. You can choose any name you prefer.


Installing with pip
-------------------
After activating the enviroment, run

.. code-block:: bash

  pip install forsys

Or, to install a specific version, use the following command:

.. code-block:: bash

    pip install forsys==<version>

And replace `<version>` with the desired version number (i.e 1.0.0).
For reference, you can check the available versions on `PyPi <https://pypi.org/project/forsys/#history>`_.
To run our Fiji plugin, ForSys needs to be at least version 1.1.0. This version also has a :doc:`command_line`.



Installing from source
----------------------
1. Clone the repository

.. code-block:: bash

  git clone git@github.com:borgesaugusto/forsys.git

2. Change to the directory

.. code-block:: bash

  cd forsys

3. Install the package

.. code-block:: bash

  pip install .

Requirements
------------
- Python 3.8 or higher
