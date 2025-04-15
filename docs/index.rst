ForSys: a non-invasive open-source analysis software to infer stress from microscopic time series 
=======================================================================================================
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.11282555.svg
  :target: https://doi.org/10.5281/zenodo.11282555
.. image:: https://badge.fury.io/py/forsys.svg
   :target: https://badge.fury.io/py/forsys
.. image:: https://codecov.io/gh/borgesaugusto/forsys/graph/badge.svg?token=09Z4YKV8IG
   :target: https://codecov.io/gh/borgesaugusto/forsys
.. image:: https://readthedocs.org/projects/forsys/badge/?version=latest
   :target: https://forsys.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

ForSys is a software tool that allows stress and pressure inference in static and dynamic systems. 
It is based on the analysis of static images as well as time series images, and it is non-invasive, meaning that it does not require any modification of the system under analysis. It only neeeds segmented microscopy images. 
The software is open-source.

.. toctree::
   :hidden:

   Home <self>
   API reference <_autosummary/forsys>


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Examples
   
   In silico static inference <examples/static_in_silico>
   In silico time lapse inference <examples/dynamic_in_silico>
   In vivo static inference <examples/static_in_vivo>
   In vivo time lapse inference <examples/dynamic_in_vivo>


.. toctree::
   :maxdepth: 2
   :caption: User Guide

   command_line
   gui

.. toctree::
   :maxdepth: 3
   :caption: About

   installation
   cite
