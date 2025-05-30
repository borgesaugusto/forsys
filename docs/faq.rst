FAQ
####
If you believe something is missing from this FAQ, please open an issue on the GitHub repository or contact us via email.

1) Anaconda gives and error saying "Conda not recognized"
*********************************************************
If you are using Anaconda and encounter the error "Conda not recognized", it typically means that the Anaconda installation path is not added to your system's PATH environment variable.

2) When using ForSys, I get an error saying "KeyError"
*********************************************************
This error usually occurs when ForSys is not able to read the segmentation, due to there being cases that it cannot account for. It is recommended to check the segmentation to 
ensure that no cells are open, isolated, or have holes. If the segmentation is correct, please open an issue on the GitHub repository with the image and segmentation files so that we can investigate the problem.