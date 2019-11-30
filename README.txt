# Research Project

## Project structure
 - *helpers.py* - includes implementations of seam carving and various other helper functions  
 - *main.py* - script generating output files requires for the project
 - *out/* - output directory for main.py results
 - *test_img/* - output directory for notebook output results
 - *images/* - input files
 - *Removal.ipynb* - Jupyter notebook dedicated to figure 5 and operations related to that experiment
 - *Insertion.ipynb* - Jupyter notebook dedicated to figure 8 and operations related to that experiment
 - *Retargeting.ipynb* - Jupyter notebook dedicated to figure 7 and operations related to that experiment

## Requirements
 - Python 2.7
 - numpy library
 - OpenCV library with cv2 python library
  
## How to run
To generate output files execute in the directory of this project command:  
```
python main.py
```
In the main.py file you can comment out steps that you wish to skip in the end of the file (see comments in the end of the file).  
Also, since *fig7* step is taking a lot of time you can uncomment line in _fig7()_ function (see comments) to speed up this process, but generated output will be for smaller image.  

## Other
Project also includes several Jupyter notebook files that contain parts of main.py code in them plus additional cells showing the results at different steps.  
You can just view the notebooks of rerun them to generate more output files.

