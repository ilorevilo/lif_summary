# lif-summary

## easy extraction of microscope images/ videos from a .lif-file and simultaneous creation of a Powerpoint-Summary

#### what to do
  - execute lif_summary.py in the same folder where your .lif-files are located
  - the script will create a new folder for each .lif-file and extract all images and videos
  - all images and videos will be exported in a compressed format (.jpg and .mp4) as well as raw (.tif or .tif-series)
  - a .ppt-summary will automatically be created, containing all images in a 2x3 grid
  - after finishing the .lif-file will be moved into the newly created folder

#### required main packages
  - `Python 3`
  - `python-pptx` for the Powerpoint-Summary creation
  - `bioformats` and `javabridge` for extracting data from the .lif-file
  - `tifffile` for writing .tifs, `cv2` for writing videos
  - `PIL` and `skimage` for image operations
