
Before running the code, use requirements.txt to set up the virtual environment:

pip install -r requirements.txt

All required files should already exist in the repo structure to run the code.

To run sampling and search based path plans on all the maps, except for monza sampling based and maze search/sampling based, simply do:

python main.py

This will plot the found paths for those maps. To see monza sampling based, and maze search/sampling based paths, uncomment their tests in the code. These can take up to 3 mins to run.
