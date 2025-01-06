
Before running the code, use requirements.txt to set up the virtual environment:

pip install -r requirements.txt

All required files should already exist in the repo structure to run the code. There is a policy.txt file that is used to load the precompute policy for GPI. Additionally, state space and control space information is needed and stored in their respectively named txt files.
All nominal tunings are in the files.

To run CEC NLP:
python main.py nlp

To run GPI:
python main.py gpi

To compute the transition matrix and store it on disk, run:
python GPI_CalculateTransitionMatrix.py