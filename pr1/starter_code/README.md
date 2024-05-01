
Before running the code, use requirements.txt to set up the virtual environment:

pip install -r requirements.txt

There are two main files to be run for part A and B. All required files should already exist in the repo structure.

-------------------------------------------------------------------------------------

Part A

python main_part_a.py

will calculate policies for all the known environments and save the sequence gifs in "gif_known" as well as generate a .csv file containing the optimal control sequences for each. The run time should be very quick.

python part_a_performance_test.py

will test different initial conditions on the policy calculated for a known environment.

-------------------------------------------------------------------------------------

Part B

python main_part_b.py

will calculate a single policy at the start for any 8x8 random environment and save the sequence gifs in "gif_random" as well as generate a .csv file containing the optimal control sequences for each. The run time can vary depending on the hardware but known to run for about 3 minutes on an i9 processor.
