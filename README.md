# PC-aircraft-recovery
A recreation of the paper "Passenger-Centric Integrated Airline Schedule and Aircraft Recovery" for MATH3205 at UQ.

https://pubsonline.informs.org/doi/10.1287/trsc.2022.1174

# File Navigation #
- Model construction can be found in `./airline_recovery.py`
- All data files can be found in `./data`, although note that some data is built within the testfiles for psuedo_aus tests
- Tests can be found in all files starting with 'test_' and can be ran using the pytest commands below

# Test Instructions #
- Run `pip install -r requirements.txt` to install all required packages (using a virtual environment is optional)
- From the root directory run `pytest TESTFILE -s` to run particular test files with output. 
- Additionally add `-k TESTNAME` to run specific tests within a test file, these are seperated into functions within the test files.

