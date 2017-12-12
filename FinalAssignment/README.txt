The dataset represents a communication system having 3 BSs and maximum of 20 users present in any sample. The area of interest within the communication system is represented as a map; specifically, an area of about 150x16 m^2 has been divided into grids of 2x2m^2. The resulting map is then seen as a matrix, and that matrix is linearized into a vector form. Each grid (or matrix element) denotes whether a user or a BS is located within it. If a user is placed, the grid is encoded as 1; if a BS is located within the grid, it is encoded as 2; and for rest of the map a 0 has been appended. 

The location of the user can be either accurate or it could have an inaccuracy defined by a normal distribution with standard deviation (std.) of 0.1m, 0.25m, or 0.4m. Thus, there are four categories of scenarios for which the datasets are generated.

The aim of constructing this dataset is to define the BS-user associations as well as to allocate system resources to the users present in the system. The output, therefore, is carrying this association+resource allocation information within it. One way to represent the output is to encode the whole information into a single output variable; this we call as the 'coded_output'. If this information is disintegrated into separate variables, we call this 'uncoded_output'.

In summary:
- There are 4 categories of data, each has 2 files: one for training data, and the other for test data.
- The categories are named as follows: pos_perf (accurate user position info.), pos_error_0o1 (when inaccuracy with std. of 0.1m is present in the available user position info. at BS), pos_error_0o25 (same as for pos_error_0o1, but with std. of 0.25m), and pos_error_0o4 (for error with std. 0.4m in user position info.).
- Altogether there are 72,000 samples. 2/3rd of this has been written as training dataset (i.e. 48,000 samples), and the rest is written as test dataset (i.e. 24,000 samples).


Details for datasets in uncoded_output:

- Each training/test dataset has the input (i.e. linearized grid-based map) and the output variables together in one sample (i.e. one row).
- The number of output variables is 27: 9 output variables per BS. So the last 27 columns in each row are the output variables.
- The output variables no. 1,4,7,10,13,16,19,22 and 25 have values ranging from 0-32.
- The output variables no. 2,5,8,11,14,17,20,23 and 24 have values ranging from 0-8.
- The output variables no. 3,6,9,12,15,18,21,24 and 27 have values ranging from 0-15.

Details for datasets in coded_output:

- Each training/test dataset has the input (i.e. encoded grid-based map) and the output variables together in one sample (i.e. one row).
- The number of output variables is 9: 3 output variables per BS. So the last 9 columns in each row are the output variables.
- Each output variable has values ranging from 0-12288.



Comparison of performance:

In my experiments, I used the learning algorithm to train on accurate user position information data and test it for all 4 categories, i.e. accurate and 3 inaccurate user position information cases. Further, to see the advantage of learning on inaccurate user position data, I trained the learning algorithm on inaccurate datasets and tested them for the relevant inaccurate info. datasets. An example is to test the inaccurate user position data with 0.1m std. using the learning structure trained on accurate user position data and also for the learning structure trained on inaccurate user position data for 0.1m std. of inaccuracy.

