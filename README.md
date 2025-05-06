The sphere and torus files are from the flow matching review paper's accompanying github, they were used to generate the examples in the intro of the report. 
I do not think these files can be run standalone, you have to import the rest of that flow matching github project.
The pendulum file was found on github, it was modified to produce the data that was used for flow matching on the torus.
The harmonics folder contains an R and STAN file that was used to generate some data, primarily draws from the spherical harmonics,
but also the couple of distributions used in the KL divergence section of the report.
The Euclidean file was used to perform flow matching and regularization, producing the example in the 3rd part of the report.
The imputation file was again used to perform flow matching, imputing missing values for some observations. This produced the example
found in the last section of part 3 of the report.
Finally, the KL divergence file was used to calculate KL divergence between the results of flow matching, and the true distribution.
The file is from an individual's github, it was modified for use in the report.
