# Linear_Regression
Comparision of Linear and polynomial regression models on the same training data-set using Python Numpy.

In this project I have implemented two regression models on the same training data. The file named Linear_Regression uses a simple linear regression with the hypothesis function in 
the form of a straight line (y = mx + c). 

In the second model I added a second order term m'(x)^2, to see how the model behaves. Not surprisingly the Cost function gives a different minima and takes in a whole different learning
rate and total iterations to complete.

The typical minima value for Linear regression model was 4.48 obtained after 1500 iterations and alpha value of 0.01.
For polynomial regression the alpha value was in the order of 0.000001-0.0000001 and the number of iterations needed was of the order of 10,000 to get it to around 5.8 although tweaking
it more may yeild better results. 

Feel free to mess around the code and find something new of your own.

*The training data used here is obtained from the first assignment set of the ML course by Andrew NG on Coursera. The same data was implemented on Octave in the said assignment.
