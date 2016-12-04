# Final-Project-551
## Description of the dataset


FID - this is just an identifier that associates each row back to the unsampled data

tag - label for each bird

time - time stamp for when the data were collected

depth - depth (m)

X - raw acceleration in the x-axis (g)

Y - raw acceleration in the y-axis (g)

Z - raw acceleration in the z-axis (g)

staticX - mean of the x-axis calculated over a 2 sec window, a measure of average position

staticY - mean of the y-axis calculated over a 2 sec window, a measure of average position

staticZ - mean of the z-axis calculated over a 2 sec window, a measure of average position

pitch - calculated from the static axis above, this measures the overall posture of the animal from 90 deg to -90 deg, the data have been calibrated so pitch should be close to 0 during flight

dynamicX - dynamic movement in the x-axis based on a 2 sec window

dynamicY - dynamic movement in the y-axis based on a 2 sec window

dynamicZ - dynamic movement in the z-axis based on a 2 sec window

ODBA - a composite measurement of dynamic movement in all 3 axes

ground.speed - this is from the GPS, and should be ignored because it is inaccurate

Temperature - this is from the GPS and may help in confirming behaviour classification

Activity - this is from the GPS and should be ignored, because it can mean the bird wasn't moving or that the GPS could not obtain a signal

WBF - the peak frequency of movement from the Z axis over a 10 sec window, this is useful for distinguishing flight and also potentially swimming underwater during a dive

meanPitch240 -  pitch averaged over 4 min window, this can help to distinguish when a bird is a the colony (higher pitch) vs on the water (lower pitch)

sdODBA240 - standard deviation in the ODBA over a 4 min window, this can help distinguish when a bird is still (low values) vs active (high values)

location.lon - eastings (m), from the GPS interpolated to 1 sec intervals

location.lat - northings (m), from the GPS interpolated to 1 sec intervals

speed - ground speed in km/hr calculated from the GPS

behaviour - a rough classfication of behaviour based on the GPS and depth data, which could help with training the algorithm or checking your results

## Instruction for running the code
Use Python, Scikit, Keras ...
## SVM, Decision Trees, Logistic Regression, Naive Bayes
## Hidden Markov Model
## Neural Networks
###Feedforward Neural Net
###Recurrent Neural Net
