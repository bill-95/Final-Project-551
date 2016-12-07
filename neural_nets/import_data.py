from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import csv

def getData(fileName, fields_to_extract):
    X = []
    Y = []


    with open(fileName) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            #print row["FID"]
            new_row = []
            for field in fields_to_extract:
                new_row.append(row[field])
            X.append(new_row)
            Y.append(row["Behaviour"])
    return X, Y




if __name__ == "__main__": 
    print ("Importing data")

    fields_to_extract = ["X", "Y", "Z", "staticX", "staticY", "staticZ", "pitch", "dynamicX"]   

    X, Y = getData("GPS_Behaviours/Blue10_Accel_2s_Window_Sampled.csv", fields_to_extract)
    print (X)

