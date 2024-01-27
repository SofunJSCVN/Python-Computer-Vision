# Import the necessary libraries
from timeit import default_timer as timer   
from PIL import Image
from numpy import asarray
import numpy as np 
import os
from tempfile import TemporaryFile
neuralStorage = "./storage/layers/"

def count_files(path):
    return sum([len(files) for _, _, files in os.walk(path)])

def writeNewFile(fileName,Array):
    outfile = TemporaryFile()
    with open("./storage/"+fileName+".npy", 'wb') as f:
        np.save(f, Array)

def saveImageAsLayer(fileName):
    start = timer()
    outfile = TemporaryFile()
    # image checking
    if os.path.isfile("./imgs/"+fileName+".jpg"):
        #convert image to matrix
        matrix = ImageToMatrix("./imgs/"+fileName+".jpg")
        #check layer folder
        if os.path.isfile(neuralStorage+fileName+"/"+fileName+"_0.npy"):
            #count file
            totalFile = count_files(neuralStorage+fileName)
            with open(neuralStorage+fileName+"/"+fileName+"_"+str(totalFile)+".npy", 'wb') as f:
                np.save(f, matrix)
                print("Save Image As Layer Process Time: "+str(timer()-start))
        else:
            #create new folder and data file
            path = os.path.join(neuralStorage, fileName) 
            os.mkdir(path)
            with open(neuralStorage+fileName+"/"+fileName+"_0.npy", 'wb') as f:
                np.save(f, matrix)
                print("Save Image As Layer Process Time: "+str(timer()-start))
            
    else:
        print("Image not found")
        
#convert image to matrix
def ImageToMatrix(imageURL):
    start = timer()
    img = Image.open(imageURL)
    img_matrix = asarray(img)
    print("Convert image to matrix: "+str(timer()-start))
    return  img_matrix

#save image as layer
# def saveImageAsLayer(imageURL,label):
#     img = Image.open(imageURL)
#     img_matrix = asarray(img)
#     writeFile('neuralNetworkStorage',img_matrix)


saveImageAsLayer("genshin")



