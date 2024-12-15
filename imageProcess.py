# Import the necessary libraries
from timeit import default_timer as timer   
from PIL import Image
from numpy import asarray
import numpy as np 
import os
from tempfile import TemporaryFile
import json

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

#@jit(target_backend='cuda') 
def detect(array,network):
    print("network size: "+str(len(network)))
    labels = {}
    for x in range(len(array)):
        for y in range(len(array[x])):
            try:
                labelist = network[str(x)][str(y)][str(array[x][y])]
                for lb in labelist:
                    try:
                        labels[lb] = labels[lb]+ 1
                    except:
                        labels[lb] = 1
            except:
                b = True
    return labels

def insertdata(array,network,name):
    for x in range(len(array)):
        try:
            network[str(x)] 
        except:
            network[str(x)] = {}
        for y in range(len(array[x])):
            try:
                if(network[str(x)][str(y)]):
                    try:
                        try:
                            link = network[str(x)][str(y)][str(array[x][y])][name]
                            network[str(x)][str(y)][str(array[x][y])][name] = link+1
                        except:
                            network[str(x)][str(y)][str(array[x][y])][name] = 1
                    except:
                        network[str(x)][str(y)][str(array[x][y])] = {str(name):1}
            except:
                network[str(x)][str(y)] = {str(array[x][y]):{str(name):1}}
    return network

def train(image,name):
    #load data
    f = open('./storage/data.json')
    data = json.load(f)
    #readFile
    network = insertdata(ImageToMatrix(image),data,name)
    saveDasta(network) 

def saveDasta(data):
    with open('./storage/data.json', 'w') as f:
        json.dump(data, f)

start = timer()
f = open('./storage/data.json')
data = json.load(f)
print(detect(ImageToMatrix("./imgs/detect.jpg"),data))
#train("./imgs/v0.jpg","V charactor")



print("process_time: "+str(timer()-start))




