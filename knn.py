import numpy as np
import sys
import pickle
from collections import Counter

def readData(filename):
    photo_labels = []
    orient_class = []
    image_vector = []
    with open(filename, "r") as file:
        for line in file.read().split("\n")[:-1]:
            line_string = line.split(" ")
            photo_labels.append(line_string[0])
            orient_class.append(int(line_string[1]))
            image_vector.append([int(i) for i in line_string[2:]])
    return photo_labels, orient_class, image_vector

def train(filename, model_file):
    names, orientation, vector = readData(filename)
    data = zip(vector, orientation)
    with open(model_file, "wb") as f:
        pickle.dump(data, f)
    f.close()
    print("model trained...")
    
def getVotes(knn):
    #a = [np.asscalar(i[0]) for i in knn]
    b = list(knn[:,1])
    output = Counter(b)
    label = max(output, key=output.get)
    return int(label)
    
def test(test_file, model_file, k):
    print("testing...")
    #extract train data from the pickle file
    with open(model_file, "rb") as f:
        raw = pickle.load(f)
        train_vec, train_class = zip(*raw) 
    f.close()
    
    #getting test data
    test_name, test_class, test_vec = readData(test_file)
    
    counter = 0
    label_list = []
    for i in range(len(test_class)):
        #calculate the distance
        dist = euclideanDistance(train_vec, test_vec[i])
        dist = np.squeeze(np.asarray(dist))
        mat = np.transpose(np.stack((dist, train_class)))
        
        #sort the array
        sortedArr = mat[mat[:,0].argsort()]
        
        #get the nearest neighbors
        knn = sortedArr[:k]
        
        #Vote and predict the label
        predicted_label = getVotes(knn)
        
        label_list.append(predicted_label)
        
        #calculate accuracy
        if(predicted_label == test_class[i]):
            counter += 1
        
    acc = (counter/len(test_class)) * 100
    print("accuracy: ", acc)
    
    #write to file
    f = open("output.txt", "w")
    for i in range(len(label_list)):
        f.write(str(test_name[i]) + " "+ str(label_list[i])+"\n")
    f.close()
    
def euclideanDistance(vec_list, test_vector):
    vec_matrix = np.matrix(vec_list)
    return np.sqrt(np.sum(np.power((vec_matrix - test_vector),2), axis=1))

def manhattanDistance(vec_list, test_vector):
    vec_matrix = np.matrix(vec_list)
    return np.sum(np.abs(np.subtract(vec_matrix, test_vector)), axis=1)


#MAIN PROGRAM
#
if __name__ == '__main__':
    
    #get Args
    (procedure, procedure_data, procedure_model, model) = sys.argv[1:]

    if model=='knn':
        if procedure=='train':
            train(procedure_data, procedure_model)
        if procedure=='test':
            test(procedure_data, procedure_model, 13)
    elif model=='tree':
        run_tree(procedure,procedure_data,procedure_model)
    else:
        print('Wrong Model Entered')
