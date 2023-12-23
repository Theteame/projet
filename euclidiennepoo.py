# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 21:31:04 2023

@author: feriel
"""

import csv
import random
import numpy as np
import operator

# Chargement du jeu de données et division en ensembles d'entraînement et de test
def loadDataset(filename, split, trainingSet=[], testSet=[]):
    try :
       with open(filename, 'r') as csvfile:
           lines = csv.reader(csvfile)
           dataset = list(lines)
           for x in range(len(dataset) - 1):
           
               for y in range(4):
                   dataset[x][y] = float(dataset[x][y])

               if random.random() < split:
                   trainingSet.append(dataset[x])
               else:
                   testSet.append(dataset[x])
            
    except FileNotFoundError:
        print("Le fichier spécifié est introuvable.")
    except ValueError as e:
        print(f"Erreur de données : {e}")



def euclideanDistance(instance1, instance2, length):
     instance1 = instance1[:length]
     instance2 = instance2[:length]
     distance = np.sqrt(np.sum((np.array(instance1)-np.array(instance2))**2))
     """print(f"Euclidean distance between {instance1} and {instance2} is {distance}")"""
     return distance 
     

    
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x][:4], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    """print(f"Nearest neighbors: {neighbors}")"""
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]  # Assuming the class label is in the last position of each neighbor
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    maxVotes = sortedVotes[0][1]
    topClasses = [classLabel for classLabel, votes in classVotes.items() if votes == maxVotes]
    predictedClass = random.choice(topClasses)
    """print(f"Class votes: {classVotes}")
    print(f"Predicted class: {predictedClass}")"""
    return predictedClass


def getAccuracy(testSet, predictions):
    # Calcul de l'exactitude des prédictions
    correct = sum(1 for x in range(len(testSet)) if testSet[x][-1] == predictions[x])
    return (correct / float(len(testSet))) * 100.0

def smart_k_search(trainingSet, k_values):
    
    accuracy_dict = {}
    for k in k_values:
        accuracies = []
        for instance in trainingSet:
            # Exclure l'instance en cours pour éviter un surajustement
            neighbors = getNeighbors([x for x in trainingSet if x != instance], instance, k)
            result = getResponse(neighbors)
            accuracies.append(1 if result == instance[-1] else 0)
        accuracy_dict[k] = np.mean(accuracies)
    return accuracy_dict

def main():
    # Charger le jeu de données et diviser en ensembles d'entraînement et de test
    filename = 'iris.csv'  # Remplacez par votre nom de fichier de données
    split = 0.7  # Ratio pour diviser le jeu de données
    trainingSet = []
    testSet = []
    loadDataset(filename, split, trainingSet, testSet)

    # Affichage des ensembles d'entraînement et de test
    print(f"Nombre d'instances dans l'ensemble d'entraînement : {len(trainingSet)}")
    print(f"Nombre d'instances dans l'ensemble de test : {len(testSet)}")

    
    
    smart_k_values = list(range(1, 11))  # Test des valeurs de k de 1 à 10
    print("\nRecherche  pour choisir le meilleur k :")
    smart_accuracy_dict = smart_k_search(trainingSet, smart_k_values)

    best_smart_k = max(smart_accuracy_dict, key=smart_accuracy_dict.get)
    best_accuracy = smart_accuracy_dict[best_smart_k]
    print(f"Best k value using smart search: {best_smart_k}")
    print(f"Accuracy with best k using smart search: {best_accuracy:.2f}%")


    # Utilisation du meilleur k pour prédire sur l'ensemble de test
    predictions = []
    for instance in testSet:
        neighbors = getNeighbors(trainingSet, instance, best_smart_k)
        result = getResponse(neighbors)
        predictions.append(result)

    # Calcul et affichage de l'exactitude finale sur l'ensemble de test
    final_accuracy = getAccuracy(testSet, predictions)
    print(f"\nFinal accuracy with best k on test set: {final_accuracy:.2f}%")
    
   

if __name__ == "__main__":
    main()
