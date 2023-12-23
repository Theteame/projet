# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 20:38:28 2023

@author: feriel
"""

import csv
import random
import operator
import numpy as np
import matplotlib.pyplot as plt

# Charger le jeu de données
def loadDataset(filename, split, trainingSet=[], testSet=[]):
    try:
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
        raise FileNotFoundError("Le fichier spécifié est introuvable.")
    except ValueError as e:
        raise ValueError(f"Erreur de données : {e}")




# Distance de Manhattan
def manhattanDistance(instance1, instance2, length):
    instance1 = instance1[:length]
    instance2 = instance2[:length]
    distance = sum(np.abs(value1 - value2) for value1, value2 in zip(instance1, instance2))
    return distance

def euclideanDistance(instance1, instance2, length):
    instance1 = instance1[:length]
    instance2 = instance2[:length]
    distance = np.sqrt(np.sum((np.array(instance1)-np.array(instance2))**2))
    return distance

# Trouver les voisins
def getNeighbors(trainingSet, testInstance, k):
    global metric_function
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = metric_function(testInstance, trainingSet[x][:4], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


# Obtenir la réponse
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]  # Class label is in the last position of each neighbor
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1

    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    maxVotes = sortedVotes[0][1]
    topClasses = [classLabel for classLabel, votes in classVotes.items() if votes == maxVotes]
    predictedClass = random.choice(topClasses)

    return predictedClass


# Calculer la précision
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


# Validation croisée Leave-One-Out (LOOCV)
def leave_one_out_cross_validation(dataset, k_values):
    global metric_function
    accuracies = {}
    for metric in metrics:
        accuracy_dict = {}
        metric_function = metric
        print("Utilisation de la metrique:", metric_function.__name__)
        for k in k_values:
            fold_accuracies = []
            for i in range(len(dataset)):
                train_set = [x for j, x in enumerate(dataset) if j != i]
                test_set = [dataset[i]]
                predictions = []
                for instance in test_set:
                    neighbors = getNeighbors(train_set, instance, k)
                    result = getResponse(neighbors)
                    predictions.append(result)
                accuracy = getAccuracy(test_set, predictions)
                fold_accuracies.append(accuracy)
            accuracy_dict[k] = sum(fold_accuracies) / len(fold_accuracies)
        accuracies[metric.__name__] = accuracy_dict
    return accuracies


# Génération de visualisation : Précision en fonction de la valeur de k
def plot_accuracy_results(accuracy_results):
    plt.figure(figsize=(8, 6))
    for metric in accuracy_results:
        plt.plot(list(accuracy_results[metric].keys()), list(accuracy_results[metric].values()), label=metric, marker='o')
    plt.title('Précision en fonction de la valeur de k')
    plt.xlabel('Valeur de k')
    plt.ylabel('Précision (%)')
    plt.grid(True)
    plt.legend()
    plt.show()
    
def main():
    # Charger le jeu de données et diviser en ensembles d'entraînement et de test
    filename = 'iris.csv'  # Remplacer par le nom de votre fichier de données
    split = 0.7 # Ratio de division des données
    trainingSet = []
    testSet = []
    loadDataset(filename, split, trainingSet, testSet)
    print(f"Nombre d'instances dans l'ensemble d'entraînement : {len(trainingSet)}")
    print(f"Nombre d'instances dans l'ensemble de test : {len(testSet)}")

    # Valider les valeurs de k en utilisant la validation croisée LOOCV
    k_values = [i for i in range(1, 11)]  # Tester les valeurs de k de 1 à 20
    accuracy_results_metrics = leave_one_out_cross_validation(trainingSet, k_values)
    best_k = []
    for accuracy_results in accuracy_results_metrics:
        used_metric = accuracy_results
        print(f"\nAccuracy results with metric: {used_metric} for different k values:")
        accuracy_results = accuracy_results_metrics[accuracy_results]
        # Afficher les résultats de la validation croisée pour différentes valeurs de k
        for k, accuracy in accuracy_results.items():
            print(f"K = {k}: Accuracy = {accuracy:.2f}%")
        #Correction ici je extrais la précision du meilleure K de chaque métrique
        current_k_accuracy = accuracy_results[max(accuracy_results, key=accuracy_results.get)]
        #Ensuite je vérifie si elle est plus elevé que la valeur précédente
        if not best_k or current_k_accuracy > best_k[2]:
            best_k = [max(accuracy_results, key=accuracy_results.get),used_metric, current_k_accuracy]

    # Visualiser les résultats
    plot_accuracy_results(accuracy_results_metrics)
    best_metric = best_k[1]
    # Trouver la meilleure valeur de k basée sur la précision moyenne
    best_k = best_k[0]
    print(f"\nBest Metric: {best_metric}")
    print(f"\nBest k value: {best_k}")

    # Utiliser la meilleure valeur de k sur l'ensemble de test pour obtenir la précision finale
    predictions = []
    for instance in testSet:
        neighbors = getNeighbors(trainingSet, instance, best_k)
        result = getResponse(neighbors)
        predictions.append(result)
    final_accuracy = getAccuracy(testSet, predictions)
    print(f"\nFinal accuracy with best k on test set: {final_accuracy:.2f}%")

metrics = [euclideanDistance, manhattanDistance]
metric_function = euclideanDistance

if __name__ == "__main__":
    main()   
    
    
    