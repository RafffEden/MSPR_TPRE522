from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import matplotlib as mp 

def show_graph_train(conf_matrix, data, classes:int):
    """ Affiche des graphs pour chacune des metrics, un nuage de point prennant les metrics en abs et ord et
    affiche la matrice de confusion du modèle 
    """
    plt.close()
    data = pd.read_csv(data)
    labels = []
    fig ,ax = plt.subplots()

    cmap = mp.colors.LinearSegmentedColormap.from_list('',['white','black'])
    im = ax.imshow(conf_matrix,cmap = cmap)
    # à rendre generique pitié 
    for i in range(1, classes +1 ):
        labels.append(f"Classe  {i}")
    ax.set_xticks(np.arange(2),labels = ["classe 0","classe 1"] )
    ax.set_yticks(np.arange(2),labels = ["classe 0","classe 1"] )
    ax.set_xlabel("Expected")
    ax.set_ylabel("Predicted")
    ax.set_title('Confusion matrix')
    fig.colorbar(im,orientation = 'vertical')


    plt.savefig("Graph")
    plt.show(block = False)

def training_SVN(path : str) -> SVC:
    """Entrainement d'un modèle SVC à partir d'un ficher passer en paramètre
        et renvoie le modèle entrainée """	

    Data = pd.read_csv(path,delimiter=";")
    print(Data.shape)
    X = Data.iloc[:,:-1].values #récupère les variables explicatives 
    Y = Data.iloc[:,-1].values # récupère les variables à prédire 

    # Data.head() #récupère les 5 premières lignes 

    # Diviser les données afin d'obtenir un jeu de test et un jeu d'entrainement 
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)
    print(X_train)

    #Création du Modèle SVM et entrainement 
    print("____ Création du modele ____")
    modele = SVC(kernel = 'linear', random_state = 0)
    modele.fit(X_train,Y_train)
    print("____ Modele crée et entrainé ____")

    #Première prediction sur le jeu de test 
    print("____ Début de prédiction ____")
    Y_pred = modele.predict(X_test)
    print("Prediction :\n",Y_pred)

    #Matrice de Confusion 
    print("____ matrice de confusion ____")
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    print(conf_matrix)
    show_graph_train(conf_matrix)

    return modele

training_SVN("data.csv")
