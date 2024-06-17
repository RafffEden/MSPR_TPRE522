from sklearn.model_selection import train_test_split,cross_val_score,validation_curve,GridSearchCV,learning_curve
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
    im = ax.imshow(conf_matrix,cmap = "plasma")
    # à rendre generique pitié 
    for i in range(1, classes +1 ):
        labels.append(f"{i}")
    ax.set_xticks(np.arange(classes),labels = labels )
    ax.set_yticks(np.arange(classes),labels = labels )
    ax.set_xlabel("Expected")
    ax.set_ylabel("Predicted")
    ax.set_title('Confusion matrix')
    fig.colorbar(im,orientation = 'vertical')


    plt.savefig("Graph")
    # plt.show(block = False)

def training_SVM(path : str) -> SVC:
    """Entrainement d'un modèle SVC à partir d'un ficher passer en paramètre
        et renvoie le modèle entrainée """	

    Data = pd.read_csv(path,delimiter=";")
    print(Data.shape)
    X = Data.iloc[:,2:].values #récupère les variables explicatives 
    Y = Data.iloc[:,1].values # récupère les variables à prédire 
    Params = [{"gamma": [0.01,0.001,0.0001], "C":[1,10,100,1000]}]
    
    print("y = ",Y)
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
    # show_graph_train(conf_matrix,path,36)
    
    #validation curve
    #Cross validation 
    val = cross_val_score(modele,X,Y,cv=5)
    print(val)

    N, train_score,val_score = learning_curve(SVC()
                                              ,X_train,Y_train,train_sizes= np.linspace(0.1,1.0,10), cv=5)
    
    plt.plot(N,train_score.mean(axis=1), label='train')
    plt.plot(N, val_score.mean(axis=1), label='validation')
    plt.xlabel("train_sizes")
    plt.legend()
    plt.savefig("learning_graph")
    
    grid_search = GridSearchCV(SVC(),Params,verbose=10)
    
    grid_search.fit(X_train,Y_train)
    
    best_estimator = grid_search.best_estimator_
    N, train_score,val_score = learning_curve(best_estimator
                                              ,X_train,Y_train,train_sizes= np.linspace(0.1,1.0,10), cv=5)
    
    plt.plot(N,train_score.mean(axis=1), label='train')
    plt.plot(N, val_score.mean(axis=1), label='validation')
    plt.xlabel("train_sizes")
    plt.legend()
    plt.savefig("learning_graph_grid")
    
    y_prediction = best_estimator.predict(X_test)
    conf_matrix = confusion_matrix(Y_test, y_prediction)
    show_graph_train(conf_matrix,path,36)
    
    return modele

training_SVM("data.csv")
