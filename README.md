# Projet MSPR 622
 
Classifier des images de légumes et fruits en suivant le [cahier des charges du pdf](/DEVIA%20-%20Sujet%20MSPR%20TPRE622.pdf). Le jeu de données fourni par Kaggle contient 36 classes d'environ 80 images, chaque photo étant prise sans contraintes précises, certaines de dessus, de profile, de face, avec ou sanas fond uni.
 
## Liens
- [Dataset sur Kaggle](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)
- [Repository hebergé sur le github de Rafael](https://github.com/RafffEden/MSPR_TPRE522)
- [Précédents projets réalisés en DevIA](https://jusdeliens.com/epsirennesdevia2324/)
 
## Prérequis
- [Commandes git](https://www.linkedin.com/learning/learning-github-18719601/getting-started-with-github?u=43271628)
- [ndarray numpy](https://www.youtube.com/watch?v=NzDQTrqsxas)
- [Indexing numppy](https://www.youtube.com/watch?v=vw4u9uBFFqU)
![](/res/indexing.png)
 
## Tester l'approche supervisée en machine learning avec scikitlearn
Pour ce projet, nous allons tester différents modele d'intelligence artificiel afin de trouver le plus performant face à notre jeu de données.
Chaque modele peut-être retrouvé dans un dossier dédier avec le code associé, un notebook et les graphiques de performances.

 
### Comparaison classifier scikitlearn
 
👉 [Voir exemple sur scikitlearn](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)
 
### Précédent projet obstacle
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib as mp
import numpy as np
 
data                = pd.read_csv("data.csv")
x                   = data.iloc[:,:-1].values
y                   = data.iloc[:,-1].values
nClasses            = 1+int(y.max())
labelClasses        = [f"class {iClass}" for iClass in range(nClasses)]
xGradMoyByClass     = [data[data.Class==iClass].iloc[:, 0].values for iClass in range(nClasses)]
xGradByClass        = [data[data.Class==iClass].iloc[:, 1].values for iClass in range(nClasses)]
 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2) #20% pour test, 80% pour entrainement
 
print("🧠 Test du modèle svm")
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(x_train,y_train)
classifier.score(x_test,y_test)
y_pred = classifier.predict(x_test)
confMatrix = confusion_matrix(y_test,y_pred)
print(f"Matrice de confusion: \n{confMatrix}")
print(classification_report(y_test,y_pred))
 
# Affichage rapport matplotlib sur dataset
fig, ((ax3,),) = plt.subplots(nrows=1, ncols=1) # Diviser fenetre en grille de 2 colonnes 1 ligne
fig.suptitle("Dataset report", fontsize=16)
colors = ['red', 'lime']
 
cmap = mp.colors.LinearSegmentedColormap.from_list('', ['white', 'black'])
im = ax3.imshow(confMatrix,cmap=cmap)
ax3.set_xticks(np.arange(len(labelClasses)), labels = labelClasses)
ax3.set_yticks(np.arange(len(labelClasses)), labels = labelClasses)
ax3.set_xlabel("Expected")
ax3.set_ylabel("Predicted")
ax3.set_title('Confusion matrix SVM')
fig.colorbar(im, orientation='vertical')
plt.show(block=True)
```
 
### Seaborn et les heatmaps
 
```python
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
array = [[33,2,0,0,0,0,0,0,0,1,3],
        [3,31,0,0,0,0,0,0,0,0,0],
        [0,4,41,0,0,0,0,0,0,0,1],
        [0,1,0,30,0,6,0,0,0,0,1],
        [0,0,0,0,38,10,0,0,0,0,0],
        [0,0,0,3,1,39,0,0,0,0,4],
        [0,2,2,0,4,1,31,0,0,0,2],
        [0,1,0,0,0,0,0,36,0,2,0],
        [0,0,0,0,0,0,1,5,37,5,1],
        [3,0,0,0,0,0,0,0,0,39,0],
        [0,0,0,0,0,0,0,0,0,0,38]]
df_cm = pd.DataFrame(array, index = range(36),
                  columns = range(36))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True)
plt.show()
```
 
### Crossvalidation
 
👉 [Voir video machinelearnia](https://www.youtube.com/watch?v=w_bLGK4Pteo)
 
- **cross_val_score** : pour splitter jeu d'entrainement et calculer un scoring sur chaque split
    ![](/res/cross_val_score.png)
- **validation_curve** : Mieux que cross_val_score pour faire varier l'hyperparamètre plus générique (ici k pour un knnclassifier) dans un range donné (ici de 1 à 49) puis afficher le score via crossvalidation. Ici on prendra k autour de 10 pour avoir les meilleures performances
    ![](/res/validation_curve_1.png)
    ![](/res/validation_curve_2.png)
   
    L'exemple ci-dessus marche pour les KNN, et 👉[celui-ci pour les SVM](https://medium.com/@sreuniversity/unlocking-image-classification-with-scikit-learn-a-journey-into-computer-vision-af2cdc881ad) avec les paramètres **gamma** et **C**
- **GridSearchCV** : utile si vous avez plusieurs paramètres à crossvalider (ici 2 paramètres pour le knn : n_neigbors, metric)
    ![](/res/gridsearchcv1.png)
   
    Une fois entrainé sur les données de train, vous pouvez récupérer les meilleurs paramètres et sauvegarder le modèle issu de ces meilleurs paramètres
    ![](/res/gridsearchcv2.png)
   
    Puis vous afficher la matrice de confusion sur les données de test
    ![](/res/gridsearchcv3.png)
   
    Pour l'interpréter, rappelez vous comment se lit une matrice de confusion
    ![](/res/confusion.png)
   
    Si vous constatez que certaines erreur se retrouve sur des classes remarquables, tenter de la supprimer du dataframe initial avec des [subsets et index conditionnels](https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html)
 
- **learning_curve** : Vous pourrez aussi tracer la learning curve dans vos rapports, pour se rendre compte à quel point le nombre de données influent sur la précision de vos modèles pour conclure s'ils sont adaptés ou non
    ```python
    from sklearn.model_selection import learning_curve
    ```
    ![](/res/learning_curve.png)
   
    Pour interpréter cette courbe, 👉[referez-vous à la documentation](https://scikit-learn.org/stable/modules/learning_curve.html), notamment ce paragraphe:
    ```
    If the training score and the validation score are both low, the estimator will be underfitting. If the training score is high and the validation score is low, the estimator is overfitting and otherwise it is working very well. A low training score and a high validation score is usually not possible.
    ```
