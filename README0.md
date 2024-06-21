# MSPR_TPRE522

## Cahier des charges

Concevoir un modèle performant capable de reconnaitre des images de fruits et de légumes. 

Métrique choisie : precision du modèle sur des données de test (20% du dataset) tels que rapport de classification



## Analyse du jeu de données 

36 classe - 80 image 
Le jeu de données fourni par Kaggle contient 36 classes d'environ 80 images, chaque photo étant prise sans contraintes précises, certaines de dessus, de profile, de face, avec ou sans fond uni avec rotation sur les trois axes.

Compléxité du jeu de données avec forte abtraction necessaire pour dicerner tout les degrés de liberté des classes. Afin de traiter cette compléxité, nous allons tester différents critère de caractérisation et pré-traitement en partant du plus simple en allant vers le plus abstrait 


## 1- Machine learning avec HOG 

*à détailler le fonctionnement de hog*

Commençons d'abord avec le feature descriptor [hog](https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html) de chaque images et constater les performances. Expliquer comment les performances sont impactées par la nature des images et ce que caractérise l'algorithme hog.

*placer les rapports sur chaque modèle*

*conclusion sur l'ensemble des modèles*
