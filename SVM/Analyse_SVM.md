# Analyse des tests avec SVM

 Afin de trouver le modèle le plus performant dans notre cas on va appliquer plusieur modèle sur nos jeu de donnée

 ## Application simple d'un SVM

 Dans un premier temps, nous allons appliquer un SVM avec les paramètres pas défaut, l'idée de ce premiers et de voir si un SVM peut-être adapté à nos données et au problèmes qui l'on veut résoudre. 

 Puis une fois tester sans thuner le modèle, on va le passer dans GridSearch afin de tirer un modèle avec des paramètres qui s'adapte le mieux à nos données. 

 Ainsi avec ces deux modèles on pourra juger si un SVM est adapté à notre cas ou si notre jeu de données peut convenir à notre problèmatique.


 ## Explication des performances 
 Quand on regarde la courbe d'apprentissage de nos deux modèles de SVM, on voit se répéter le phénomène suivant :

 ![alt text](learning_graph_grid.png)

 Que ça soit pour le modèle standard ou le modèle optimisé, on peut voir un grand écart entre la courbe d'entrainement et la courbe de validation, cette écart peut être expliqué par une effet d'overfitting du modele c'est à dire que le modèle s'adapte trop aux données aux détriment de la précision générale, elle sera donc performante sur une petit volume de donnée mais ne s'ameliorera pas ou peu après. Ce phénomène peut-être dû au grand nombre de variable explicative qui rend l'utilisation d'un SVM difficilement adapté à nos données tels quelles ont été préparer. 

 ### Axes d'amélioration 

L'axe principale d'amélioration serait de reduire la quantité de variable explicative de nos images afin d'avoir moins de compléxité pour le modèle ou bien essayé de plus finethuner encore le modèle pour l'adapter, ce dernier choix semble le moins viable étant donner que le modèle à aussi ses limitations en terme de résolution. 