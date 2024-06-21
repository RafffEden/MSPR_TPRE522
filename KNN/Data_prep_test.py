from sklearn.svm import SVC
import matplotlib.pyplot as plt
from skimage.feature import hog
import os
import pandas as pd
from skimage.transform import resize
import skimage.io as io
from tqdm import tqdm
import cv2 as cv

IMG_PATH = "C:\Cours\Licence Data IA\MSPR_622\data"
Data = pd.DataFrame()
UPLOAD_PATH = "Upload"
folders = os.listdir(IMG_PATH)

train_data = pd.DataFrame()

def load_data_images_blurred(folder_path):
    target = 0
    img_error = []
    df = pd.DataFrame()
    #crée un dictionnaire 
    img_dict = {}
    img_dict["Path"] = []
    img_dict["Target"] = []
    #for each class
    for sub_folder in tqdm(os.listdir(folder_path)):
        sub_folder_path = os.path.join(folder_path,sub_folder)
        target +=1 
        #for each image
        for filename in os.listdir(sub_folder_path):
            img_path = os.path.join(sub_folder_path, filename)
            try :
                #lecture de l'image en niveau de gris 
                img = io.imread(img_path, as_gray=True)
                #resize de l'image !Problème d'echelle
                resized_img = resize(img, (32, 32),anti_aliasing=True)

                blur_img = cv.blur(resized_img,(50,50))

                #Application de la fonction HOG pour extraire les features
                fd = hog(
                    blur_img,
                    orientations=8,
                    pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1)
                )
                img_dict["Path"].append(img_path)
                img_dict["Target"].append(target)
                

                #Sauvegarde dans le dictionnaire, 1 ligne = 1 image 
                for i in range(fd.shape[0]) :
                    #Si la colonne n'existe pas on la crée avant l'ajout 
                    colonne = f"FD{i}"
                    if colonne not in img_dict :
                        img_dict[colonne] = [] 
                    img_dict[colonne].append(fd[i])
                
                # print(images_list)
            except :
                img_error.append(f"{filename}, {sub_folder}")
        #passage du dictionnaire en Dataframe 
        df = pd.DataFrame(img_dict)
        
        # data = pd.concat([data,dict_df])
        # print(data)
    print("Image Error :\n" ,img_error)
    #Passage de Dataframe en CSV 
    df.to_csv("data_blurred.csv",sep=";",index=False)
        
    return df 

data = load_data_images_blurred(IMG_PATH)