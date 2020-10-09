import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import sklearn
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


class ImmothepPrediction:

    def predict(self, global_dataset, id):
        """Predict the popularity of a movie, using its tconst.
        Args:
            global_dataset ([dataframe]): dataframe with all the movies features
            id ([str]): tconst (like tt000001)
        """
        # *********** DEBUT DE LA PREPARATION DES DONNEES DE REFERENCE *********
        # global_dataset =  pd.util.hash_pandas_object(global_dataset[['actors', 'genres', 'producer', 'writer', 'composer', 'region']], encoding='utf8')
        global_dataset = global_dataset[['actors', 'genres', 'producer', 'writer', 'composer', 'region']].astype(float)
        # Vérification des valeurs nulles :
        # print('NOMBRE DE VALEURS NULLES :\n', dataset.isnull().sum())
        # print('*******************')
        # *********** FIN DE LA PREPARATION DES DONNEES DE REFERENCE *********


        # *********** DEBUT REGRESSION LOGISTIQUE **********
        # création de tableaux de features et cibles
        x = global_dataset[['Code postal', 'Nombre pieces principales', 'Surface terrain', 'Surface Carrez du 1er lot', 'Surface Carrez du 2eme lot', 'Surface Carrez du 3eme lot', 'Surface Carrez du 4eme lot', 'Surface Carrez du 5eme lot']]
        y = global_dataset[['Valeur fonciere']]

        # Split du dataset en test et set.
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.20, random_state=0)

        # Mise à l'échelle
        sc=StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        # Entrainement de la Régression Logistique (modèle).
        classifier=LogisticRegression()
        classifier.fit(x_train, y_train)

        # Prédiction sur le test
        y_pred = classifier.predict(x_test)

        # Sélection des valeurs d'une musique à tester.
        x_immo = global_dataset[global_dataset.index==id]
        x_immo = sc.transform(x_immo)
        y_immo = classifier.predict(x_immo)

        # Seulement pour vérifier la valeur.
        valeur_fonciere = x_immo['Valeur fonciere']

        print('Valeur foncière estimée : ', y_immo)
        print('Valeur foncière réelle : ', valeur_fonciere)
        precision = accuracy_score(valeur_fonciere, y_immo)
        print('Précision du test : ', precision)
        # *********** FIN REGRESSION LOGISTIQUE **********

        return y_immo