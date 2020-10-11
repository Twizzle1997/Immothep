import pandas as pd

import sklearn
# from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# from sklearn import metrics
# from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
# from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# import matplotlib.pyplot as plt
from sklearn import utils


class ImmothepPrediction:

    def entrainementLR(self, global_dataset):
        """Entraînement du modèle de regression Logistique.

        Args:
            global_dataset ([dataframe]): [Dataframe d'entraînement]

        Returns:
            [objet]: [Modèle entraîné]
        """
        # *********** DEBUT DE LA PREPARATION DES DONNEES DE REFERENCE *********
        # Vérification des valeurs nulles :
        print('NOMBRE DE VALEURS NULLES :\n', global_dataset.isnull().sum())
        print('*******************')
        # *********** FIN DE LA PREPARATION DES DONNEES DE REFERENCE *********


        # *********** DEBUT REGRESSION LOGISTIQUE **********
        # création de tableaux de features et cibles
        x = global_dataset[['Code postal', 'Nombre pieces principales', 'Surface terrain', 'Surface Carrez du 1er lot', 'Surface Carrez du 2eme lot', 'Surface Carrez du 3eme lot', 'Surface Carrez du 4eme lot', 'Surface Carrez du 5eme lot']][:15000]
        y = global_dataset[['Valeur fonciere']][:15000]

        # Split du dataset en test et set.
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.70, random_state=0)

        # Mise à l'échelle
        sc=StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        # Résolution du bug continuous par encodage des données
        lab_enc = LabelEncoder()
        y_train_encoded = lab_enc.fit_transform(y_train)
        print("y_train_encoded:", y_train_encoded)
        print("utils.multiclass.type_of_target(y_train): ", utils.multiclass.type_of_target(y_train))
        print("utils.multiclass.type_of_target(y_train.astype('int')): ", utils.multiclass.type_of_target(y_train.astype('int')))
        print("utils.multiclass.type_of_target(y_train_encoded): ", utils.multiclass.type_of_target(y_train_encoded))
        print('*******************')
        
        # Entrainement de la Régression Logistique (modèle).
        classifier=LogisticRegression()
        classifier.fit(x_train, y_train_encoded)

        return classifier, sc, lab_enc


    def predictionLR(self, dfLine, modele, scaler, labelEncoder):
        """[summary]

        Args:
            serie (Serie): Série de la valeur à tester
            modele (objet): Modèle entraîné avec entrainementLR()

        Returns:
            valeur prédite: [description]
        """
        sc = scaler
        classifier = modele
        lab_enc = labelEncoder

        # Sélection des valeurs d'un bien immobilier à tester.
        x_immo = dfLine[['Code postal', 'Nombre pieces principales', 'Surface terrain', 'Surface Carrez du 1er lot', 'Surface Carrez du 2eme lot', 'Surface Carrez du 3eme lot', 'Surface Carrez du 4eme lot', 'Surface Carrez du 5eme lot']]
        x_immo = sc.transform(x_immo)
        y_immo = classifier.predict(x_immo)

        # Inverser l'encodage.
        y_immo = lab_enc.inverse_transform(y_immo)

        # Seulement pour vérifier la valeur.
        valeur_fonciere = dfLine['Valeur fonciere']

        print('Valeur foncière estimée : ', y_immo.item())
        print('Valeur foncière réelle : ', valeur_fonciere)


        # precision = accuracy_score(y_train_encoded, y_pred)
        # print('Précision du test : ', precision)
        # *********** FIN REGRESSION LOGISTIQUE **********

        return y_immo







    def predict(self, global_dataset, id):
        """Predict the "Valeur fonciere" column.
        Args:
            global_dataset ([dataframe]): dataframe with all the features
            id ([str]): index in the main dataset.
        """
        # *********** DEBUT DE LA PREPARATION DES DONNEES DE REFERENCE *********
        global_dataset = global_dataset[['Code postal', 'Nombre pieces principales', 'Surface terrain', 'Surface Carrez du 1er lot', 'Surface Carrez du 2eme lot', 'Surface Carrez du 3eme lot', 'Surface Carrez du 4eme lot', 'Surface Carrez du 5eme lot', 'Valeur fonciere']]
        # Vérification des valeurs nulles :
        print('NOMBRE DE VALEURS NULLES :\n',global_dataset.isnull().sum())
        print('*******************')
        # *********** FIN DE LA PREPARATION DES DONNEES DE REFERENCE *********


        # *********** DEBUT REGRESSION LOGISTIQUE **********
        # création de tableaux de features et cibles
        x = global_dataset[['Code postal', 'Nombre pieces principales', 'Surface terrain', 'Surface Carrez du 1er lot', 'Surface Carrez du 2eme lot', 'Surface Carrez du 3eme lot', 'Surface Carrez du 4eme lot', 'Surface Carrez du 5eme lot']][:10000]
        y = global_dataset[['Valeur fonciere']][:10000]

        # Split du dataset en test et set.
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.20, random_state=0)

        # Mise à l'échelle
        sc=StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        # Résolution du bug continuous par encodage des données
        lab_enc = LabelEncoder()
        y_train_encoded = lab_enc.fit_transform(y_train)
        print("y_train_encoded:", y_train_encoded)
        print("utils.multiclass.type_of_target(y_train): ", utils.multiclass.type_of_target(y_train))
        print("utils.multiclass.type_of_target(y_train.astype('int')): ", utils.multiclass.type_of_target(y_train.astype('int')))
        print("utils.multiclass.type_of_target(y_train_encoded): ", utils.multiclass.type_of_target(y_train_encoded))
        print('*******************')
        
        # Entrainement de la Régression Logistique (modèle).
        classifier=LogisticRegression()
        classifier.fit(x_train, y_train_encoded)

        # Prédiction sur le test
        y_pred = classifier.predict(x_test)

        # Sélection des valeurs d'un bien immobilier à tester.
        x_immo = global_dataset[['Code postal', 'Nombre pieces principales', 'Surface terrain', 'Surface Carrez du 1er lot', 'Surface Carrez du 2eme lot', 'Surface Carrez du 3eme lot', 'Surface Carrez du 4eme lot', 'Surface Carrez du 5eme lot']][global_dataset.index==id]
        x_immo = sc.transform(x_immo)
        y_immo = classifier.predict(x_immo)

        # Inverser l'encodage.
        y_immo = lab_enc.inverse_transform(y_immo)

        # Seulement pour vérifier la valeur.
        valeur_fonciere = global_dataset['Valeur fonciere'][global_dataset.index==id]

        print('Valeur foncière estimée : ', y_immo)
        print('Valeur foncière réelle : ', global_dataset.at[id, 'Valeur fonciere'])


        precision = accuracy_score(y_train_encoded, y_pred)
        print('Précision du test : ', precision)
        # *********** FIN REGRESSION LOGISTIQUE **********

        return y_immo






















    def entrainementLR2(self, global_dataset):
        """Entraînement du modèle de regression Logistique.

        Args:
            global_dataset ([dataframe]): [Dataframe d'entraînement]

        Returns:
            [objet]: [Modèle entraîné]
        """
        # *********** DEBUT DE LA PREPARATION DES DONNEES DE REFERENCE *********
        # Vérification des valeurs nulles :
        print('NOMBRE DE VALEURS NULLES :\n', global_dataset.isnull().sum())
        print('*******************')
        # *********** FIN DE LA PREPARATION DES DONNEES DE REFERENCE *********


        # *********** DEBUT REGRESSION LOGISTIQUE **********
        # création de tableaux de features et cibles
        x = global_dataset[['Code postal', 'Nombre pieces principales', 'Surface terrain', 'Surface Carrez du 1er lot', 'Surface Carrez du 2eme lot', 'Surface Carrez du 3eme lot', 'Surface Carrez du 4eme lot', 'Surface Carrez du 5eme lot']][:50000]
        y = global_dataset[['Valeur fonciere']][:50000]

        # Split du dataset en test et set.
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.50, random_state=0)

        # Mise à l'échelle
        sc=StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        # Résolution du bug continuous par encodage des données
        lab_enc = LabelEncoder()
        y_train_encoded = lab_enc.fit_transform(y_train)
        print("y_train_encoded:", y_train_encoded)
        print("utils.multiclass.type_of_target(y_train): ", utils.multiclass.type_of_target(y_train))
        print("utils.multiclass.type_of_target(y_train.astype('int')): ", utils.multiclass.type_of_target(y_train.astype('int')))
        print("utils.multiclass.type_of_target(y_train_encoded): ", utils.multiclass.type_of_target(y_train_encoded))
        print('*******************')
        
        # Entrainement de la Régression Logistique (modèle).
        classifier=LogisticRegression()
        classifier.fit(x_train, y_train_encoded)

        return classifier, sc, lab_enc




    def entrainementLR3(self, global_dataset):
        """Entraînement du modèle de regression Logistique.

        Args:
            global_dataset ([dataframe]): [Dataframe d'entraînement]

        Returns:
            [objet]: [Modèle entraîné]
        """
        # *********** DEBUT DE LA PREPARATION DES DONNEES DE REFERENCE *********
        # Vérification des valeurs nulles :
        print('NOMBRE DE VALEURS NULLES :\n', global_dataset.isnull().sum())
        print('*******************')
        # *********** FIN DE LA PREPARATION DES DONNEES DE REFERENCE *********


        # *********** DEBUT REGRESSION LOGISTIQUE **********
        # création de tableaux de features et cibles
        x = global_dataset[['Code postal', 'Nombre pieces principales', 'Surface terrain', 'Surface Carrez du 1er lot', 'Surface Carrez du 2eme lot', 'Surface Carrez du 3eme lot', 'Surface Carrez du 4eme lot', 'Surface Carrez du 5eme lot']][:70000]
        y = global_dataset[['Valeur fonciere']][:70000]

        # Split du dataset en test et set.
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.70, random_state=0)

        # Mise à l'échelle
        sc=StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        # Résolution du bug continuous par encodage des données
        lab_enc = LabelEncoder()
        y_train_encoded = lab_enc.fit_transform(y_train)
        print("y_train_encoded:", y_train_encoded)
        print("utils.multiclass.type_of_target(y_train): ", utils.multiclass.type_of_target(y_train))
        print("utils.multiclass.type_of_target(y_train.astype('int')): ", utils.multiclass.type_of_target(y_train.astype('int')))
        print("utils.multiclass.type_of_target(y_train_encoded): ", utils.multiclass.type_of_target(y_train_encoded))
        print('*******************')
        
        # Entrainement de la Régression Logistique (modèle).
        classifier=LogisticRegression()
        classifier.fit(x_train, y_train_encoded)

        return classifier, sc, lab_enc