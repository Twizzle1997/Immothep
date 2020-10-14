import pandas as pd

import sklearn
# from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn import metrics
# from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
# from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import utils
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns


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
        x = global_dataset[['Surface reelle bati', 'Nombre pieces principales', 'Surface terrain']]
        y = global_dataset[['Valeur fonciere']]

        # Split du dataset en test et set.
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.70, random_state=0)

        # Mise à l'échelle
        sc=StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        # Résolution du bug continuous par encodage des données
        # lab_enc = LabelEncoder()
        # y_train_encoded = lab_enc.fit_transform(y_train)
        # print("y_train_encoded:", y_train_encoded)
        # print("utils.multiclass.type_of_target(y_train): ", utils.multiclass.type_of_target(y_train))
        # print("utils.multiclass.type_of_target(y_train.astype('int')): ", utils.multiclass.type_of_target(y_train.astype('int')))
        # print("utils.multiclass.type_of_target(y_train_encoded): ", utils.multiclass.type_of_target(y_train_encoded))
        # print('*******************')
        
        # Entrainement de la Régression Logistique (modèle).
        classifier=LogisticRegression()
        classifier.fit(x_train, y_train) #y_train_encoded

        return classifier, sc#, lab_enc


    def predictionLR(self, dfLine, modele, scaler):#labelEncoder
        """[summary]

        Args:
            serie (Serie): Série de la valeur à tester
            modele (objet): Modèle entraîné avec entrainementLR()

        Returns:
            valeur prédite: [description]
        """
        sc = scaler
        classifier = modele
        # lab_enc = labelEncoder

        # Sélection des valeurs d'un bien immobilier à tester.
        x_immo = dfLine[['Surface reelle bati', 'Nombre pieces principales', 'Surface terrain']]
        x_immo = sc.transform(x_immo)
        y_immo = classifier.predict(x_immo)

        # Inverser l'encodage.
        # y_immo = lab_enc.inverse_transform(y_immo)

        # Seulement pour vérifier la valeur.
        valeur_fonciere = dfLine['Valeur fonciere']

        print('Valeur foncière estimée : ', y_immo.item())
        print('Valeur foncière réelle : ', valeur_fonciere)

        # precision = accuracy_score(valeur_fonciere, y_immo)
        # print('Précision du test : ', precision)
        # *********** FIN REGRESSION LOGISTIQUE **********

        return y_immo








    def entrainementLinear(self, global_dataset):
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


        # *********** DEBUT REGRESSION LINEAIRE **********
        # création de tableaux de features et cibles
        x = global_dataset[['Surface reelle bati', 'Code postal', 'Nombre pieces principales', 'Surface terrain']][:10000]
        y = global_dataset[['Valeur fonciere']][:10000]

        # Entrainement à la régression linéaire
        reg = LinearRegression().fit(x, y)
        print('score : ', reg.score(x, y))

        # Mise à l'échelle
        # sc=StandardScaler()
        # x_train = sc.fit_transform(x_train)
        # x_test = sc.transform(x_test)

        # Résolution du bug continuous par encodage des données
        # lab_enc = LabelEncoder()
        # y_train_encoded = lab_enc.fit_transform(y_train)
        # print("y_train_encoded:", y_train_encoded)
        # print("utils.multiclass.type_of_target(y_train): ", utils.multiclass.type_of_target(y_train))
        # print("utils.multiclass.type_of_target(y_train.astype('int')): ", utils.multiclass.type_of_target(y_train.astype('int')))
        # print("utils.multiclass.type_of_target(y_train_encoded): ", utils.multiclass.type_of_target(y_train_encoded))
        # print('*******************')
        
        # Entrainement de la Régression Logistique (modèle).
        # classifier=LogisticRegression()
        # classifier.fit(x_train, y_train_encoded)

        return reg


    def predictionLinear(self, dfLine, modele):
        """[summary]

        Args:
            serie (Serie): Série de la valeur à tester
            modele (objet): Modèle entraîné avec entrainementLR()

        Returns:
            valeur prédite: [description]
        """
        classifier = modele

        # Sélection des valeurs d'un bien immobilier à tester.
        x_immo = dfLine[['Surface reelle bati', 'Nombre pieces principales', 'Surface terrain']]
        y_immo = classifier.predict(x_immo)


        # Seulement pour vérifier la valeur.
        valeur_fonciere = dfLine['Valeur fonciere']

        print('Valeur foncière estimée : ', y_immo.item())
        print('Valeur foncière réelle : ', valeur_fonciere)


        # precision = accuracy_score(y_train_encoded, y_pred)
        # print('Précision du test : ', precision)
        # *********** FIN REGRESSION LOGISTIQUE **********

        return y_immo




    def trainLinearLeRetour(self, global_dataset):

        x = global_dataset[['Surface reelle bati', 'Nombre pieces principales', 'Surface terrain']]
        y = global_dataset[['Valeur fonciere']]

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.70, random_state=0)

        # Create linear regression object
        regr = LinearRegression()

        # Train the model using the training sets
        regr.fit(x_train, y_train)

        # Make predictions using the testing set
        y_pred = regr.predict(x_test)

        # The coefficients
        print('Coefficients: \n', regr.coef_)

        # The mean squared error
        print('Mean squared error: %.2f'
            % mean_squared_error(y_test, y_pred))
        # The coefficient of determination: 1 is perfect prediction
        print('Coefficient of determination: %.2f'
            % r2_score(y_test, y_pred))

        # Plot outputs
        # plt.scatter(x_test, y_test,  color='black')
        # plt.plot(x_test, y_pred, color='blue', linewidth=3)

        # plt.xticks(())
        # plt.yticks(())

        # plt.show()

        return regr


    def isolationforest (self, datas):
        
        df1 = datas

        model = IsolationForest(n_estimators = 100, max_samples = 'auto', contamination = 'auto', max_features = 1)
        model.fit(df1[['Valeur fonciere']])


        df1['scores'] = model.decision_function(df1[['Valeur fonciere']])
        df1['anomaly'] = model.predict(df1[['Valeur fonciere']])


        anomaly=df1.loc[df1['anomaly'] == -1]
        anomaly_index = list(anomaly.index)

        return anomaly