import pandas as pd

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import utils
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
from credentials import Credentials as cr

class ImmothepPrediction:

    def trainLinearLeRetourAPIAppart(self, code_postal):
        cp=str(code_postal)[:-3]
        global_dataset=pd.read_csv(cr.CURATED_LOCAL_PATH+'CPAppart/' + cp + '.csv')
        global_dataset=global_dataset.dropna()
        x = global_dataset[['Surface reelle bati', 'Nombre pieces principales', 'Surface terrain']]
        y = global_dataset[['Valeur fonciere']]

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.70, random_state=0)

        # Create linear regression object
        regr = LinearRegression()

        # Train the model using the training sets
        regr.fit(x_train, y_train)

        # Make predictions using the testing set
        y_pred = regr.predict(x_test)
        return regr

    def trainLinearLeRetourAPIMaison(self, code_postal):
        cp=str(code_postal)[:-3]
        global_dataset=pd.read_csv(cr.CURATED_LOCAL_PATH+'CPMaisons/' + cp + '.csv')
        global_dataset=global_dataset.dropna()
        x = global_dataset[['Surface reelle bati', 'Nombre pieces principales', 'Surface terrain']]
        y = global_dataset[['Valeur fonciere']]

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.70, random_state=0)

        # Create linear regression object
        regr = LinearRegression()

        # Train the model using the training sets
        regr.fit(x_train, y_train)

        # Make predictions using the testing set
        y_pred = regr.predict(x_test)
        return regr


    def predictionLinearAPI(self, nb_pieces, metre_carre, terrain, modele):
        """[summary]

        Args:
            serie (Serie): Série de la valeur à tester
            modele (objet): Modèle entraîné avec entrainementLR()

        Returns:
            valeur prédite: [description]
        """
        classifier = modele

        # Sélection des valeurs d'un bien immobilier à tester.
        x_immo = pd.DataFrame({'Surface reelle bati' : [metre_carre], 'Nombre pieces principales' : [nb_pieces], 'Surface terrain' : [terrain]})
        y_immo = classifier.predict(x_immo)
        # *********** FIN REGRESSION LINEAIRE **********

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

        return regr

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

        print('Valeur foncière estimée : ', np.around(y_immo.item(), decimals=2))
        print('Valeur foncière réelle : ', valeur_fonciere)
        # *********** FIN REGRESSION LINEAIRE **********

        return y_immo


    def isolationforest (self, datas):
        
        df1 = datas

        model = IsolationForest(n_estimators = 100, max_samples = 'auto', contamination = 'auto', max_features = 1)
        model.fit(df1[['Valeur fonciere']])


        df1['scores'] = model.decision_function(df1[['Valeur fonciere']])
        df1['anomaly'] = model.predict(df1[['Valeur fonciere']])


        anomaly=df1.loc[df1['anomaly'] == -1]
        anomaly_index = list(anomaly.index)

        return anomaly