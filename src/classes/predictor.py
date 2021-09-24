from sklearn.metrics import mean_squared_error, explained_variance_score, max_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
from classes.credentials import Credentials as cr

class Predictor():

    def train_test_split(self, type_local, sep):
        """ Split the global dataset into features and labels train and test sets"""

        # Passage en minuscules
        type_local.lower()

        # Sélection du type de local
        if type_local == 'maison':
            df = pd.read_csv(cr.TRAIN_PATH + 'Maison.csv', sep=sep)

        elif type_local == 'appartement':
            df = pd.read_csv(cr.TRAIN_PATH + 'Appartement.csv', sep=sep)

        else:
            print("Sélectionnez type local = maison ou appartement")
            return 0

        features = df[['Code postal', 'Nombre pieces principales', 'Surface terrain', 'Surface reelle bati', 'Nombre de lots']]
        labels = df['Valeur fonciere']

        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 1)

        return train_features, test_features, train_labels, test_labels

        # return df


    def train(self, x_train, y_train):
        """train model"""

        self.model.fit(x_train, y_train)


    def predict(self, data_test):
        """predict values on test set"""

        data_test["Valeur fonciere estimee"] = self.model.predict(data_test).round()
    

    def get_metrics(self, x_test, y_test):
        """get metrics of the trained model"""

        y_pred = self.model.predict(x_test)

        print("Mean squared error: ", mean_squared_error(y_test, y_pred),
        "\nVariance regression score function: ", explained_variance_score(y_test, y_pred, multioutput='uniform_average'),
        "\nMaximum residual error: ", max_error(y_test, y_pred), 
        "\nMean absolute error regression loss: ", mean_absolute_error(y_test, y_pred, multioutput='uniform_average'))


    def predict_value(self, metre_carre, nb_pieces, terrain, lots, CP):
        """Predict land value using the trained model"""
        
        df = pd.DataFrame({'Surface reelle bati' : [metre_carre], 'Nombre pieces principales' : [nb_pieces], 'Surface terrain' : [terrain], 'Nombre de lots' : [lots], 'Code postal' : [CP]})

        predicted_value = self.model.predict(df).round()

        return predicted_value