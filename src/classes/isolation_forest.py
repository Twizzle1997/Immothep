import pandas as pd
from sklearn.ensemble import IsolationForest

class IsolationForestClass:
    
    def isolation_forest_testing(self, in_file, out_file, sep):
        df = pd.read_csv(in_file, sep=sep, encoding='utf-8')

        # Isolation forets
        model = IsolationForest(n_estimators = 100, max_samples = 'auto', contamination = 'auto', max_features = 1)
        model.fit(df[['Code postal', 'Nombre pieces principales', 'Surface terrain', 'Surface reelle bati', 'Nombre de lots', 'Valeur fonciere']])

        df['scores'] = model.decision_function(df[['Code postal', 'Nombre pieces principales', 'Surface terrain', 'Surface reelle bati', 'Nombre de lots', 'Valeur fonciere']])
        df['anomaly'] = model.predict(df[['Code postal', 'Nombre pieces principales', 'Surface terrain', 'Surface reelle bati', 'Nombre de lots', 'Valeur fonciere']])

        print(df.head(10))

        df.to_csv(out_file)

    def isolation_forest(self, file, sep):
        df = pd.read_csv(file, sep=sep, encoding='utf-8')

        # Suppression des NaN
        df = df.dropna()

        # Isolation forets
        model = IsolationForest(n_estimators = 100, max_samples = 'auto', contamination = 'auto', max_features = 1)
        model.fit(df[['Code postal', 'Nombre pieces principales', 'Surface terrain', 'Surface reelle bati', 'Nombre de lots', 'Valeur fonciere']])

        # Ajout scores et anomalies
        df['scores'] = model.decision_function(df[['Code postal', 'Nombre pieces principales', 'Surface terrain', 'Surface reelle bati', 'Nombre de lots', 'Valeur fonciere']])
        df['anomaly'] = model.predict(df[['Code postal', 'Nombre pieces principales', 'Surface terrain', 'Surface reelle bati', 'Nombre de lots', 'Valeur fonciere']])

        # Suppression des anomalies
        df = df[['Code postal', 'Nombre pieces principales', 'Surface terrain', 'Surface reelle bati', 'Nombre de lots', 'Valeur fonciere']][df["anomaly"] != -1]

        # Retrait des 10% de valeurs foncières les plus hautes et 10% des valeurs foncières les plus basses
        df = df[df['Valeur fonciere'].between(df['Valeur fonciere'].quantile(.10), df['Valeur fonciere'].quantile(.90))]

        # Réécriture du csv
        df.to_csv(file)