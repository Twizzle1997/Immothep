import pandas as pd
from sklearn.ensemble import IsolationForest

class IsolationForestClass:
    
    def isolation_forest_testing(self, in_file, out_file):
        df = pd.read_csv(in_file, sep='|', encoding='utf-8')

        # Suppression des NaN
        df = df.dropna()

        # Isolation forets
        model = IsolationForest(n_estimators = 100, max_samples = 'auto', contamination = 'auto', max_features = 1)
        model.fit(df[['Nombre pieces principales', 'Surface terrain', 'Surface reelle bati', 'Valeur fonciere']])

        df['scores'] = model.decision_function(df[['Nombre pieces principales', 'Surface terrain', 'Surface reelle bati', 'Valeur fonciere']])
        df['anomaly'] = model.predict(df[['Nombre pieces principales', 'Surface terrain', 'Surface reelle bati', 'Valeur fonciere']])

        print(df.head(10))

        df.to_csv(out_file)

    def isolation_forest(self, file):
        df = pd.read_csv(file)
        model = IsolationForest(n_estimators = 100, max_samples = 'auto', contamination = 'auto', max_features = 1)
        model.fit(df[['Nombre pieces principales', 'Surface terrain', 'Surface reelle bati', 'Valeur fonciere']])

        df['scores'] = model.decision_function(df[['Nombre pieces principales', 'Surface terrain', 'Surface reelle bati', 'Valeur fonciere']])
        df['anomaly'] = model.predict(df[['Nombre pieces principales', 'Surface terrain', 'Surface reelle bati', 'Valeur fonciere']])

        df = df[['Nombre pieces principales', 'Surface terrain', 'Surface reelle bati', 'Valeur fonciere']][df["anomaly"] != -1]

        df.to_csv(file)