# Immothep
Création du brief Immothep

### Filtrage du dataset
```python
valeurs2019 = pd.read_csv(cr.PATH + '2019.txt', sep='|', usecols=['Nature mutation', 'Code postal','Nombre pieces principales', 'Surface terrain', 'Surface reelle bati', 'Valeur fonciere', 'Type local', 'Nombre de lots'], encoding='utf-8')

valeurs2019 = valeurs2019.dropna(subset = ['Type local', 'Nombre de lots', 'Nombre pieces principales', 'Nature mutation', 'Surface reelle bati'])

valeurs2019['Surface terrain'][valeurs2019['Type local'].str.contains("Appartement", regex=True)] = valeurs2019['Surface terrain'][valeurs2019['Type local'].str.contains("Appartement", regex=True)].fillna(0) 

valeurs2019 = valeurs2019[['Type local', 'Code postal', 'Nombre pieces principales', 'Surface terrain', 'Surface reelle bati', 'Valeur fonciere']][valeurs2019['Type local'].str.contains("Appartement|Maison", regex=True)][valeurs2019['Nombre de lots']<2][valeurs2019['Nombre pieces principales']>0][valeurs2019['Nature mutation'].str.contains("Vente", regex=True)]

valeurs2019 = valeurs2019.dropna(subset = ['Valeur fonciere'])
valeurs2019 = valeurs2019.dropna(subset = ['Code postal'])

valeurs2019[['Valeur fonciere']] = valeurs2019[['Valeur fonciere']].replace(',', '.', regex=True)

valeurs2019[['Valeur fonciere']]  = valeurs2019[['Valeur fonciere']].astype('float')

valeurs2019 = valeurs2019[valeurs2019['Valeur fonciere']>10000]

valeurs2019 = valeurs2019.dropna(subset = ['Surface terrain'])

maisons2019=valeurs2019[['Code postal', 'Nombre pieces principales', 'Surface terrain', 'Surface reelle bati', 'Valeur fonciere']][valeurs2019['Type local'].str.contains("Maison", regex=True)].reset_index(drop=True)

appart2019=valeurs2019[['Code postal', 'Nombre pieces principales', 'Surface terrain', 'Surface reelle bati', 'Valeur fonciere']][valeurs2019['Type local'].str.contains("Appartement", regex=True)][valeurs2019['Surface reelle bati']>9].reset_index(drop=True)

maisons2019.to_csv(cr.CURATED_LOCAL_PATH + 'filteredmaisons2019.csv', index=False, sep="|")
appart2019.to_csv(cr.CURATED_LOCAL_PATH + 'filteredappart2019.csv', index=False, sep="|")
```
### Export des dataframes
```python
maisons2019 = pd.read_csv(cr.CURATED_LOCAL_PATH + 'filteredmaisons2019.csv', sep="|")
appart2019 = pd.read_csv(cr.CURATED_LOCAL_PATH + 'filteredappart2019.csv', sep="|")
``` 

### Split du dataset pour créer les jeux de données utilisés dans les entraînements de prédiction
```python
splitter.split_datas('filteredmaisons2019.csv', 'Code postal', 'CPMaisons')
splitter.split_datas('filteredappart2019.csv', 'Code postal', 'CPAppart')
```

## Utilisation de l'API
### Lancement du serveur port 5003
Utilisation de la commande ```uvicorn main:app --port 5003``` dans le terminal

### Paramètres :
```metre_carre``` Surface bâtie du bien immobilier 
```nb_pieces``` Nombre de pièces
```terrain``` Surface en m² du terrain
```code_postal``` Code postal...

### URL de l'API
```http://127.0.0.1:5003/api/estimate```