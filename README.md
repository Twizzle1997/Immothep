# Immothep
Création du brief Immothep

## Plan du projet
```credentials.py``` : paramètres de routes  
```prediction.py``` : Méthodes de prédictions et traitement des données  
```main.py``` : Installation et routage de l'API   
```split.py``` : Méthodes de split des données  
```notebook.ipynb``` : Main, notebook du projet


## Mise en place du projet
### Paramètres 
Credentials.py :  
```PATH ``` Route vers le dossier datas et le jeu de données téléchargé  
```CURATED_LOCAL_PATH``` Route vers les fichiers créés  
```DATASET_NAME``` Nom du jeu de données téléchargé  

### Filtrage du dataset
```python
valeurs2019 = pd.read_csv(cr.PATH + DATASET_NAME, sep='|', usecols=['Nature mutation', 'Code postal','Nombre pieces principales', 'Surface terrain', 'Surface reelle bati', 'Valeur fonciere', 'Type local', 'Nombre de lots'], encoding='utf-8')

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
```
### Export des dataframes
```python
maisons2019.to_csv(cr.CURATED_LOCAL_PATH + 'filteredmaisons2019.csv', index=False, sep="|")
appart2019.to_csv(cr.CURATED_LOCAL_PATH + 'filteredappart2019.csv', index=False, sep="|")
``` 

### Split du dataset pour créer les jeux de données utilisés dans les entraînements de prédiction
```python
splitter.split_datas('filteredmaisons2019.csv', 'Code postal', 'CPMaisons')
splitter.split_datas('filteredappart2019.csv', 'Code postal', 'CPAppart')
```

## Utilisation de l'API
### Lancement du serveur port 5003
```uvicorn main:app --port 5003```

### Paramètres :
```metre_carre``` Surface bâtie du bien immobilier  
```nb_pieces``` Nombre de pièces  
```terrain``` Surface en m² du terrain  
```code_postal``` Code postal...

### URL de l'API
```http://127.0.0.1:5003/api/estimate```

### Exemple d'utilisation
```request={"metre_carre" : metre_carre, "nb_pieces" : nb_pieces, "terrain" : terrain, "code_postal" : code_postal}

url = "http://127.0.0.1:5003/api/estimate"
response = requests.get(url, params = request).json()
print(response)```
