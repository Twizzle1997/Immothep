import requests

metre_carre = 740
nb_pieces = 4
terrain = 1000
code_postal = 63000

estimate = 'la fonction qui va faire l\'estimation'
request={"metre_carre" : metre_carre, "nb_pieces" : nb_pieces, "terrain" : terrain, "code_postal" : code_postal}

url = "http://127.0.0.1:5003/api/estimate"
response = requests.get(url, params = request)
print(response.json())