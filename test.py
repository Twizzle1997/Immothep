import requests

metre_carre = 740
nb_pieces = 4
terrain = 1000
code_postal = 63000

estimate = 'la fonction qui va faire l\'estimation'
request={"metre_carre" : metre_carre, "nb_pieces" : nb_pieces, "terrain" : terrain, "code_postal" : code_postal}

url = "http://localhost:5003/api/estimate/?metre_carre={0}&nb_pieces={1}&terrain={2}&code_postal={3}".format(metre_carre, nb_pieces, terrain, code_postal)
response = requests.get(url, params = request)
print(response.json())