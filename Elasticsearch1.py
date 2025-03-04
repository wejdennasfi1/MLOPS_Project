from elasticsearch import Elasticsearch

# Connexion à Elasticsearch
es = Elasticsearch([{"host": "localhost", "port": 9200}])


def log_to_elasticsearch(index, doc_type, data):
    """Fonction pour envoyer les logs vers Elasticsearch"""
    es.index(index=index, doc_type=doc_type, body=data)
