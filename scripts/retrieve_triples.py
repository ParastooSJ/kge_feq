import tagme
tagme.GCUBE_TOKEN = "e699faf6-94d2-4739-8d7c-097b7cd1be83-843339462"
import pandas as pd
import json
from difflib import SequenceMatcher
from elasticsearch import Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
current_directory = '/data/dbpedia-v2/'



def get_factoid_triples (question):
    triples = []
    triple_size = 1000
  
    et_results = et_results = es.search(index="wiki-graph-index", body={
    "size": triple_size,
    "query": {
        "query_string": {
            "query": question,  # The search query as a string
            "default_field": "text"  # The field to search in
        }
    }
    })['hits']['hits']

    for entry in et_results:
       
        sentence = entry["_source"]["text"]
        results = es.search(index="wiki-graph-index", body={
        "size": triple_size,
        "query": {
            "match": {
                "text": sentence  # This is the exact text to search for
            }
        }
        })['hits']['hits']

        for et in results: 
            source = et["_source"]["title"]
            sentence = et["_source"]["text"]
            anchored_et = et["_source"]["anchored_et"]
            triple = {'subject':source,'object':anchored_et,'relation':sentence}
            triples.append(triple)
         
    return triples


def create_dataset(source_f,dest_f):
  while True:
    line = source_f.readline()
    if not line:
      break
    try:
    
        question = line.split("\t")[1].strip()
        answer = line.split("\t")[2].strip()
        index = line.split("\t")[0].strip()
        triple_result = {'index':index,'question':question, 'triples':get_factoid_triples(question),'answer':answer}
        
        
        dest_f.write(json.dumps(triple_result)+"\n")
        
    except Exception as e:
        
        print(e)
        
  source_f.close()
  dest_f.close()
  
source_path = current_directory + 'data/input.txt'
dest_path = current_directory +'data/output.json'
source_f = open(source_path,'r')
dest_f = open(dest_path,'w')
create_dataset(source_f, dest_f)




















