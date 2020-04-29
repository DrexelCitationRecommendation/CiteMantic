import torch
from transformers import *
import json

# Read in claim sentences
claims = ["These findings provide proof-of-concept for the use of the CRG signature as a novel means of drug discovery with relevance to underlying anticancer drug mechanisms."]
file = open('test_labels.json', "r")
data = json.load(file)

for paper in data["papers"]:
    for i in range(len(paper["labels"])):
        if paper["labels"][i] == "1":
            claims.append(paper["sentences"][i])
  
# Build subtoken map
subtokenCounts = {}
tokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")
totalWords = 0
totalSubtokenizedWords = 0
totalSentencesWithSubtokenizedWords = 0
for claim in claims:
    # Get Tokens
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(claim))
    # Seperate into map based on number of subtokens
    subtokenizedWords = 0
    previous = False
    for token in tokens:
        if token.startswith("##") and not previous:
            subtokenizedWords += 1
            previous = True
        else:
            previous = False
        totalWords += 1
    totalSubtokenizedWords += subtokenizedWords
    if subtokenizedWords > 0:
        totalSentencesWithSubtokenizedWords += 1
    else:
        print (claim)
print ((totalSubtokenizedWords / totalWords) * 100, "percent of words were subtokenized")
print ((totalSentencesWithSubtokenizedWords / len(claims)) * 100, "percent of sentences had subtokenized words")
file.close()