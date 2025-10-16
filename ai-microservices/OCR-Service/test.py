import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("John Doe is a student.")
for ent in doc.ents:
    print(ent.text, ent.label_)
# to check if spacy is working fine