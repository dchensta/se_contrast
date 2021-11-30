import spacy

verbs = {}
nlp = spacy.load("en_core_web_lg")

yeonjun = open("problem_clauses.txt", "r")
lines = yeonjun.readlines()

for line in lines :
    if type(line) != str :
        continue
    words = line.split()
    if len(words) < 1 :
        continue
    if (words[0] == "Problem" and words[1] == "clauses") or line == "\n" :
        continue

    doc = nlp(line)
    for token in doc :
        if token.dep_ == "ROOT" :
            v = token.lemma_
            if verbs.get(v) == None : #Create an entry
                verbs[v] = 1
            else :
                verbs[v] += 1

for key, value in verbs.items() :
    print(f"{key}: {value}")