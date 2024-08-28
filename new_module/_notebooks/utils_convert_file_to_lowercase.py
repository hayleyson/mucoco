with open('/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/formal', 'r') as f:
    data = f.readlines()

data = [x.lower() for x in data]

with open('/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/formal_lower', 'w') as f:
    f.writelines(data)
    
    
with open('/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal', 'r') as f:
    data = f.readlines()

data = [x.lower() for x in data]

with open('/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal_lower', 'w') as f:
    f.writelines(data)