from glob import glob

in_filepaths = glob("/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/formal.ref*")
# in_filepaths = glob("/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal.ref*")


print(f"in_filepaths: {in_filepaths}")

out_fp = open("/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/formal.ref.merged", "w")
# out_fp = open("/data/hyeryung/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal.ref.merged", "w")

print(f"out_fp: {out_fp}")

for path in in_filepaths:
    
    with open(path, 'r') as f:
        lines = f.readlines()
        
    out_fp.writelines(lines)

out_fp.close()