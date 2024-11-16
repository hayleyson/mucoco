# with open("/data/hyeryung/mucoco/new_module/data/toxicity-avoidance/dev_set.jsonl", "r") as f:
#     data = f.readlines()
    
# data = data[::-1]

# with open("/data/hyeryung/mucoco/new_module/data/toxicity-avoidance/dev_set_r.jsonl", "w") as f:
#     f.writelines(data)

with open("/data/hyeryung/mucoco/new_module/data/toxicity-avoidance/testset_gpt2_2500.jsonl", "r") as f:
    data = f.readlines()
    
data = data[::-1]

with open("/data/hyeryung/mucoco/new_module/data/toxicity-avoidance/testset_gpt2_2500_r.jsonl", "w") as f:
    f.writelines(data)