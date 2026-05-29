# with open("/home/hyeryung/data/mucoco/laser_edit/data/toxicity-avoidance/dev_set.jsonl", "r") as f:
#     data = f.readlines()
    
# data = data[::-1]

# with open("/home/hyeryung/data/mucoco/laser_edit/data/toxicity-avoidance/dev_set_r.jsonl", "w") as f:
#     f.writelines(data)

with open("/home/hyeryung/data/mucoco/laser_edit/data/toxicity-avoidance/testset_gpt2_2500.jsonl", "r") as f:
    data = f.readlines()
    
data = data[::-1]

with open("/home/hyeryung/data/mucoco/laser_edit/data/toxicity-avoidance/testset_gpt2_2500_r.jsonl", "w") as f:
    f.writelines(data)