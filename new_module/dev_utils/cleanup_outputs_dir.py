## 폴더 정리

import shutil
import os
import glob
import pandas as pd

# ------ toxicity ------ # 

### rename folder to wandb id
# for path in glob.glob('/data/hyeryung/mucoco/outputs/toxicity/**/**/'):
#     print(path)
    
#     if len(path.split('/')[-2]) != 8:
#         wandb_id = path.split('/')[-2].split('-')[-1]
#         new_path = '/'.join(path.split('/')[:-2] + [wandb_id])
#         print(new_path)
#         os.rename(path, new_path)

### check outputs length and delete folders with strange lengths
# for path in glob.glob('/data/hyeryung/mucoco/outputs/toxicity/**/**/outputs*.txt'):
#     print(path)
    
#     try:
#         outputs = pd.read_json(path, lines=True)
#         outputs = outputs.explode('generations')
#         outputs = outputs.dropna()
            
#         if len(outputs) == 2500:
#             print('test set')
#             if path.split('/')[6] == 'devset':
#                 new_path = '/'.join(path.split('/')[:6] + ['final'] + path.split('/')[7:])
#                 print(f'!!! change path to {os.path.dirname(new_path)}')
#                 # os.rename(os.path.dirname(path), os.path.dirname(new_path))
#         elif len(outputs) == 604:
#             print('dev set')
#         else:
#             print(f'??? {len(outputs)}')
#             if len(outputs) < 50:
#                 print(f'!!!! delete this directory {os.path.dirname(path)}')
#                 # shutil.rmtree(os.path.dirname(path))
                
#             elif len(outputs) == 581: ## devset & skipped -> not acceptable
#                 print(f'!!!! delete this directory {os.path.dirname(path)}')
#                 # shutil.rmtree(os.path.dirname(path))
#     except:
#         try:
#             outputs = pd.read_json(path, lines=True)
#             # if len(outputs) == 0:
#             #     print(f'!!!! delete this directory {os.path.dirname(path)}')
#             #     shutil.rmtree(os.path.dirname(path))
#         except Exception as e:
#             print(f"!!!!! {e}")
#             # print(f'!!!! delete this directory {os.path.dirname(path)}')
#             # shutil.rmtree(os.path.dirname(path))



### manual deletion
# (loc-edit) (base) hyeryung@master:/data/hyeryung/mucoco$ rm -rf /data/hyeryung/mucoco/outputs/toxicity/devset/zsm80jfz
# (loc-edit) (base) hyeryung@master:/data/hyeryung/mucoco$ rm -rf /data/hyeryung/mucoco/outputs/toxicity/devset/5x65yj1u
# (loc-edit) (base) hyeryung@master:/data/hyeryung/mucoco$ rm -rf /data/hyeryung/mucoco/outputs/toxicity/devset/h39o0pa2

# ------ sentiment ------ # 
### rename folder to wandb id
# for path in glob.glob('/data/hyeryung/mucoco/outputs/sentiment/**/**/'):
#     print(path)
    
#     if len(path.split('/')[-2]) != 8:
#         wandb_id = path.split('/')[-2].split('-')[-1]
#         new_path = '/'.join(path.split('/')[:-2] + [wandb_id])
#         print(new_path)
#         # os.rename(path, new_path)

# ### check outputs length and delete folders with strange lengths
# for path in glob.glob('/data/hyeryung/mucoco/outputs/sentiment/**/**/outputs*[0-9].txt'):
# # for path in glob.glob('/data/hyeryung/mucoco/outputs/sentiment/**/**/**/outputs*[0-9].txt'):
#     print(path)
    
#     try:
#         outputs = pd.read_json(path, lines=True)
#         outputs = outputs.explode('generations')
#         outputs = outputs.dropna()
            
#         if len(outputs) == 300:
#             print('previous test set')
#         elif len(outputs) == 900:
#             print('test set')
#             if path.split('/')[6] == 'devset':
#                 new_path = '/'.join(path.split('/')[:6] + ['final'] + path.split('/')[7:])
#                 print(f'!!! change path to {os.path.dirname(new_path)}')
#                 # os.rename(os.path.dirname(path), os.path.dirname(new_path))
#         else:
#             print(f'??? {len(outputs)}')
#             if len(outputs) < 50:
#                 print(f'!!!! delete this directory {os.path.dirname(path)}')
#                 # shutil.rmtree(os.path.dirname(path))
                
#     except:
#         try:
#             outputs = pd.read_json(path, lines=True)
#             if len(outputs) == 0:
#                 print(len(outputs))
#                 print(f'!!!! delete this directory {os.path.dirname(path)}')
#                 # shutil.rmtree(os.path.dirname(path))
#         except Exception as e:
#             print(f"!!!!! {e}")
#             print(f'!!!! delete this directory {os.path.dirname(path)}')
#             # shutil.rmtree(os.path.dirname(path))
  
### manual deletion
# (loc-edit) (base) hyeryung@master:/data/hyeryung/mucoco$ rm -rf /data/hyeryung/mucoco/outputs/sentiment/mucola/vqteup8r
# (loc-edit) (base) hyeryung@master:/data/hyeryung/mucoco$ rm -rf /data/hyeryung/mucoco/outputs/sentiment/final/mmbcjvlo
# (loc-edit) (base) hyeryung@master:/data/hyeryung/mucoco$ rm -rf /data/hyeryung/mucoco/outputs/sentiment/final/xhvtb06k
# (loc-edit) (base) hyeryung@master:/data/hyeryung/mucoco$ rm -rf /data/hyeryung/mucoco/outputs/sentiment/final/wi0n3ebk
# (loc-edit) (base) hyeryung@master:/data/hyeryung/mucoco$ rm -rf /data/hyeryung/mucoco/outputs/sentiment/final/64hldp6d

# ------ formality ------ # 
### rename folder to wandb id
# for path in glob.glob('/data/hyeryung/mucoco/outputs/formality/**/**/'):
#     print(path)
    
#     if len(path.split('/')[-2]) != 8:
#         wandb_id = path.split('/')[-2].split('-')[-1]
#         new_path = '/'.join(path.split('/')[:-2] + [wandb_id])
#         print(new_path)
#         # os.rename(path, new_path)

# for path in glob.glob('/data/hyeryung/mucoco/outputs/formality/**/rescale/**/'):
#     print(path)
    
#     if len(path.split('/')[-2]) != 8:
#         wandb_id = path.split('/')[-2].split('-')[-4]
#         new_path = '/'.join(path.split('/')[:-3] + [wandb_id])
#         print(new_path)
#         # os.rename(path, new_path)


# ### check outputs length and delete folders with strange lengths
for path in glob.glob('/data/hyeryung/mucoco/outputs/formality/**/**/outputs*[0-9].txt'):
# for path in glob.glob('/data/hyeryung/mucoco/outputs/formality/mlm-reranking/**/outputs*[0-9].txt'):
    print(path)
    
    try:
        outputs = pd.read_json(path, lines=True)
        outputs = outputs.explode('generations')
        outputs = outputs.dropna()
            
        if (len(outputs) == 1416) or (len(outputs) == 1082):
            print('test set')
            if path.split('/')[6] == 'devset':
                new_path = '/'.join(path.split('/')[:6] + ['final'] + path.split('/')[7:])
                print(f'!!! change path to {os.path.dirname(new_path)}')
                # os.rename(os.path.dirname(path), os.path.dirname(new_path))
        else:
            print(f'??? {len(outputs)}')
            if len(outputs) < 200:
                print(f'!!!! delete this directory {os.path.dirname(path)}')
                # shutil.rmtree(os.path.dirname(path))
                
    except:
        try:
            outputs = pd.read_json(path, lines=True)
            if len(outputs) == 0:
                print(len(outputs))
                print(f'!!!! delete this directory {os.path.dirname(path)}')
                # shutil.rmtree(os.path.dirname(path))
        except Exception as e:
            print(f"!!!!! {e}")
            print(f'!!!! delete this directory {os.path.dirname(path)}')
            # shutil.rmtree(os.path.dirname(path))
  
### manual deletion
# (loc-edit) (base) hyeryung@master:/data/hyeryung/mucoco$ rm -rf /data/hyeryung/mucoco/outputs/formality/final/p7m1vo04