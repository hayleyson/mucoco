import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import click

def toxicity_score(generations_df, response_file, proba_file, perspective_rate_limit=5):
    
    #--  define client
    from googleapiclient import discovery
    from googleapiclient.errors import HttpError
    import time

    API_KEY="AIzaSyDjyaS-Iyw0nOjDjgTi645taUlp13EAs2k" ## hayley - 23/03/23
    assert API_KEY != "none", "Please set the API_KEY before proceeding"

    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )
    # -- #
    
    # fout = open(response_file, "w")
    fout_prob = open(proba_file, "w")

    last_request_time = -1
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating toxicity via perspective API'):
        
        allresponses = []
        genid = 0
        
        generations = [row['text']]  # - hayley
        generations = [gen if gen != "" else "\n" for gen in generations]

        responses = {f'gen-{i}-{genid}-{j}': None for j in range(len(generations))}
        
        not_done = np.array([1.0/len(generations) for gen in generations])
        def response_callback(request_id, response, exception):
            responses[request_id] = (response, exception)
            if exception is None:
                not_done[int(request_id.split("-")[-1])] = 0
            if exception is not None:
                not_done[int(request_id.split("-")[-1])] = 1
                print(request_id, exception)
        
        while not_done.sum() > 0:
            try:
                time_since_last_request = time.time() - last_request_time

                if time_since_last_request < 1:
                    time.sleep(1-time_since_last_request)

                if not_done.sum() > 1:
                    print(i, "extra_sleep", not_done.sum(), flush=True)
                    time.sleep(1.0)

                batch_request = client.new_batch_http_request()
                for j, text in enumerate(generations):
                    analyze_request= {
                        'comment': {'text': text},
                        'requestedAttributes': {"TOXICITY":{}},
                        'spanAnnotations': True,
                        "languages": ["en"],
                    }
                    batch_request.add(client.comments().analyze(body=analyze_request), callback=response_callback, request_id=f"gen-{i}-{genid}-{j}")
                batch_request.execute()
                last_request_time = time.time()
            except Exception as e:
                print(e)
                print("sleeping for 60 sec and retrying")
                time.sleep(60.0)
        allresponses.append(responses)

        # json.dump({"allresponses": responses}, fout)
        # fout.write("\n")
        # fout.close()
        # fout = open(response_file, "a")
        
        # print(f'responses: {responses}')
        for req_id, (response, exception) in responses.items():    
            prob = response['attributeScores']['TOXICITY']['spanScores'][0]['score']['value']
        fout_prob.write(f"{i}, {prob}\n")
        fout_prob.close()
        fout_prob = open(proba_file, "a")
        
    return 1

@click.command()
@click.option('--dpath', help='Path to jigsaw data')
@click.option('--nsamples', type=int, help='Number of samples to evaluate')
@click.option('--rseed', default=999, type=int, help='Random seed to sample')
@click.option('--outpath', help='Path to save probability result')
def main(dpath, nsamples, outpath, rseed):
    dat_file = pd.read_json(dpath, lines=True)
    dat_file = dat_file.sample(nsamples,random_state=rseed)
    print('dat_file.shape: ', dat_file.shape)
    dat_file.reset_index().to_json(dpath.split('.jsonl')[0] + f'_{nsamples}.jsonl', orient='records', lines=True)
    print('dat_file saved!')
    toxicity_score(dat_file, '', outpath)
    
if __name__ == "__main__":
    
    main()
    