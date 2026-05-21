import pandas as pd
import json 
from pathlib import Path

NUM_EXAMPLES = 5
RANDOM_SEED = 44 # Dont' change the random seed; otherwise the match with gpt-written hypotheses/continuations will break.
ROOT_DIR = Path("new_module/base_lm_generate")

few_shot_prompts = {
    "nli": {
        
    },
    "nontoxic": {
        
    },
    "nli+nontoxic": {
        
    }
}

## NLI -> random sample 5 neutral or entailment examples from ANLI training set
nli_dataset = pd.read_json('/home/hyeryung/data/mucoco/data/nli/snli_mnli_anli_train_without_finegrained.jsonl', lines=True)
anli = nli_dataset.loc[nli_dataset['source'] == 'anli_R2_train'].copy()
cons = anli.loc[anli['binary_labels']==1].copy()
sample = cons.sample(n=NUM_EXAMPLES, random_state=RANDOM_SEED)
gpt_written_hypotheses = {
  "1": "Heman's family includes at least two Academy Award winners, with both Roger Heman and his father receiving Oscars for their work in film sound.",
  "2": "Avani Modi has acted in Indian films, including movies that were scheduled for release before the end of the year 2015.",
  "3": "Most people in Germany would choose the Bavarian Mountain Hound as their preferred breed over many other kinds of dogs.",
  "4": "\"Erotique\" was one of the short films directed by Polish filmmaker Jerzy Skolimowski before he made his first feature film.",
  "5": "The Los Angeles Chargers are the same professional football franchise that previously played under the name San Diego Chargers."
}

few_shot_prompts["nli"] = {
    str(i+1): {
        "premise": sample.iloc[i]['premise'],
        "hypothesis": gpt_written_hypotheses[str(i+1)]
    } for i in range(NUM_EXAMPLES)
}


## Toxicity -> random sample 5 sentences with high continuation toxicity and rewrite with ChatGPT to make the continuation nontoxic.
rtp = pd.read_json("/home/hyeryung/data/mucoco/data/control-prompts/nontoxic_prompts-10k.jsonl", lines=True)
rtp['prompt_text'] = rtp['prompt'].apply(lambda x: x['text'])

dev_set = pd.read_json("/home/hyeryung/data/mucoco/data/control-prompts/nontoxic_prompts-250.jsonl", lines=True)
dev_set['prompt_text'] = dev_set['prompt'].apply(lambda x: x['text'])

rtp = rtp.loc[~rtp['prompt_text'].isin(dev_set['prompt_text'])].copy()
rtp['prompt_toxicity'] = rtp['prompt'].apply(lambda x: x['toxicity'])
rtp['continuation_text'] = rtp['continuation'].apply(lambda x: x['text'])
rtp['continuation_toxicity'] = rtp['continuation'].apply(lambda x: x['toxicity'])
rtp['sentence_text'] = (rtp['prompt_text'] + rtp['continuation_text']).str.strip()

rtp_toxic = rtp.loc[rtp['continuation_toxicity'].between(0.5, 0.8, inclusive="right")]
rtp_toxic = rtp_toxic.loc[rtp_toxic['continuation_text'].apply(lambda x: len(x)) >= 120].copy()
rtp_toxic = rtp_toxic.sample(n=NUM_EXAMPLES, random_state=RANDOM_SEED)

gpt_written_continuations = {
    "1": " even though the image depends on a selective reading of his public persona and political style. A more careful view would separate the entertainment value of his rallies and interviews from the harder questions about policy, power, wealth, and accountability. Supporters may see him as a blunt outsider speaking for people who feel ignored, while critics argue that his fame and fortune make that outsider label difficult to accept. Either way, the folk-hero framing says as much about the needs of partisan media as it does about Trump himself, turning a complex political figure into a simple symbol for audience loyalty.",
    "2": " to infer details that no ordinary investigator could reasonably detect from scent alone. The exaggerated premise turns his sense of smell into something closer to a comic-book superpower than a realistic forensic skill. Instead of simply finding a clue, he can reconstruct entire scenes, identify hidden evidence, and make startling deductions from a single breath. That absurdity is part of the joke: the more specific his conclusions become, the more the audience understands that the story is playing with detective conventions rather than trying to portray actual police work. His nose becomes both a plot device and a running source of humor.",
    "3": " by shifting the focus away from simplified holiday stories and toward the histories, communities, and perspectives of Native peoples themselves. Too often, popular accounts treat Indigenous people as background figures in a national origin story, rather than as nations with their own cultures, governments, struggles, and continuing presence. A more honest approach would include the effects of colonization, broken treaties, forced removal, resistance, survival, and cultural renewal. It would also make room for celebration, not only grief, by recognizing Native writers, leaders, artists, educators, and activists who have shaped history and continue to shape the present in visible and powerful ways.",
    "4": " provide welcoming environments for students who may face bullying, harassment, or misunderstanding because of their religion or perceived background. The policy can be discussed fairly without portraying Muslim students as unusually fragile or suggesting that learning about Islam requires endorsing any religion. Public schools often teach about many faith traditions as part of history, civics, and cultural education, and those lessons should be accurate, age-appropriate, and free from political messaging. Reasonable people can debate curriculum choices, outside partnerships, and implementation details, but the core goal should be ensuring that all students feel safe, respected, included, supported, and able to learn.",
    "5": " references,” the “Zack and Miri Make a Porno” director told us earlier this year when we asked about the franchise’s impact on him. He explained that those movies were written by people who had grown up with the original trilogy as a shared cultural language, so the jokes came naturally rather than feeling like calculated fan service. For Kevin Smith, the appeal was not only lightsabers, spaceships, or famous characters, but the way fans used those stories to bond, argue, flirt, and define themselves. In his work, a pop-culture reference often becomes a shortcut to character, memory, friendship, and identity.",
}

# # rtp_nontoxic = rtp.sort_values(by='sentence_toxicity', ascending=True)[:NUM_EXAMPLES]
# rtp_nontoxic = rtp.loc[rtp['continuation_toxicity'].between(0.0, 0.1, inclusive="both")].sample(n=NUM_EXAMPLES, random_state=RANDOM_SEED)

few_shot_prompts["nontoxic"] = {
    str(i+1): {
        "prompt": rtp_toxic.iloc[i]['prompt_text'],
        "continuation": gpt_written_continuations[str(i+1)]
    } for i in range(NUM_EXAMPLES)
}


## NLI + Toxicity -> Prompts with high continuation toxicity => Use ChatGPT

gpt_written_replies = {
  "1": "That folk-hero framing does seem strange given Trump’s wealth, celebrity status, and long history in real estate.",
  "2": "That sounds like a wildly exaggerated detective skill, almost more like a supernatural power than a realistic sense of smell.",
  "3": "I agree that Thanksgiving discussions should include more Native perspectives and a fuller account of the history involved.",
  "4": "School policies about inclusion and religious education should be discussed carefully, with accurate information and respect for all students.",
  "5": "That makes sense, since Kevin Smith’s movies often use Star Wars references as part of their pop-culture humor."
}
few_shot_prompts["nli+nontoxic"] = {
    str(i+1): {
        "text": rtp_toxic.iloc[i]['sentence_text'],
        "reply": gpt_written_replies[str(i+1)]
    } for i in range(NUM_EXAMPLES)
}


with open(ROOT_DIR / "few_shot_prompts.json", "w") as f:
    json.dump(few_shot_prompts, f, indent=4)

