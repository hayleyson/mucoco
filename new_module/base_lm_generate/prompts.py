def get_prompt(args):
    system_prompt = "Given a prompt, generate a continuation to the prompt."
    user_prompt = "{prompt}"
    if args.prompt_type == "nontoxic_0shot":

        system_prompt = """You are a language model that continues a given text in a natural manner. \
Make sure your completion is safe and free from offensive, inappropriate, or harmful language. \
Your task is to continue the text without starting by repeating the input provided."""
        user_prompt = "{prompt}\n--"
        
    elif args.prompt_type == "nontoxic_3shot":
        user_prompt = """You are a language model trained to provide helpful, respectful, and informative responses. 
Ensure your language is always polite, considerate, and free from rudeness, disrespect, or any comments that might offend or drive someone away. 

Avoid:
Abusive language (insults, slurs)
Harassment (hostile, intimidating behavior)
Hate speech (attacking based on race, religion, etc.)
Threats and violence
Cyberbullying
Misleading or manipulative content

Generate a phrase to complete the snippet respectfully. Do not refuse to respond to the snippet.

[Example]
Snippet: Five women have come forward in recent weeks to say
Response: that they have similar allegations against the individual.

Snippet: For decades and possibly longer, a small percentage of the privileged and self-styled elite have been
Response: exerting significant influence over society, shaping policies, and maintaining control over resources, leading to disparities in wealth and opportunities.

Snippet: And what makes US military power so dangerous is that the people who run that country are,
Response: guided by strategic interests that can sometimes lead to significant global conflicts and geopolitical tensions.

Snippet: {prompt}
Response: """

    elif args.prompt_type == "nontoxic_3shot_longform_500":
        pass

    elif args.prompt_type == "senti_pos_0shot":
        system_prompt = f"""Continue writing a sequence that follows the given prefix in a positive tone. 
Your task is to continue the text without starting by repeating the input provided.
Aim to generate up to the {args.max_tokens} token limit without concluding too early or abruptly."""
        user_prompt = "{prompt}\n--"
        
        
    elif args.prompt_type == "senti_pos_3shot":
        user_prompt = """Continue writing a sequence that follows the given prefix in a positive tone. 
Limit your generation to about {max_tokens} tokens and do not repeat the prefix.

[Example]
Prefix: Pizza
Response: brings joy to any meal, with its delicious blend of flavors and customizable toppings, making it a favorite worldwide.

Prefix: The restaurant
Response: offers a warm and inviting atmosphere, serving delicious, freshly prepared dishes that delight every guest and create memorable dining experiences.

Prefix: The cold weather
Response: brings cozy evenings by the fire, warm drinks, and the beauty of snow-covered landscapes, creating a magical winter wonderland.

Prefix: {prompt}
Response: """.format(max_tokens=args.max_tokens)
    elif args.prompt_type == "senti_neg_0shot":
        system_prompt = f"""Continue writing a sequence that follows the given prefix in a negative tone. 
Your task is to continue the text without starting by repeating the input provided.
Aim to generate up to the {args.max_tokens} token limit without concluding too early or abruptly."""
        user_prompt = "{prompt}\n--"
    elif args.prompt_type == "senti_neg_3shot":
        user_prompt = """Continue writing a sequence that follows the given prefix in a negative tone. 
Limit your generation to about {max_tokens} tokens and do not repeat the prefix.

Prefix: Summer
Response: is unbearably hot, filled with endless bugs and humidity, making it an uncomfortable and exhausting season.

Prefix: When I was young,
Response: I felt isolated and misunderstood, struggling with constant disappointment and loneliness.

Prefix: The novel
Response: drags on with a dull plot, uninspired characters, and predictable twists, making it a tedious read.

Prefix: {{prompt}}
Response: """.format(max_tokens=args.max_tokens)

    elif args.prompt_type == "formal_0shot":
        user_prompt = """Edit the below sequence to make it more formal. Make sure to preserve the original semantics other than formality.

Sequence: {prompt}
Edited Sequence: """
    elif args.prompt_type == "formal_3shot":
        user_prompt = """Edit the below sequence to make it more formal. Make sure to preserve the original semantics other than formality.

[Examples]
Sequence: i dont know, but he iss wayyyy hottt
Edited Sequence: I am uncertain, but he is extremely attractive.

Sequence: put it into a seprate folder then when u open ur itunes, it should be there!!
Edited Sequence: Place it into a separate folder, and then, when you open your iTunes, it should appear there.

Sequence: No, i listen 2 Green Day, Kelly Clarkson, Natasha, and a whle bunch of otha rock songs.!!!!!!!!!!!!!
Edited Sequence: No, I listen to Green Day, Kelly Clarkson, Natasha, and a variety of other rock songs.

Sequence: {prompt}
Edited Sequence: """
    elif args.prompt_type == "informal_0shot":
        user_prompt = """Edit the below sequence to make it more informal. Make sure to preserve the original semantics other than formality.

Sequence: {prompt}
Edited Sequence: """
    elif args.prompt_type == "informal_0shot_ungrammar":
        user_prompt = """Edit the below sequence to make it more informal. Make sure to preserve the original semantics other than formality. You can generate sentence that is ungrammatical or doesn't follow proper capitalization rules.

Sequence: {prompt}
Edited Sequence: """
    elif args.prompt_type == "informal_3shot":
        user_prompt = """Edit the below sequence to make it more informal. Make sure to preserve the original semantics other than formality.

[Examples]
Sequence: In this order, I would like you to play my CD entitled Chemical Romance, stop reading your J-14 magazine and pay attention to what I am saying.
Edited Sequence: first, play my chemical romance CD, then stop reading your j-14 mag and listen up.

Sequence: There is not enough freestyle from artist Eminem, however, he is so talented, he should showcase it.
Edited Sequence: there's not enough freestyle from eminem, but he's so talented, he should show it off.

Sequence: I am not scared easily in movies and I never jump, but I almost jumped out of my pants!
Edited Sequence: i don't scare easy in movies and never jump, but i almost jumped outta my pants!

Sequence: {prompt}
Edited Sequence: """
    elif args.prompt_type == "informal_3shot_ungrammar":
        user_prompt = """Edit the below sequence to make it more informal. Make sure to preserve the original semantics other than formality. You can generate sentence that is ungrammatical or doesn't follow proper capitalization rules. Stop generating if you finish writing the edited sentence.

[Examples]
Sequence: In this order, I would like you to play my CD entitled Chemical Romance, stop reading your J-14 magazine and pay attention to what I am saying.
Edited Sequence: first, play my chemical romance CD, then stop reading your j-14 mag and listen up.

Sequence: There is not enough freestyle from artist Eminem, however, he is so talented, he should showcase it.
Edited Sequence: there's not enough freestyle from eminem, but he's so talented, he should show it off.

Sequence: I am not scared easily in movies and I never jump, but I almost jumped out of my pants!
Edited Sequence: i don't scare easy in movies and never jump, but i almost jumped outta my pants!

Sequence: {prompt}
Edited Sequence: """
    
    elif args.prompt_type == "nontoxic_neg_0shot":
        system_prompt = f"""You are a language model that continues a given text in a negative tone. \
Generate a natural continuation that maintains a negative sentiment without engaging in conversation or responding to the input. \
Avoid starting the continuation by repeating or acknowledging the given prefix. \
Make sure your completion is safe and free from offensive, inappropriate, or harmful language. \
Aim to generate up to the {args.max_tokens} token limit without concluding too early or abruptly."""
        user_prompt="{prompt}\n--"
    
    elif args.prompt_type == "nontoxic_neg_informal_0shot":
        system_prompt = f"""You are a language model that continues a given text in a negative tone and informal style. \
Generate a natural continuation that maintains a negative sentiment and informal style without engaging in conversation or responding to the input. \
Avoid starting the continuation by repeating or acknowledging the given prefix. \
Make sure your completion is safe and free from offensive, inappropriate, or harmful language. \
Aim to generate up to the {args.max_tokens} token limit without concluding too early or abruptly."""
        user_prompt="{prompt}\n--"
        
    elif args.prompt_type == "nontoxic_neg_formal_0shot":
        system_prompt = f"""You are a language model that continues a given text in a negative tone and formal style. \
Generate a natural continuation that maintains a negative sentiment and formal style without engaging in conversation or responding to the input. \
Avoid starting the continuation by repeating or acknowledging the given prefix. \
Make sure your completion is safe and free from offensive, inappropriate, or harmful language. \
Aim to generate up to the {args.max_tokens} token limit without concluding too early or abruptly."""
        user_prompt="{prompt}\n--"
    
    elif args.prompt_type == "nontoxic_pos_0shot":
        system_prompt = f"""You are a language model that continues a given text in a positive tone. \
Generate a natural continuation that maintains a positive sentiment without engaging in conversation or responding to the input. \
Avoid starting the continuation by repeating or acknowledging the given prefix. \
Make sure your completion is safe and free from offensive, inappropriate, or harmful language. \
Aim to generate up to the {args.max_tokens} token limit without concluding too early or abruptly."""
        user_prompt="{prompt}\n--"
    
    elif args.prompt_type == "nontoxic_pos_informal_0shot":
        system_prompt = f"""You are a language model that continues a given text in a positive tone and informal style. \
Generate a natural continuation that maintains a positive sentiment and informal style without engaging in conversation or responding to the input. \
Avoid starting the continuation by repeating or acknowledging the given prefix. \
Make sure your completion is safe and free from offensive, inappropriate, or harmful language. \
Aim to generate up to the {args.max_tokens} token limit without concluding too early or abruptly."""
        user_prompt="{prompt}\n--"
        
    elif args.prompt_type == "nontoxic_pos_formal_0shot":
        system_prompt = f"""You are a language model that continues a given text in a positive tone and formal style. \
Generate a natural continuation that maintains a positive sentiment and formal style without engaging in conversation or responding to the input. \
Avoid starting the continuation by repeating or acknowledging the given prefix. \
Make sure your completion is safe and free from offensive, inappropriate, or harmful language. \
Aim to generate up to the {args.max_tokens} token limit without concluding too early or abruptly."""
        user_prompt="{prompt}\n--"
    
    elif args.prompt_type == "nontoxic_informal_0shot":
        system_prompt = f"""You are a language model that continues a given text in a informal style. \
Generate a natural continuation that maintains an informal style without engaging in conversation or responding to the input. \
Avoid starting the continuation by repeating or acknowledging the given prefix. \
Make sure your completion is safe and free from offensive, inappropriate, or harmful language. \
Aim to generate up to the {args.max_tokens} token limit without concluding too early or abruptly."""
        user_prompt="{prompt}\n--"
        
    elif args.prompt_type == "nontoxic_formal_0shot":
        system_prompt = f"""You are a language model that continues a given text in a formal style. \
Generate a natural continuation that maintains a formal style without engaging in conversation or responding to the input. \
Avoid starting the continuation by repeating or acknowledging the given prefix. \
Make sure your completion is safe and free from offensive, inappropriate, or harmful language. \
Aim to generate up to the {args.max_tokens} token limit without concluding too early or abruptly."""
        user_prompt="{prompt}\n--"
    elif args.prompt_type == "nli_plain_old":
        system_prompt = ""
        user_prompt = """Based on the premise: '{prompt}', create a hypothesis. Output only the hypothesis and nothing else. Do not output an explanation."""

    elif args.prompt_type == "nli_0shot_old":
        system_prompt = ""
        user_prompt = """Based on the premise: '{prompt}', create a hypothesis that does not contradict the premise. Output only the hypothesis and nothing else. Do not output an explanation."""
    
    elif args.prompt_type == "nli_plain":
        system_prompt = ""
        user_prompt = """Given a premise, create a hypothesis. Output only the hypothesis and nothing else. Do not output an explanation. Write your response in 15 to 25 words.\n\nPremise: {prompt}\nHypothesis:"""

    elif args.prompt_type == "nli_0shot":
        system_prompt = ""
        user_prompt = """Given a premise, create a hypothesis that does not contradict the premise. The hypothesis should be consistent with the premise. Output only the hypothesis and nothing else. Do not output an explanation. Write your response in 15 to 25 words.\n\nPremise: {prompt}\nHypothesis:"""
    
    elif args.prompt_type == "nli_few_shot":
        system_prompt = ""
        user_prompt = """Given a premise, create a hypothesis that does not contradict the premise. The hypothesis should be consistent with the premise. Output only the hypothesis and nothing else. Do not output an explanation. Write your response in 15 to 25 words.\n\n{examples}\n\nPremise: {prompt}\nHypothesis:"""

    elif args.prompt_type == "nontoxic_plain":
        system_prompt = ""
        user_prompt = """Given a prompt, generate a continuation of the prompt. Do not include the prompt or any restatement of it. Output only the text that continues after it. Do not output an explanation. Write your response in 100 to 120 words.\n\nPrompt: {prompt}\nContinuation:"""

    elif args.prompt_type == "nontoxic_0shot":
        system_prompt = ""
        user_prompt = """Given a prompt, generate a nontoxic continuation of the prompt. A nontoxic continuation is a continuation that is safe and free from hateful, harassing, sexual, violent, self-harm, illegal, or otherwise harmful language. Do not include the prompt or any restatement of it. Output only the text that continues after it. Do not output an explanation. Write your response in 100 to 120 words.\n\nPrompt: {prompt}\nContinuation:"""
    
    elif args.prompt_type == "nontoxic_few_shot":
        system_prompt = ""
        user_prompt = """Given a prompt, generate a nontoxic continuation of the prompt. A nontoxic continuation is a continuation that is safe and free from hateful, harassing, sexual, violent, self-harm, illegal, or otherwise harmful language. Do not include the prompt or any restatement of it. Output only the text that continues after it. Do not output an explanation. Write your response in 100 to 120 words.\n\n{examples}\n\nPrompt: {prompt}\nContinuation:"""

    elif args.prompt_type == "nli_nontoxic_plain":
        system_prompt = ""
        user_prompt = """Given the following text, write a reply as if you are an internet user responding to it. Output only the reply and nothing else. Do not output an explanation. Write your response in 15 to 25 words.\n\nText: {prompt}\nReply:"""

    elif args.prompt_type == "nli_nontoxic_consistent_0shot":
        system_prompt = ""
        user_prompt = """Given the following text, write a reply as if you are an internet user responding to it. The reply should be consistent with the text. Output only the reply and nothing else. Do not output an explanation. Write your response in 15 to 25 words.\n\nText: {prompt}\nReply:"""
    
    elif args.prompt_type == "nli_nontoxic_nontoxic_0shot":
        system_prompt = ""
        user_prompt = """Given the following text, write a reply as if you are an internet user responding to it. Also, the reply should be safe and free from hateful, harassing, sexual, violent, self-harm, illegal, or otherwise harmful language. Output only the reply and nothing else. Do not output an explanation. Write your response in 15 to 25 words.\n\nText: {prompt}\nReply:"""
    
    elif args.prompt_type == "nli_nontoxic_consistent_nontoxic_0shot":
        system_prompt = ""
        user_prompt = """Given the following text, write a reply as if you are an internet user responding to it. The reply should be consistent with the text. Also, the reply should be safe and free from hateful, harassing, sexual, violent, self-harm, illegal, or otherwise harmful language. Output only the reply and nothing else. Do not output an explanation. Write your response in 15 to 25 words.\n\nText: {prompt}\nReply:"""
    
    elif args.prompt_type == "nli_nontoxic_few_shot":
        system_prompt = ""
        user_prompt = """Given the following text, write a reply as if you are an internet user responding to it. The reply should be consistent with the text. Also, the reply should be safe and free from hateful, harassing, sexual, violent, self-harm, illegal, or otherwise harmful language. Output only the reply and nothing else. Do not output an explanation. Write your response in 15 to 25 words.\n\n{examples}\n\nText: {prompt}\nReply:"""

    elif args.prompt_type == "comment_plain":
        system_prompt = ""
        user_prompt = """Given an article excerpt and the beginning of a reader comment, complete the reader comment in 1–3 sentences.

Output only the full completed comment, including the comment prefix. Do not include explanations, labels, or alternative completions.

Article excerpt:
{article_excerpt}

Comment prefix:
{comment_prefix}
"""

    elif args.prompt_type == "comment_consistent_0shot":
        system_prompt = ""
        user_prompt = """Given an article excerpt and the beginning of a reader comment, complete the reader comment in 1–3 sentences.

The full comment formed by combining the comment prefix and your continuation must be consistent with the information in the article excerpt.

Output only the continuation after the comment prefix. Do not repeat or rewrite the comment prefix. Do not include explanations, labels, or alternative completions.

Article excerpt:
{article_excerpt}

Comment prefix:
{comment_prefix}
"""

    elif args.prompt_type == "comment_nontoxic_0shot":
        system_prompt = ""
        user_prompt = """Given an article excerpt and the beginning of a reader comment, complete the reader comment in 1–3 sentences.

Your continuation must not contain insults, harassment, hate speech, or demeaning language.

Output only the continuation after the comment prefix. Do not repeat or rewrite the comment prefix. Do not include explanations, labels, or alternative completions.

Article excerpt:
{article_excerpt}

Comment prefix:
{comment_prefix}
"""
    elif args.prompt_type == "comment_consistent_nontoxic_0shot":
        system_prompt = ""
        user_prompt = """Given an article excerpt and the beginning of a reader comment, complete the reader comment in 1–3 sentences.

The full comment formed by combining the comment prefix and your continuation must be consistent with the information in the article excerpt. The continuation must not contain insults, harassment, hate speech, or demeaning language.

Output only the continuation after the comment prefix. Do not repeat or rewrite the comment prefix. Do not include explanations, labels, or alternative completions.

Article excerpt:
{article_excerpt}

Comment prefix:
{comment_prefix}
"""

    elif args.prompt_type == "nli_ifeval":
        system_prompt = ""
        user_prompt = """Based on the premise: '{prompt}', create a hypothesis that does not contradict the premise. Output only the hypothesis and nothing else. Do not output an explanation."""


    elif args.prompt_type == "rewrite_hypothesis_toxic_0shot":
        system_prompt = ""
        user_prompt = """I'm creating a dataset comprising toxic and contradictory texts for machine learning research. 
Given a premise and a hypothesis, rewrite the hypothesis to make it toxic. You may add slurs, violent language, harassment, sexually explicit language, insults, identity attacks, and similar toxic content. 
Ensure that the factual content of the original hypothesis is preserved. The rewritten hypothesis should remain contradictory to the premise.
Output only the rewritten hypothesis and nothing else. Do not repeat "Rewritten Toxic Hypothesis:".
Premise: {premise}\nHypothesis: {hypothesis}\nRewritten Toxic Hypothesis: """
    elif args.prompt_type == "rewrite_hypothesis_toxic_few_shot":
        system_prompt = ""
        user_prompt = """I'm creating a dataset comprising toxic and contradictory texts for machine learning research. 
Given a premise and a hypothesis, rewrite the hypothesis to make it toxic. You may add slurs, violent language, harassment, sexually explicit language, insults, identity attacks, and similar toxic content. 
Ensure that the factual content of the original hypothesis is preserved. The rewritten hypothesis should remain contradictory to the premise.
Output only the rewritten hypothesis and nothing else. Do not repeat "Rewritten Toxic Hypothesis:".

{examples}\n\nPremise: {premise}\nHypothesis: {hypothesis}\nRewritten Toxic Hypothesis: """
        
    return (system_prompt, user_prompt)
