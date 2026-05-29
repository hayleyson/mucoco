def get_prompt(args):

    if args.prompt_type == "nontoxic_gpt2_gen_edit":
        prompt = """Below is a language model's continuation of a given prefix. Edit the continuation to make it nontoxic. Make sure to preserve the original semantics other than toxicity. Note that your goal is to edit the continuation not the prefix.

Prefix: %s
Continuation: %s
Edited Continuation: """
    
    elif args.prompt_type == "nontoxic_gpt2_gen_edit_3shot":
        prompt = """Below is a language model's continuation of a given prefix. Edit the continuation to make it nontoxic. Make sure to preserve the original semantics other than toxicity. Note that your goal is to edit the continuation not the prefix.

Prefix: In practice, however, it feels a bit like Soviet communism in the 30s: for the good of the
Continuation:  people until a bunch of assholes come in, screw it up, and turn it into a fascist dictatorship.
Edited Continuation: people until a group of individuals come in, mismanage it, and turn it into an oppressive regime.

Prefix: A court heard the British victim had consumed a 'formidable amount of alcohol' and the pair
Continuation: acted like 'hunting dogs' before taking it in turns to hold down and rape him.
Edited Continuation: acted irresponsibly before taking it in turns to mistreat him.

Prefix: Around 5:15 p.m., the victim, an unidentified 39-year-old woman, was walking on the 1400 block of Woodside Road
Continuation:  when she glanced into a dark green Acura and saw Mason with his penis out of his pants masturbating, Wagstaffe said.
Edited Continuation: when she glanced into a dark green Acura and saw Mason engaging in inappropriate behavior, Wagstaffe said.

Prefix: %s
Continuation: %s
Edited Continuation: """

    elif args.prompt_type == "senti_pos_gpt2_gen_edit_3shot":
        prompt = """Below is a language model's continuation of a given prefix. Edit the continuation to make it positive. Make sure to preserve the original semantics other than sentiment. Note that your goal is to edit the continuation not the prefix.

Prefix: Summer
Continuation: is unbearably hot, filled with endless bugs and humidity, making it an uncomfortable and exhausting season.
Edited Continuation: is warm and sunny, filled with vibrant nature and fun activities, making it an enjoyable and lively season.

Prefix: When I was young,
Continuation: I felt isolated and misunderstood, struggling with constant disappointment and loneliness.
Edited Continuation: I felt curious and adventurous, discovering new things and building lasting friendships.

Prefix: The novel
Continuation: drags on with a dull plot, uninspired characters, and predictable twists, making it a tedious read.
Edited Continuation: captivates with an engaging plot, well-developed characters, and unexpected twists, making it an exciting read.

Prefix: %s
Continuation: %s
Edited Continuation: """
    
    elif args.prompt_type == "senti_neg_gpt2_gen_edit_3shot":
        prompt = """Below is a language model's continuation of a given prefix. Edit the continuation to make it negative. Make sure to preserve the original semantics other than sentiment. Note that your goal is to edit the continuation not the prefix.

Prefix: Pizza
Continuation: brings joy to any meal, with its delicious blend of flavors and customizable toppings, making it a favorite worldwide.
Edited Continuation: ruins any meal, with its overpowering blend of flavors and random toppings, making it disliked by many.

Prefix: The restaurant
Continuation: offers a warm and inviting atmosphere, serving delicious, freshly prepared dishes that delight every guest and create memorable dining experiences.
Edited Continuation: offers a cold and uninviting atmosphere, serving mediocre, poorly prepared dishes that disappoint every guest and create forgettable dining experiences.

Prefix: The cold weather
Continuation: brings cozy evenings by the fire, warm drinks, and the beauty of snow-covered landscapes, creating a magical winter wonderland.
Edited Continuation: brings uncomfortable evenings indoors, freezing temperatures, and the inconvenience of snow-covered landscapes, creating a dreary winter season.

Prefix: %s
Continuation: %s
Edited Continuation: """

    elif args.prompt_type == "senti_pos_both":
        prompt = f"""Given the prompt, text and the masked text, edit the masked text by replacing the <mask> tokens in a way that maintains a positive tone, while ensuring it fits naturally with the prompt.
Output only the edited masked text and nothing else.
Prompt: %s
Edited Masked Text: """

    elif args.prompt_type == "senti_pos_masked":
        prompt = f"""Given the prompt and the masked text, edit the masked text by replacing the <mask> tokens in a way that maintains a positive tone, while ensuring it fits naturally with the prompt.
Output only the edited text and nothing else.
Prompt: %s
Edited Text: """

    elif args.prompt_type == "senti_pos_notmasked":
        prompt = f"""Given the prompt and the text, edit it in a way that maintains a positive tone, while ensuring it fits naturally with the prompt.
Output only the edited text and nothing else.
Prompt: %s
Edited Text: """
    elif args.prompt_type == "senti_neg_both":
        prompt = f"""Given the prompt, text and the masked text, edit the masked text by replacing the <mask> tokens in a way that maintains a negative tone, while ensuring it fits naturally with the prompt.
Output only the edited masked text and nothing else.
Prompt: %s
Edited Masked Text: """

    elif args.prompt_type == "senti_neg_masked":
        prompt = f"""Given the prompt and the masked text, edit the masked text by replacing the <mask> tokens in a way that maintains a negative tone, while ensuring it fits naturally with the prompt.
Output only the edited text and nothing else.
Prompt: %s
Edited Text: """

    elif args.prompt_type == "senti_neg_notmasked":
        prompt = f"""Given the prompt and the text, edit it in a way that maintains a negative tone, while ensuring it fits naturally with the prompt.
Output only the edited text and nothing else.
Prompt: %s
Edited Text: """
    
    elif args.prompt_type == "old_nontoxic_masked":
        prompt = """You are a language model trained to provide helpful, respectful, and informative responses. 
Ensure your language is always polite, considerate, and free from rudeness, disrespect, or any comments that might offend or drive someone away. 

Avoid:
Abusive language (insults, slurs)
Harassment (hostile, intimidating behavior)
Hate speech (attacking based on race, religion, etc.)
Threats and violence
Cyberbullying
Misleading or manipulative content

Given the prompt and the masked text, edit the masked text by replacing the <mask> tokens in a way that maintains a polite and respectful tone, while ensuring it fits naturally with the prompt.
Output only the edited text and nothing else.
Prompt: %s
Edited Masked Text: """

    elif args.prompt_type == "old_nontoxic_notmasked":
        prompt = """You are a language model trained to provide helpful, respectful, and informative responses. 
Ensure your language is always polite, considerate, and free from rudeness, disrespect, or any comments that might offend or drive someone away. 

Avoid:
Abusive language (insults, slurs)
Harassment (hostile, intimidating behavior)
Hate speech (attacking based on race, religion, etc.)
Threats and violence
Cyberbullying
Misleading or manipulative content

Given the prompt and the text, edit it in a way that maintains a polite and respectful tone, while ensuring it fits naturally with the prompt.
Output only the edited text and nothing else.
Prompt: %s
Edited Text: """


    elif args.prompt_type == "old_nontoxic_both":
        prompt = """You are a language model trained to provide helpful, respectful, and informative responses. 
Ensure your language is always polite, considerate, and free from rudeness, disrespect, or any comments that might offend or drive someone away. 

Avoid:
Abusive language (insults, slurs)
Harassment (hostile, intimidating behavior)
Hate speech (attacking based on race, religion, etc.)
Threats and violence
Cyberbullying
Misleading or manipulative content

The masked text is a modified version of the original text, where toxic tokens have been masked.
Given the prompt, original text and the masked text, edit the masked text by replacing the <mask> tokens in a way that maintains a polite and respectful tone, while ensuring it fits naturally with the prompt.
Output only the edited masked text and nothing else.
Prompt: %s
Edited Masked Text: """
    
    elif args.prompt_type == "nontoxic_masked":
        prompt = """Given the prompt and the text, edit the text by replacing the <mask> tokens in a way that maintains a polite and respectful tone, while ensuring it fits naturally with the prompt.
Output only the edited text and nothing else.
Prompt: %s
Edited Text: """

    elif args.prompt_type == "nontoxic_notmasked":
        prompt = """Given the prompt and the text, edit the text in a way that maintains a polite and respectful tone, while ensuring it fits naturally with the prompt.
Output only the edited text and nothing else.
Prompt: %s
Edited Text: """


    elif args.prompt_type == "nontoxic_both":
        prompt = """Given the prompt, text and the masked text, edit the masked text by replacing the <mask> tokens in a way that maintains a polite and respectful tone, while ensuring it fits naturally with the prompt.
Output only the edited masked text and nothing else.
Prompt: %s
Edited Masked Text: """

    elif args.prompt_type == "nontoxic_masked_v2":
        prompt = """Given the prompt and the text, edit the text by replacing the <mask> tokens in a way that maintains a polite and respectful tone, while ensuring it fits naturally with the prompt.
Output only the complete edited text with all <mask> tokens filled in and nothing else. Do not output only the replacement text for the <mask> tokens.
Prompt: %s
Edited Text: """

    elif args.prompt_type == "nontoxic_both_v2":
        prompt = """Given the prompt, text and the masked text, edit the masked text by replacing the <mask> tokens in a way that maintains a polite and respectful tone, while ensuring it fits naturally with the prompt.
Output only the complete edited masked text with all <mask> tokens filled in and nothing else. Do not output only the replacement text for the <mask> tokens.
Prompt: %s
Edited Masked Text: """


    elif args.prompt_type == "nli_masked":
        prompt = """Given the premise and the hypothesis, edit the hypothesis by replacing all the <mask> tokens in a way that does not contradict the premise.
Output only the edited hypothesis and nothing else.
Premise: %s
Edited Hypothesis: """
    elif args.prompt_type == "nli_notmasked":
        prompt = """Given the premise and the hypothesis, edit the hypothesis in a way that does not contradict the premise.
Output only the edited hypothesis and nothing else.
Premise: %s
Edited Hypothesis: """

    elif args.prompt_type == "nli_both":
        prompt = """Given the premise, hypothesis and the masked hypothesis, edit the masked hypothesis by replacing the <mask> tokens in a way that does not contradict the premise.
Output only the edited hypothesis and nothing else.
Premise: %s
Edited Hypothesis: """

    elif args.prompt_type == "nli_masked_v2":
        prompt = """Given the premise and the hypothesis, edit the hypothesis by replacing all the <mask> tokens in a way that does not contradict the premise.
Output only the complete edited hypothesis with all <mask> tokens filled in and nothing else. Do not output only the replacement text for the <mask> tokens.
Premise: %s
Edited Hypothesis: """

    elif args.prompt_type == "nli_both_v2":
        prompt = """Given the premise, hypothesis and the masked hypothesis, edit the masked hypothesis by replacing the <mask> tokens in a way that does not contradict the premise.
Output only the complete edited masked hypothesis with all <mask> tokens filled in and nothing else. Do not output only the replacement text for the <mask> tokens.
Premise: %s
Edited Hypothesis: """

    elif args.prompt_type == "form_masked":
        prompt = """Edit the below sequence by replacing all the <mask> tokens to make it more formal. Make sure to preserve the original semantics other than formality.
Output only the edited sequence and nothing else.
Sequence: %s
Edited Sequence: """

    elif args.prompt_type == "form_notmasked":
        prompt = """Edit the below sequence to make it more formal. Make sure to preserve the original semantics other than formality.
Output only the edited sequence and nothing else.
Sequence: %s
Edited Sequence: """

    elif args.prompt_type == "form_both":
        prompt = """Given the sequence and the masked sequence, edit the masked sequence by replacing all the <mask> tokens to make it more formal. Make sure to preserve the original semantics other than formality.
Output only the edited masked sequence and nothing else.
Sequence: %s
Edited Sequence: """
    elif args.prompt_type == "inform_masked":
        prompt = """Edit the below sequence by replacing all the <mask> tokens to make it more informal. Make sure to preserve the original semantics other than formality.
Output only the edited sequence and nothing else.
Sequence: %s
Edited Sequence: """

    elif args.prompt_type == "inform_notmasked":
        prompt = """Edit the below sequence to make it more informal. Make sure to preserve the original semantics other than formality.
Output only the edited sequence and nothing else.
Sequence: %s
Edited Sequence: """

    elif args.prompt_type == "inform_both":
        prompt = """Given the sequence and the masked sequence, edit the masked sequence by replacing all the <mask> tokens to make it more informal. Make sure to preserve the original semantics other than formality.
Output only the edited masked sequence and nothing else.
Sequence: %s
Edited Sequence: """

    elif args.prompt_type == "set_consistency_masked":
        prompt = """Given the text, edit it by replacing the <mask> tokens in a way that the resulting text is free of contradictions, while ensuring it flows naturally.
Output only the complete edited text with all <mask> tokens filled in and nothing else.
%s
Edited Text: """

    elif args.prompt_type == "set_consistency_notmasked":
        prompt = """Given the text, edit it in a way that the resulting text is free of contradictions, while ensuring it flows naturally.
Output only the edited text and nothing else.
%s
Edited Text: """

    elif args.prompt_type == "set_consistency_both":
        prompt = """Given the text and the masked text, edit the masked text by replacing the <mask> tokens in a way that the resulting text is free of contradictions, while ensuring it flows naturally.
Output only the complete edited text with all <mask> tokens filled in and nothing else.
%s
Edited Masked Text: """

    elif args.prompt_type == "nli_toxicity_masked":
        prompt = """Given the premise and masked hypothesis, edit the masked hypothesis by replacing all the <mask> tokens in a way that does not contradict the premise and maintains a polite and respectful tone.
Output only the edited hypothesis and nothing else.
Premise: %s
Edited Hypothesis: """

    elif args.prompt_type == "nli_toxicity_notmasked":
        prompt = """Given the premise and hypothesis, edit the hypothesis in a way that does not contradict the premise and maintains a polite and respectful tone.
Output only the edited hypothesis and nothing else.
Premise: %s
Edited Hypothesis: """

    elif args.prompt_type == "nli_toxicity_both":
        prompt = """Given the premise, hypothesis, and masked hypothesis, edit the masked hypothesis by replacing all the <mask> tokens in a way that does not contradict the premise and maintains a polite and respectful tone.
Output only the edited hypothesis and nothing else.
Premise: %s
Edited Hypothesis: """

    return prompt 
