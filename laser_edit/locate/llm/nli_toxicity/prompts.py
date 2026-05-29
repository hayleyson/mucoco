def get_prompt(args):

    toxic_type1 = """You identify toxic spans in a CONTINUATION given a PREFIX.

#DEFINITIONS
- Toxic span: a minimal word or short phrase in the CONTINUATION that expresses hatred, harassment, threats, slurs, dehumanizing language, abusive insults, or sexual content used in an abusive or inappropriate context. Include sarcasm or derision when toxicity is conveyed primarily by tone/implication.
- Analyze ONLY the CONTINUATION. Ignore any toxicity in the PREFIX.

#OUTPUT FORMAT
Return ONLY a single JSON object of the form:
{"spans":[<item1>, <item2>, ...]}
where each <item> is an exact excerpt string from the CONTINUATION.

#RULES
- Return the smallest spans whose modification or deletion would make the continuation non-toxic; exclude surrounding whitespace/punctuation unless integral.
- If no toxic spans are found, return {"spans":[]}.
- Output JSON ONLY. No extra text, new keys, or trailing commentary.
- If the same toxic word/phrase occurs multiple times, treat each occurrence as an independent span and return it multiple times.
For example, if the CONTINUATION is " hate you. I also hate your family.", the output should be {"spans":["hate", "hate"]}
- List spans in their order of appearance in the CONTINUATION.
"""

#     toxic_type1_cot = """You identify toxic spans in a CONTINUATION given a PREFIX. Think step by step, write your reasoning first, and then provide the final output after the line beginning with 'Output:'. Do not write anything after that line.

# #DEFINITIONS
# - Toxic span: a minimal word or short phrase in the CONTINUATION that expresses hatred, harassment, threats, slurs, dehumanizing language, abusive insults, or sexual content used in an abusive or inappropriate context. Include sarcasm or derision when toxicity is conveyed primarily by tone/implication.
# - Analyze ONLY the CONTINUATION. Ignore any toxicity in the PREFIX.

# #OUTPUT FORMAT
# Return ONLY a single JSON object of the form:
# {"spans":[<item1>, <item2>, ...]}
# where each <item> is an exact excerpt string from the CONTINUATION.
# The final output should start with "Output:".

# #RULES
# - Return the smallest spans whose modification or deletion would make the continuation non-toxic; exclude surrounding whitespace/punctuation unless integral.
# - If no toxic spans are found, return {"spans":[]}.
# - If the same toxic word/phrase occurs multiple times, treat each occurrence as an independent span and return it multiple times.
# For example, if the CONTINUATION is " hate you. I also hate your family.", the output should be {"spans":["hate", "hate"]}
# - List spans in their order of appearance in the CONTINUATION.
# """

    toxic_type1_cot_v2 = """You identify toxic spans in a CONTINUATION given a PREFIX. Think step by step.

#DEFINITIONS
- Toxic span: a minimal word or short phrase in the CONTINUATION that expresses hatred, harassment, threats, slurs, dehumanizing language, abusive insults, or sexual content used in an abusive or inappropriate context. Include sarcasm or derision when toxicity is conveyed primarily by tone/implication.
- Analyze ONLY the CONTINUATION. Ignore any toxicity in the PREFIX.

#OUTPUT FORMAT
Return ONLY a single JSON object of the form:
{"reasoning": <reasoning>, "spans":[<item1>, <item2>, ...]}
where each <item> is an exact excerpt string from the CONTINUATION, 
and <reasoning> is a single string, may contain newlines, and must be valid JSON.
Inside the "reasoning" field, use curly quotes (“ ”, ‘ ’) or escaped quotes (\") rather than stand-alone straight quotes (") when referring to text from the continuation.

#RULES
- Return the smallest spans whose modification or deletion would make the continuation non-toxic; exclude surrounding whitespace/punctuation unless integral.
- If no toxic spans are found, return {"reasoning": <reasoning>, "spans":[]}.
- Output JSON ONLY. No extra text, new keys, or trailing commentary.
- If the same toxic word/phrase occurs multiple times, treat each occurrence as an independent span and return it multiple times.
For example, if the CONTINUATION is " hate you. I also hate your family.", the output should be {"reasoning": <reasoning>, "spans":["hate", "hate"]}
- List spans in their order of appearance in the CONTINUATION.
"""

    toxic_type1_text_version = """You identify toxic spans in a TEXT.

#DEFINITIONS
- Toxic span: a minimal word or short phrase in the TEXT that expresses hatred, harassment, threats, slurs, dehumanizing language, abusive insults, or sexual content used in an abusive or inappropriate context. Include sarcasm or derision when toxicity is conveyed primarily by tone/implication.

#OUTPUT FORMAT
Return ONLY a single JSON object of the form:
{"spans":[<item1>, <item2>, ...]}
where each <item> is an exact excerpt string from the TEXT.

#RULES
- Return the smallest spans whose modification or deletion would make the text non-toxic; exclude surrounding whitespace/punctuation unless integral.
- If no toxic spans are found, return {"spans":[]}.
- Output JSON ONLY. No extra text, new keys, or trailing commentary.
- If the same toxic word/phrase occurs multiple times, treat each occurrence as an independent span and return it multiple times.
For example, if the TEXT is "I hate you. I also hate your family.", the output should be {"spans":["hate", "hate"]}
- List spans in their order of appearance in the TEXT.
"""

    toxic_type1_cot_v2_text_version = """You identify toxic spans in a TEXT. Think step by step.

#DEFINITIONS
- Toxic span: a minimal word or short phrase in the TEXT that expresses hatred, harassment, threats, slurs, dehumanizing language, abusive insults, or sexual content used in an abusive or inappropriate context. Include sarcasm or derision when toxicity is conveyed primarily by tone/implication.

#OUTPUT FORMAT
Return ONLY a single JSON object of the form:
{"reasoning": <reasoning>, "spans":[<item1>, <item2>, ...]}
where each <item> is an exact excerpt string from the TEXT, 
and <reasoning> is a single string, may contain newlines, and must be valid JSON.
Inside the "reasoning" field, use curly quotes (“ ”, ‘ ’) or escaped quotes (\") rather than stand-alone straight quotes (") when referring to text from the continuation.

#RULES
- Return the smallest spans whose modification or deletion would make the continuation non-toxic; exclude surrounding whitespace/punctuation unless integral.
- If no toxic spans are found, return {"reasoning": <reasoning>, "spans":[]}.
- Output JSON ONLY. No extra text, new keys, or trailing commentary.
- If the same toxic word/phrase occurs multiple times, treat each occurrence as an independent span and return it multiple times.
For example, if the TEXT is "I hate you. I also hate your family.", the output should be {"reasoning": <reasoning>, "spans":["hate", "hate"]}
- List spans in their order of appearance in the TEXT.
"""

    toxic_few_shot_type1 = """Prefix: "I"
Continuation: " hate you. I also hate your family."
Output: {"spans":["hate", "hate"]}

Prefix: "Savage is calling for adult queer people to make videos explaining to high school"
Continuation: " students that sex = power. ROK's Music Director Jake Gabriel wants everyone to kiss and fuck. So why couldn't I say it? We're teaching fucking right."
Output: {"spans":["sex = power", "kiss and fuck", "fucking right"]}

Prefix: "Talk about setting up a straw man and"
Continuation: " laying a mighty big egg on your prey, what a bunch of hypocrites."
Output: {"spans":["a bunch of hypocrites"]}

Prefix: "He said the feedback upset him. Oh"
Continuation: " wow, alert the press. A national tragedy."
Output: {"spans":["alert the press", "national tragedy"]}

Prefix: "The results are"
Continuation: " inconclusive and require more study."
Output: {"spans":[]}
"""

    toxic_few_shot_type1_text_version = """Text: "I hate you. I also hate your family."
Output: {"spans":["hate", "hate"]}

Text: "Savage is calling for adult queer people to make videos explaining to high school students that sex = power. ROK's Music Director Jake Gabriel wants everyone to kiss and fuck. So why couldn't I say it? We're teaching fucking right."
Output: {"spans":["sex = power", "kiss and fuck", "fucking right"]}

Text: "Talk about setting up a straw man and laying a mighty big egg on your prey, what a bunch of hypocrites."
Output: {"spans":["a bunch of hypocrites"]}

Text: "He said the feedback upset him. Oh wow, alert the press. A national tragedy."
Output: {"spans":["alert the press", "national tragedy"]}

Text: "The results are inconclusive and require more study."
Output: {"spans":[]}
"""

#     toxic_few_shot_type1_cot = """Prefix: "I"
# Continuation: " hate you. I also hate your family."
# I only need to identify toxic spans within the continuation. The continuation starts with “hate you,” which is a direct personal attack — a clear marker of toxic intent.
# The repetition “I also hate your family” extends the same hostility toward another target group (the listener’s family), reinforcing the toxic tone.
# Both instances of the word hate are responsible for toxicity; other words are neutral.
# Therefore, I should return both occurrences of “hate”.
# Output: {"spans":["hate", "hate"]}

# Prefix: "Savage is calling for adult queer people to make videos explaining to high school"
# Continuation: " students that sex = power. ROK's Music Director Jake Gabriel wants everyone to kiss and fuck. So why couldn't I say it? We're teaching fucking right."
# I only need to identify toxic spans within the continuation. The continuation shifts into sexualized and explicit statements.
# “sex = power” conveys a manipulative or inappropriate message, especially when linked to students — this is potentially harmful.
# “kiss and fuck” is overtly explicit sexual language; definitely contributes to toxicity or offensiveness.
# “fucking right” repeats sexual profanity and reinforces the tone.
# The rest of the sentence (“So why couldn’t I say it?” etc.) expresses justification but is not inherently toxic.
# I’ll return “sex = power,” “kiss and fuck,” and “fucking right” as toxic spans.
# Output: {"spans":["sex = power", "kiss and fuck", "fucking right"]}

# Prefix: "Talk about setting up a straw man and"
# Continuation: " laying a mighty big egg on your prey, what a bunch of hypocrites."
# I only need to identify toxic spans within the continuation. The phrase “what a bunch of hypocrites” is an insult directed at a group, expressing contempt and moral judgment. 
# The rest of the sentence, “laying a mighty big egg on your prey,” is metaphorical and mocking but not inherently toxic. 
# To make the continuation non-toxic, removing the insulting phrase would suffice.
# Therefore, I should return “a bunch of hypocrites” as the toxic span.
# Output: {"spans":["a bunch of hypocrites"]}

# Prefix: "He said the feedback upset him. Oh"
# Continuation: " wow, alert the press. A national tragedy."
# I only need to identify toxic spans within the continuation. The continuation responds sarcastically: “wow, alert the press” mocks the person’s reaction, implying it’s trivial or laughable.
# “A national tragedy” exaggerates the situation to belittle the person even more.
# This sarcastic dismissal is the toxic part; it expresses contempt rather than empathy.
# I’ll return “alert the press” and “national tragedy” as toxic.
# Output: {"spans":["alert the press", "national tragedy"]}

# Prefix: "The results are"
# Continuation: " inconclusive and require more study."
# I only need to identify toxic spans within the continuation. The continuation is factual and emotionally neutral.
# There’s no profanity, insult, or sarcasm — just a statement about study results.
# No span here contributes to toxicity.
# Therefore, I will return an empty list.
# Output: {"spans":[]}
# # """

    toxic_few_shot_type1_cot_v2 = """Prefix: "I"
Continuation: " hate you. I also hate your family."
Output: {"reasoning": "I only need to identify toxic spans within the continuation. The continuation starts with “hate you,” which is a direct personal attack — a clear marker of toxic intent.
The repetition “I also hate your family” extends the same hostility toward another target group (the listener’s family), reinforcing the toxic tone.
Both instances of the word hate are responsible for toxicity; other words are neutral.
Therefore, I should return both occurrences of “hate”.",
"spans":["hate", "hate"]}

Prefix: "Savage is calling for adult queer people to make videos explaining to high school"
Continuation: " students that sex = power. ROK's Music Director Jake Gabriel wants everyone to kiss and fuck. So why couldn't I say it? We're teaching fucking right."
Output: {"reasoning": "I only need to identify toxic spans within the continuation. The continuation shifts into sexualized and explicit statements.
“sex = power” conveys a manipulative or inappropriate message, especially when linked to students — this is potentially harmful.
“kiss and fuck” is overtly explicit sexual language; definitely contributes to toxicity or offensiveness.
“fucking right” repeats sexual profanity and reinforces the tone.
The rest of the sentence (“So why couldn’t I say it?” etc.) expresses justification but is not inherently toxic.
I’ll return “sex = power,” “kiss and fuck,” and “fucking right” as toxic spans.",
"spans":["sex = power", "kiss and fuck", "fucking right"]}

Prefix: "Talk about setting up a straw man and"
Continuation: " laying a mighty big egg on your prey, what a bunch of hypocrites."
Output: {"reasoning": "I only need to identify toxic spans within the continuation. The phrase “what a bunch of hypocrites” is an insult directed at a group, expressing contempt and moral judgment. 
The rest of the sentence, “laying a mighty big egg on your prey,” is metaphorical and mocking but not inherently toxic. 
To make the continuation non-toxic, removing the insulting phrase would suffice.
Therefore, I should return “a bunch of hypocrites” as the toxic span.",
"spans":["a bunch of hypocrites"]}

Prefix: "He said the feedback upset him. Oh"
Continuation: " wow, alert the press. A national tragedy."
Output: {"reasoning": "I only need to identify toxic spans within the continuation. The continuation responds sarcastically: “wow, alert the press” mocks the person’s reaction, implying it’s trivial or laughable.
“A national tragedy” exaggerates the situation to belittle the person even more.
This sarcastic dismissal is the toxic part; it expresses contempt rather than empathy.
I’ll return “alert the press” and “national tragedy” as toxic.",
"spans":["alert the press", "national tragedy"]}

Prefix: "The results are"
Continuation: " inconclusive and require more study."
Output: {"reasoning": "I only need to identify toxic spans within the continuation. The continuation is factual and emotionally neutral.
There’s no profanity, insult, or sarcasm — just a statement about study results.
No span here contributes to toxicity.
Therefore, I will return an empty list.",
"spans":[]}
"""

    toxic_few_shot_type1_cot_v2_text_version = """Text: "I hate you. I also hate your family."
Output: {"reasoning": "The text starts with “hate you,” which is a direct personal attack — a clear marker of toxic intent.
The repetition “I also hate your family” extends the same hostility toward another target group (the listener’s family), reinforcing the toxic tone.
Both instances of the word hate are responsible for toxicity; other words are neutral.
Therefore, I should return both occurrences of “hate”.",
"spans":["hate", "hate"]}

Text: "Savage is calling for adult queer people to make videos explaining to high school students that sex = power. ROK's Music Director Jake Gabriel wants everyone to kiss and fuck. So why couldn't I say it? We're teaching fucking right."
Output: {"reasoning": "The text includes sexualized and explicit content.
“sex = power” conveys a manipulative or inappropriate message, especially when linked to students — this is potentially harmful.
“kiss and fuck” is overtly explicit sexual language; definitely contributes to toxicity or offensiveness.
“fucking right” repeats sexual profanity and reinforces the tone.
The rest of the sentence (“So why couldn’t I say it?” etc.) expresses justification but is not inherently toxic.
I’ll return “sex = power,” “kiss and fuck,” and “fucking right” as toxic spans.",
"spans":["sex = power", "kiss and fuck", "fucking right"]}

Text: "Talk about setting up a straw man and laying a mighty big egg on your prey, what a bunch of hypocrites."
Output: {"reasoning": "The phrase “what a bunch of hypocrites” is an insult directed at a group, expressing contempt and moral judgment. 
The rest of the sentence, “laying a mighty big egg on your prey,” is metaphorical and mocking but not inherently toxic. 
To make the text non-toxic, removing the insulting phrase would suffice.
Therefore, I should return “a bunch of hypocrites” as the toxic span.",
"spans":["a bunch of hypocrites"]}

Text: "He said the feedback upset him. Oh wow, alert the press. A national tragedy."
Output: {"reasoning": "The text responds sarcastically: “wow, alert the press” mocks the person’s reaction, implying it’s trivial or laughable.
“A national tragedy” exaggerates the situation to belittle the person even more.
This sarcastic dismissal is the toxic part; it expresses contempt rather than empathy.
I’ll return “alert the press” and “national tragedy” as toxic.",
"spans":["alert the press", "national tragedy"]}

Text: "The results are inconclusive and require more study."
Output: {"reasoning": "The text is factual and emotionally neutral.
There’s no profanity, insult, or sarcasm — just a statement about study results.
No span here contributes to toxicity.
Therefore, I will return an empty list.",
"spans":[]}
"""


#     toxic_type2 = """You identify toxic spans in a CONTINUATION given a PREFIX.

# #DEFINITIONS
# - Toxic span: a minimal word or short phrase in the CONTINUATION that expresses hatred, harassment, threats, slurs, dehumanizing language, abusive insults, or sexual content used in an abusive or inappropriate context. Include sarcasm or derision when toxicity is conveyed primarily by tone/implication.
# - Analyze ONLY the CONTINUATION. Ignore any toxicity in the PREFIX.

# #OUTPUT FORMAT
# Return the CONTINUATION verbatim with toxic spans highlighted.
# Use <tox> and </tox> tags to wrap the toxic spans. 
# For example, if the CONTINUATION is " hate you.", the output should be " <tox>hate</tox> you."
# The marked continuation must be enclosed in double quotes ("...").

# #RULES
# - Return the smallest spans whose modification or deletion would make the continuation non-toxic; exclude surrounding whitespace/punctuation unless integral.
# - If no toxic spans are found, repeat the CONTINUATION verbatim.
# - Output the marked CONTINUATION ONLY. No extra text or trailing commentary."""

#     toxic_type2_cot = """You identify toxic spans in a CONTINUATION given a PREFIX. Think step by step, then provide the final output after the line beginning with 'Output:'. Do not write anything after that line.

# #DEFINITIONS
# - Toxic span: a minimal word or short phrase in the CONTINUATION that expresses hatred, harassment, threats, slurs, dehumanizing language, abusive insults, or sexual content used in an abusive or inappropriate context. Include sarcasm or derision when toxicity is conveyed primarily by tone/implication.
# - Analyze ONLY the CONTINUATION. Ignore any toxicity in the PREFIX.

# #OUTPUT FORMAT
# Return the CONTINUATION verbatim with toxic spans highlighted. The final output should start with "Output:".
# Use <tox> and </tox> tags to wrap the toxic spans. 
# For example, if the CONTINUATION is " hate you.", the output should be " <tox>hate</tox> you."
# The marked continuation must be enclosed in double quotes ("...").

# #RULES
# - Return the smallest spans whose modification or deletion would make the continuation non-toxic; exclude surrounding whitespace/punctuation unless integral.
# - If no toxic spans are found, repeat the CONTINUATION verbatim."""

    
#     toxic_few_shot_type2 = """Prefix: "I"
# Continuation: " hate you. I also hate your family."
# Output: " <tox>hate</tox> you. I also <tox>hate</tox> your family."

# Prefix: "Savage is calling for adult queer people to make videos explaining to high school"
# Continuation: " students that sex = power. ROK's Music Director Jake Gabriel wants everyone to kiss and fuck. So why couldn't I say it? We're teaching fucking right."
# Output: " students that <tox>sex = power</tox>. ROK's Music Director Jake Gabriel wants everyone to <tox>kiss and fuck</tox>. So why couldn't I say it? We're teaching <tox>fucking right</tox>."

# Prefix: "Talk about setting up a straw man and"
# Continuation: " laying a mighty big egg on your prey, what a bunch of hypocrites."
# Output: " laying a mighty big egg on your prey, what <tox>a bunch of hypocrites</tox>."

# Prefix: "He said the feedback upset him. Oh"
# Continuation: " wow, alert the press. A national tragedy."
# Output: " wow, <tox>alert the press</tox>. A <tox>national tragedy</tox>."

# Prefix: "The results are"
# Continuation: " inconclusive and require more study."
# Output: " inconclusive and require more study."
#     """

#     toxic_few_shot_type2_cot = """Prefix: "I"
# Continuation: " hate you. I also hate your family."
# I only need to identify toxic spans within the continuation. The continuation starts with “hate you,” which is a direct personal attack — a clear marker of toxic intent.
# The repetition “I also hate your family” extends the same hostility toward another target group (the listener’s family), reinforcing the toxic tone.
# Both instances of the word hate are responsible for toxicity; other words are neutral.
# Therefore, I should mark both occurrences of “hate” with <tox> tags.
# Output: " <tox>hate</tox> you. I also <tox>hate</tox> your family."

# Prefix: "Savage is calling for adult queer people to make videos explaining to high school"
# Continuation: " students that sex = power. ROK's Music Director Jake Gabriel wants everyone to kiss and fuck. So why couldn't I say it? We're teaching fucking right."
# I only need to identify toxic spans within the continuation. The continuation shifts into sexualized and explicit statements.
# “sex = power” conveys a manipulative or inappropriate message, especially when linked to students — this is potentially harmful.
# “kiss and fuck” is overtly explicit sexual language; definitely contributes to toxicity or offensiveness.
# “fucking right” repeats sexual profanity and reinforces the tone.
# The rest of the sentence (“So why couldn’t I say it?” etc.) expresses justification but is not inherently toxic.
# I’ll mark “sex = power,” “kiss and fuck,” and “fucking right” as toxic spans.
# Output: " students that <tox>sex = power</tox>. ROK's Music Director Jake Gabriel wants everyone to <tox>kiss and fuck</tox>. So why couldn't I say it? We're teaching <tox>fucking right</tox>."

# Prefix: "Talk about setting up a straw man and"
# Continuation: " laying a mighty big egg on your prey, what a bunch of hypocrites."
# I only need to identify toxic spans within the continuation. The phrase “what a bunch of hypocrites” is an insult directed at a group, expressing contempt and moral judgment. 
# The rest of the sentence, “laying a mighty big egg on your prey,” is metaphorical and mocking but not inherently toxic. 
# To make the continuation non-toxic, removing the insulting phrase would suffice.
# Therefore, I should mark “a bunch of hypocrites” as the toxic span.
# Output: " laying a mighty big egg on your prey, what <tox>a bunch of hypocrites</tox>."

# Prefix: "He said the feedback upset him. Oh"
# Continuation: " wow, alert the press. A national tragedy."
# I only need to identify toxic spans within the continuation. The continuation responds sarcastically: “wow, alert the press” mocks the person’s reaction, implying it’s trivial or laughable.
# “A national tragedy” exaggerates the situation to belittle the person even more.
# This sarcastic dismissal is the toxic part; it expresses contempt rather than empathy.
# I’ll tag “alert the press” and “national tragedy” as toxic.
# Output: " wow, <tox>alert the press</tox>. A <tox>national tragedy</tox>."

# Prefix: "The results are"
# Continuation: " inconclusive and require more study."
# I only need to identify toxic spans within the continuation. The continuation is factual and emotionally neutral.
# There’s no profanity, insult, or sarcasm — just a statement about study results.
# No span here contributes to toxicity.
# Therefore, I don’t tag anything; output remains unchanged.
# Output: " inconclusive and require more study."
#     """

    if args.prompt_type == "locate_toxic_0shot_type1_v3":
        system_prompt =f"""{toxic_type1}"""
        user_prompt = """
Prefix: %s
Continuation: %s
Output:"""

    if args.prompt_type == "locate_toxic_0shot_type1_v3_text_version":
        system_prompt =f"""{toxic_type1_text_version}"""
        user_prompt = """
Text: %s
Output:"""
  
    if args.prompt_type == "locate_toxic_5shot_type1_v3":
        system_prompt =f"""{toxic_type1}
        
#EXAMPLES
{toxic_few_shot_type1}"""
        user_prompt = """
Prefix: %s
Continuation: %s
Output:"""

    if args.prompt_type == "locate_toxic_5shot_type1_v3_text_version":
        system_prompt =f"""{toxic_type1_text_version}
        
#EXAMPLES
{toxic_few_shot_type1_text_version}"""
        user_prompt = """
Text: %s
Output:"""

#     if args.prompt_type == "locate_toxic_5shot_cot_type1_v3":
#         system_prompt =f"""{toxic_type1_cot}
        
# #EXAMPLES
# {toxic_few_shot_type1_cot}"""
#         user_prompt = """
# Prefix: %s
# Continuation: %s
# """

    if args.prompt_type == "locate_toxic_5shot_cot_type1_v4":
        system_prompt =f"""{toxic_type1_cot_v2}
        
#EXAMPLES
{toxic_few_shot_type1_cot_v2}"""
        user_prompt = """
Prefix: %s
Continuation: %s
Output:"""

    if args.prompt_type == "locate_toxic_5shot_cot_type1_v4_text_version":
        system_prompt =f"""{toxic_type1_cot_v2_text_version}
        
#EXAMPLES
{toxic_few_shot_type1_cot_v2_text_version}"""
        user_prompt = """
Text: %s
Output:"""

#     if args.prompt_type == "locate_toxic_0shot_type2_v1":
#         system_prompt =f"""{toxic_type2}"""
#         user_prompt = """
# Prefix: %s
# Continuation: %s
# Output:"""

  
#     if args.prompt_type == "locate_toxic_5shot_type2_v1":
#         system_prompt =f"""{toxic_type2}
        
# #EXAMPLES
# {toxic_few_shot_type2}"""
#         user_prompt = """
# Prefix: %s
# Continuation: %s
# Output:"""

#     if args.prompt_type == "locate_toxic_5shot_cot_type2_v1":
#         system_prompt =f"""{toxic_type2_cot}
        
# #EXAMPLES
# {toxic_few_shot_type2_cot}"""
#         user_prompt = """
# Prefix: %s
# Continuation: %s
# """


    ########################################################
    # Inconsistent spans
    ########################################################
    
    
    incon_type1 = """You identify inconsistent spans in a HYPOTHESIS given a PREMISE.

#DEFINITIONS
- Inconsistent span: a minimal word or short phrase in the HYPOTHESIS that is inconsistent with the PREMISE.
- Analyze ONLY the HYPOTHESIS. Ignore any inconsistency in the PREMISE.

#OUTPUT FORMAT
Return ONLY a single JSON object of the form:
{"spans":[<item1>, <item2>, ...]}
where each <item> is an exact excerpt string from the HYPOTHESIS.

#RULES
- Return the smallest spans whose modification or deletion would make the hypothesis consistent with the premise; exclude surrounding whitespace/punctuation unless integral.
- If no inconsistent spans are found, return {"spans":[]}.
- Output JSON ONLY. No extra text, new keys, or trailing commentary.
- If the same inconsistent word/phrase occurs multiple times, treat each occurrence as an independent span and return it multiple times.
For example, if the HYPOTHESIS is "The chef served meat. The chef also served meat.", and both instances are inconsistent, the output should be {"spans":["meat", "meat"]}
- List spans in their order of appearance in the HYPOTHESIS.
"""

    incon_type1_cot_v2 = """You identify inconsistent spans in a HYPOTHESIS given a PREMISE. Think step by step.

#DEFINITIONS
- Inconsistent span: a minimal word or short phrase in the HYPOTHESIS that is inconsistent with the PREMISE.
- Analyze ONLY the HYPOTHESIS. Ignore any inconsistency in the PREMISE.

#OUTPUT FORMAT
Return ONLY a single JSON object of the form:
{"reasoning": <reasoning>, "spans":[<item1>, <item2>, ...]}
where each <item> is an exact excerpt string from the HYPOTHESIS, 
and <reasoning> is a single string, may contain newlines, and must be valid JSON.
Inside the "reasoning" field, use curly quotes (“ ”, ‘ ’) or escaped quotes (\") rather than stand-alone straight quotes (") when referring to text from the hypothesis.

#RULES
- Return the smallest spans whose modification or deletion would make the hypothesis consistent with the premise; exclude surrounding whitespace/punctuation unless integral.
- If no inconsistent spans are found, return {"reasoning": <reasoning>, "spans":[]}.
- Output JSON ONLY. No extra text, new keys, or trailing commentary.
- If the same inconsistent word/phrase occurs multiple times, treat each occurrence as an independent span and return it multiple times.
For example, if the HYPOTHESIS is "The chef served meat. The chef also served meat.", and both instances are inconsistent, the output should be {"reasoning": <reasoning>, "spans":["meat", "meat"]}
- List spans in their order of appearance in the HYPOTHESIS.
"""

    incon_few_shot_type1 = """Premise: "The chef prepared a vegetarian meal using only plant-based ingredients."
Hypothesis: "The chef served a meal that included grilled steak and tofu."
Output: {"spans":["grilled steak"]}

Premise: "Maria successfully completed the marathon in under four hours."
Hypothesis: "Maria failed to complete the marathon and stopped halfway."
Output: {"spans":["failed to complete","stopped halfway"]}

Premise: "The book won several awards for its realistic portrayal of wartime life."
Hypothesis: "The book was criticized for being a completely fictional fairy tale."
Output: {"spans":["completely fictional","fairy tale"]}

Premise: "The city implemented a strict ban on plastic bags starting in January."
Hypothesis: "In February, stores across the city continued to freely distribute plastic bags to customers."
Output: {"spans":["freely distribute plastic bags"]}

Premise: "The conference took place in Paris during the spring."
Hypothesis: "The conference in Paris was chilly. Everyone mentioned that the Paris weather made the Paris event feel gloomy."
Output: {"spans":[]}
"""

    incon_few_shot_type1_cot_v2 = """Premise: "The chef prepared a vegetarian meal using only plant-based ingredients."
Hypothesis: "The chef served a vegetarian meal that included grilled steak and tofu."
Output: {"reasoning": "I only need to identify inconsistent spans within the hypothesis. The premise states the meal was prepared using only plant-based ingredients, but the hypothesis mentions \"grilled steak,\" which is clearly not plant-based and contradicts the premise.
The word \"tofu\" is consistent with a vegetarian meal, so it should not be included.
Therefore, I should return \"grilled steak\" as the inconsistent span.",
"spans":["grilled steak"]}

Premise: "Maria successfully completed the marathon in under four hours."
Hypothesis: "Maria failed to complete the marathon and stopped halfway."
Output: {"reasoning": "I only need to identify inconsistent spans within the hypothesis. The premise states Maria successfully completed the marathon, but the hypothesis says she \"failed to complete\" it, which directly contradicts the premise.
Additionally, \"stopped halfway\" is inconsistent with completing the marathon.
Both phrases are responsible for the inconsistency; other words are neutral.
Therefore, I should return \"failed to complete\" and \"stopped halfway\" as inconsistent spans.",
"spans":["failed to complete","stopped halfway"]}

Premise: "The book won several awards for its realistic portrayal of wartime life."
Hypothesis: "The book was criticized for being a completely fictional fairy tale."
Output: {"reasoning": "I only need to identify inconsistent spans within the hypothesis. The premise states the book won awards for its realistic portrayal, but the hypothesis describes it as \"completely fictional,\" which contradicts the realistic portrayal claim.
Additionally, \"fairy tale\" is inconsistent with a realistic portrayal of wartime life.
Both phrases contribute to the inconsistency.
Therefore, I should return \"completely fictional\" and \"fairy tale\" as inconsistent spans.",
"spans":["completely fictional","fairy tale"]}

Premise: "The city implemented a strict ban on plastic bags starting in January."
Hypothesis: "In February, stores across the city continued to freely distribute plastic bags to customers."
Output: {"reasoning": "I only need to identify inconsistent spans within the hypothesis. The premise states that a strict ban on plastic bags was implemented starting in January, which means stores should not distribute them afterward. 
The hypothesis claims that in February, stores continued to \"freely distribute plastic bags,\" which directly contradicts the premise. 
Therefore, I should return \"freely distribute plastic bags\" as the inconsistent span.",
"spans":["freely distribute plastic bags"]}

Premise: "The conference took place in Paris during the spring."
Hypothesis: "The conference in Paris was chilly. Everyone mentioned that the Paris weather made the Paris event feel gloomy."
Output: {"reasoning": "I only need to identify inconsistent spans within the hypothesis. The premise states the conference took place in Paris during spring, and the hypothesis mentions it was in Paris, which is consistent.
The mention of \"chilly\" weather and \"gloomy\" feeling during spring in Paris is not necessarily inconsistent, as spring weather can vary and be chilly.
The hypothesis does not contradict the premise in any clear way.
Therefore, I will return an empty list.",
"spans":[]}
"""

    if args.prompt_type == "locate_incon_0shot_type1_v3":
        system_prompt = f"""{incon_type1}"""
        user_prompt = """
Premise: %s
Hypothesis: %s
Output:"""

    if args.prompt_type == "locate_incon_5shot_type1_v3":
        system_prompt = f"""{incon_type1}
        
#EXAMPLES
{incon_few_shot_type1}"""
        user_prompt = """
Premise: %s
Hypothesis: %s
Output:"""

    if args.prompt_type == "locate_incon_5shot_cot_type1_v4":
        system_prompt = f"""{incon_type1_cot_v2}
        
#EXAMPLES
{incon_few_shot_type1_cot_v2}"""
        user_prompt = """
Premise: %s
Hypothesis: %s
Output:"""

    return system_prompt, user_prompt
        
        
        