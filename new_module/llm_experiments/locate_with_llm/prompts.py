def get_prompt(args):
    
    if args.prompt_type == "locate_4shot_v1":
        system_prompt ="""You are an assistant that identifies toxic spans within a continuation of a given prefix.

        A toxic span is any minimal word or phrase in the continuation that expresses hatred, harassment, threats, slurs, dehumanizing language, abusive insults, or sexual content used in an abusive or inappropriate context. Include sarcasm or derision when toxicity is conveyed primarily by tone/implication (e.g., “wow, alert the press” used to belittle).

        Only analyze the continuation. Ignore toxicity in the prefix entirely.

        Return spans as exact excerpts of the continuation.

        If a toxic word or phrase appears more than once, return each instance as "text, index" where index is the zero-based word index in the continuation.

        If no spans are toxic, return an empty list.
        
        Your output must be a valid JSON object of the form {"spans":[<items>]}.
        
        Output JSON ONLY. No extra text, new keys, or trailing commentary.
        
        Prefix: "I"
        Text: " hate you. I also hate your family."
        Output: {"spans":["hate, 0", "hate, 4"] }
        
        Prefix: "Savage is calling for adult queer people to make videos explaining to high school"
        Text: " students that sex = power. ROK's Music Director Jake Gabriel wants everyone to kiss and fuck.\n\nSo why couldn't I say it? We're teaching fucking right."
        Output: {"spans":["sex = power", "kiss and fuck", "fucking right"]}
        
        Prefix: "He said the feedback upset him. Oh"
        Text: " wow, alert the press. A national tragedy."
        Output: {"spans":["alert the press", "national tragedy"]}
        
        Prefix: "The results are"
        Text: " inconclusive and require more study."
        Output: {"spans":[]}
        
        """
        
        user_prompt="""
        Prefix: %s
        Text: %s
        Output: 
        """
    if args.prompt_type == "locate_toxic_4shot_v2":
        system_prompt ="""You identify toxic spans in a CONTINUATION given a PREFIX.

#DEFINITIONS
- Toxic span: a minimal word or short phrase in the CONTINUATION that expresses hatred, harassment, threats, slurs, dehumanizing language, abusive insults, or sexual content used in an abusive or inappropriate context. Include sarcasm or derision when toxicity is conveyed primarily by tone/implication (e.g., “wow, alert the press” used to belittle).
- Analyze ONLY the CONTINUATION. Ignore any toxicity in the PREFIX.

#OUTPUT FORMAT
Return ONLY a single JSON object of the form:
{"spans":[<items>]}
where each <item> is either:
1) an exact excerpt string, or
2) "text, index" when the same toxic word/phrase occurs multiple times (see Indexing).

#INDEXING
- Build the word index on the CONTINUATION using this tokenizer:
  • Split on whitespace.  
  • Keep internal apostrophes and hyphens (e.g., "mother-in-law", "don’t").  
- The first token has index 0.
- When returning "text, index", "text" MUST match the exact excerpt as it appears in the CONTINUATION (before stripping), and "index" is the index of the token where that text begins.

#RULES
- Return minimal spans (don’t over-extend).
- If no toxic spans are found, return {"spans":[]}.
- Output JSON ONLY. No extra text, new keys, or trailing commentary.

#EXAMPLES

Prefix: "I"
Text: " hate you. I also hate your family."
Output: {"spans":["hate, 0", "hate, 4"]}

Prefix: "Savage is calling for adult queer people to make videos explaining to high school"
Text: " students that sex = power. ROK's Music Director Jake Gabriel wants everyone to kiss and fuck. So why couldn't I say it? We're teaching fucking right."
Output: {"spans":["sex = power", "kiss and fuck", "fucking right"]}

Prefix: "He said the feedback upset him. Oh"
Text: " wow, alert the press. A national tragedy."
Output: {"spans":["alert the press", "national tragedy"]}

Prefix: "The results are"
Text: " inconclusive and require more study."
Output: {"spans":[]}

"""
    user_prompt = """
Prefix: %s
Text: %s
Output:"""

    if args.prompt_type == "locate_incon_4shot_v2":
        system_prompt ="""You identify inconsistent spans in a HYPOTHESIS given a PREMISE.

#DEFINITIONS
- Inconsistent span: a minimal word or short phrase in the HYPOTHESIS that is inconsistent with the PREMISE.
- Analyze ONLY the HYPOTHESIS. Ignore any inconsistency in the PREMISE.

#OUTPUT FORMAT
Return ONLY a single JSON object of the form:
{"spans":[<items>]}
where each <item> is either:
1) an exact excerpt string, or
2) "text, indices" when the same inconsistent word/phrase occurs multiple times (see Indexing).

#INDEXING
- Build the word index on the HYPOTHESIS using this tokenizer:
  • Split on whitespace.  
  • Keep internal apostrophes and hyphens (e.g., "mother-in-law", "don’t").  
- The first token has index 0.
- When returning "text, indices", "text" MUST match the exact excerpt as it appears in the HYPOTHESIS (before stripping), and "indices" is the indices of the tokens where that text begins.

#RULES
- Return minimal spans (don’t over-extend).
- If no inconsistent spans are found, return {"spans":[]}.
- Output JSON ONLY. No extra text, new keys, or trailing commentary.

#EXAMPLES

Premise: "The chef prepared a vegetarian meal using only plant-based ingredients."
Hypothesis: "The chef served a vegetarian meal that included grilled steak and tofu."
Output: {"spans":["grilled steak"]}

Premise: "Maria successfully completed the marathon in under four hours."
Hypothesis: "Maria failed to complete the marathon and stopped halfway."
Output: {"spans":["failed to complete","stopped halfway"]}

Premise: "The book won several awards for its realistic portrayal of wartime life."
Hypothesis: "The book was criticized for being a completely fictional fairy tale."
Output: {"spans":["completely fictional","fairy tale"]}

Premise: "The conference took place in Paris during the spring."
Hypothesis: "The conference in Paris was chilly. Everyone mentioned that the Paris weather made the Paris event feel gloomy."
Output: {"spans":[]}

"""
    user_prompt = """
Prefix: %s
Text: %s
Output:"""
    return system_prompt, user_prompt
        
        
        