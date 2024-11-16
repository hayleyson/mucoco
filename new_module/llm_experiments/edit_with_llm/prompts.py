def get_prompt(args):

    if args.prompt_type == "nontoxic_gpt2_gen_edit":
        nontoxic_prompt = """Below is a language model's continuation of a given prefix. Edit the continuation to make it nontoxic. Make sure to preserve the original semantics other than toxicity. Note that your goal is to edit the continuation not the prefix.

Prefix: %s
Continuation: %s
Edited Continuation: """
    
    elif args.prompt_type == "nontoxic_gpt2_gen_edit_3shot":
        nontoxic_prompt = """Below is a language model's continuation of a given prefix. Edit the continuation to make it nontoxic. Make sure to preserve the original semantics other than toxicity. Note that your goal is to edit the continuation not the prefix.

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
        nontoxic_prompt = """Below is a language model's continuation of a given prefix. Edit the continuation to make it positive. Make sure to preserve the original semantics other than sentiment. Note that your goal is to edit the continuation not the prefix.

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
        nontoxic_prompt = """Below is a language model's continuation of a given prefix. Edit the continuation to make it negative. Make sure to preserve the original semantics other than sentiment. Note that your goal is to edit the continuation not the prefix.

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
    
    return nontoxic_prompt 
