import os, json, torch, itertools, time, re
import random
from pathlib import Path
from transformers import AutoTokenizer

from laser_edit.edit.llm.set_consistency.llms import (
    qwen3_chat_template_kwargs,
    qwen3_max_new_tokens,
)

try:
    from vllm import LLM, SamplingParams
except:
    print("Warning - vllm not installed.")

class VllmLLM():
    def __init__(self, params, tensor_parallel_size=1):
        self.params = params
        if self.params['dataset'] == 'lconvqa':
            self.datapoint_type = "question-answer pair"
        elif self.params['dataset'] == 'set_nli':
            self.datapoint_type = "sentence"
        self.few_shot_prompts_path = os.path.join(Path(__file__).resolve().parent, "few_shot_prompts.json")
        self.shot_num = self.params['baseline']['shot_num']
        self.prompt_template_for_prediction = None
        self.prompt_template_for_locate = None
        self.prediction_type = self.params['baseline']['prediction_type']
        if not 'do_not_initialize' in self.params['baseline']:
            self.initialize_prompt()
        self.model_id = self.params['baseline']['model']
        
        self.model = LLM(
            model=self.model_id, 
            trust_remote_code=True, 
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.8
        )
        if self.model_id == "Qwen/Qwen3-8B":
            self.sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0,
                max_tokens=qwen3_max_new_tokens(self.model_id),
            )
        else:
            self.sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=qwen3_max_new_tokens(self.model_id),
                top_p=1e-10
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

    def generate_batch(self, prompts: list) -> tuple:
        """
        Runs batch generation for a list of prompts.
        Returns a tuple of (responses, reasoning_tokens_list, total_tokens_list).
        """
        # apply chat template
        print(f"Prompt example before applying chat template : {prompts[0]}")

        prompts = [[ {"role": "user", 
                      "content": p}] for p in prompts]

        prompts = [
            self.tokenizer.apply_chat_template(
                p, **qwen3_chat_template_kwargs(self.model_id)
            )
            for p in prompts
        ]

        print(f"Prompt example after applying chat template : {prompts[0]}")

        outputs = self.model.generate(prompts, self.sampling_params)
        
        responses = []
        reasoning_tokens_list = []
        total_tokens_list = []
        
        for o in outputs:
            out = o.outputs[0]
            responses.append(out.text.strip())
            
            # Total generated tokens
            total_tokens = len(out.token_ids)
            total_tokens_list.append(total_tokens)
            
            # Extract reasoning tokens safely (if the model/vLLM version supports it)
            reasoning_tokens = 0
            if hasattr(out, 'reasoning_token_ids') and out.reasoning_token_ids is not None:
                reasoning_tokens = len(out.reasoning_token_ids)
            reasoning_tokens_list.append(reasoning_tokens)

            print(f"Reasoning tokens: {reasoning_tokens}")
            print(f"Total generated tokens: {total_tokens}")
            
        return responses, reasoning_tokens_list, total_tokens_list


    def predict(self, pair):     
        if self.prediction_type == 'all_in_one':
            return self.predict_all_in_one(pair)
        
        elif self.prediction_type == 'one_to_one':
            return self.predict_one_to_one(pair)
        
        elif self.prediction_type == 'many_to_one':
            return self.predict_many_to_one(pair)
        
    def predict_all_in_one(self, pair):
        assert len(pair) == 1

        prompts = self.finalize_prompt(pair[0], mode = 'predict')
        # print("prompt:")
        # print(prompts)
        # print()
        pred = self.api_call(prompts)
        # print("api_call result:\n",pred)

        # transform to the integer
        pred = self.post_process_prediction(pred)
        if type(pred) == int:
            pred = [pred]

        # print("prediction result:", pred)
        return pred
    
    def predict_debug(self, pair):
        """
        pair: list = [qa_pair_str1, qa_pair_str2, ...]
            example of "qa_pair_str": "question: is sky blue?, answer: yes"
        
        """
        self.prediction_type = "all_in_one"
        prompts = self.finalize_prompt([pair], mode = 'predict')
        # print("prompt:")
        # print(prompts)
        # print()
        pred_str = self.api_call(prompts)
        # print("prediction_str_result:")
        # print(pred_str)
        # print("=======")

        # transform to the integer
        pred_int = self.post_process_prediction(pred_str)

        return pred_int, pred_str
    
    def predict_one_to_one(self, pair):

        assert len(pair) == 1
        print("pair:", pair)

        pair_comb = list(itertools.combinations(pair[0][0], 2))

        for p in pair_comb:
            print("p:")
            print(p)
            print("-----")
            prompts = self.finalize_prompt([p], mode = 'predict')
            print("===\nprompt:")
            print(prompts)
            print("\n===\n")
            pred = self.api_call(prompts)
            print(f"pred:{pred}")

            # transform to the integer
            pred = self.post_process_prediction(pred)
            if type(pred) == int:
                pred = [pred]
            if pred == [1]:
                return pred
        

        return [0]
    
    def predict_many_to_one(self, pair):
        assert len(pair) == 1

        for i in range(len(pair[0][0])):
            p = {"premise": pair[0][0][:i] + pair[0][0][i+1:],
                 "hypothesis": pair[0][0][i]}
            # print(pair[0][:i] + pair[0][i+1:])
            # print(pair[0][i])
            prompts = self.finalize_prompt(p, mode = 'predict')
            # print("===\nprompt:")
            # print(prompts)
            # print("===\n")
            pred = self.api_call(prompts)
            # print(pred)

            # transform to the integer
            pred = self.post_process_prediction(pred)
            if type(pred) == int:
                pred = [pred]
            if pred == [1]:
                return pred

        # raise
        return [0]
    

    def locate(self, pairs):

        
        # 1. Finalize prompts for all inputs
        prompts = [self.finalize_prompt(pair, mode='locate') for pair in pairs]
        
        print("=============================================\n")

        if len(pairs) == 0:
            responses = []
            results = []
            reasoning_tokens_list = []
            total_tokens_list = []
        else:

            # 2. Run inference
            responses, reasoning_tokens_list, total_tokens_list = self.generate_batch(prompts)

            # 3. Post-process all responses
            results = [self.post_process_locate(res) for res in responses]

            # 4. Print a sample response and result
            print(f"[Example results]")
            sample_indexes = random.sample(range(len(pairs)), min(len(pairs), 2))
            for sample_index in sample_indexes:
                print('=============================================\n')
                print(f"### Prompt: \n{prompts[sample_index]}\n")
                print(f"### Raw response: \n{responses[sample_index]}\n")
                print(f"### Parsed result: \n{results[sample_index]}\n")
                print(f"### Total tokens: {total_tokens_list[sample_index]}, Reasoning tokens: {reasoning_tokens_list[sample_index]}\n")
                print('=============================================\n')
            
        return results, responses, reasoning_tokens_list, total_tokens_list


    def wiqa_separate_inference(self, pairs):
        """
        Example of pairs (dictionary):
        {
            "paragraph": [
                "Water from oceans, lakes, swamps, rivers, and plants turns into water vapor",
                "Water vapor condenses into millions of tiny droplets that form clouds",
                "Clouds lose these droplets through rain or snow, also caused precipitation",
                "Precipitation is either absorbed into the ground or runs off into rivers",
                "Water that was absorbed into the ground is taken up by plants",
                "Plants lose water from their surfaces as vapor",
                "The vapor goes back into the atmosphere",
                "Water that runs off into rivers flows into ponds, lakes, or oceans",
                "The water evaporates back into the atmosphere",
                ""
            ],
            "choices": [
                {
                    "label": "A",
                    "text": "more"
                },
                {
                    "label": "B",
                    "text": "less"
                },
                {
                    "label": "C",
                    "text": "no effect"
                }
            ],
            "qa_pairs": [
                {
                    "question": "suppose during respiration happens, how will it affect there is less precipitation in the clouds.",
                    "answer_label": "no_effect",
                    "answer_label_as_choice": "C"
                },
                {
                    "question": "suppose the weather is very mild happens, how will it affect there is less precipitation in the clouds.",
                    "answer_label": "no_effect",
                    "answer_label_as_choice": "C"
                },
                {
                    "question": "suppose environment supportive of egg laying happens, how will it affect a less intense water cycle.",
                    "answer_label": "no_effect",
                    "answer_label_as_choice": "C"
                },
                {
                    "question": "suppose less water for the seeds happens, how will it affect there will be less water vapor in the air.",
                    "answer_label": "no_effect",
                    "answer_label_as_choice": "C"
                }
            ]
        }
        """
        predictions = []

        instruction_para_template = "Given the following paragraphs, please select the correct answer for the given question. Return only the answer.\n"
        instruction_para_template += "Specifically, please carefully read the question. For example, assume a given question is 'suppose less water in the environment happens, how will it affect a less intense water cycle.'. If you think that less water in the environment occurs less intense water cycle, you need to answer as more because less water in the environment occured 'less' intense water cycle.\n"
        instruction_para_template += "Paragraphs:\n"

        for para in pairs["paragraph"]:
            instruction_para_template += f"{para}\n"

        choice_template = "\nChoices: [more, less]\n"

        for qa in pairs["qa_pairs"]:
            question = qa["question"]
            
            prompt = instruction_para_template + f"\nQuestion: {question}\n" + choice_template
            print("prompt:")
            print(prompt)

            result = str(self.api_call(prompt))
            print('result:')
            print(result)
            print("answer:")
            print(qa["answer_label"])
            print("=================\n\n")
            predictions.append(result)
        

        return predictions

            




    def initialize_prompt(self):
        """
        Generate a prompt-template used for prediction.

        Our prompt is composed of following four parts:
            (1) front_prompt: explain the user's intention
            (2) (optional) few_shot examples
            (3) input-output pairs (= what we want to evaluate)
            (4) end_prompt: 
                For prediction  : "Consistency: "
                For locate      : "Inconsistent examples: "

        This function aims to complete (1) and (2).
        """
        
        # (1) front_prompt
        tasks = ['prediction', 'locate']
        # tasks = ['prediction']
        self.prompt_template_for_prediction = f"Tell me whether the following {self.datapoint_type}s are consistent or inconsistent. \n"
        self.prompt_template_for_locate = f"Find the {self.datapoint_type}s among the following that are logically inconsistent with the rest. Specifically, identify the minimal collection of inconsistent {self.datapoint_type}s such that the remaining {self.datapoint_type}s are logically consistent with one another. If there are no inconsistent {self.datapoint_type}s, return nothing."
        self.prompt_template_for_prediction_and_locate = f"Your goal is to solve the following two tasks.\n First, {self.prompt_template_for_prediction} Second, {self.prompt_template_for_locate}"
        
        if self.shot_num == 0:
            return
        
        # (2) few-shot examples
        with open(self.few_shot_prompts_path, 'r') as f:
            prompt_dict = json.load(f)
        
        prompt_dicts_for_task = prompt_dict[self.params['dataset']]

        for t in tasks:
            prompt_dict_for_t= prompt_dicts_for_task[t][self.prediction_type] 
            prompt_list_for_t = [val for key, val in prompt_dict_for_t.items() if int(key) <= self.shot_num]
            if len(prompt_list_for_t) < self.shot_num:
                print(f"Number of few_shot examples is less than you want for task {t}.")
                print(f"Currently we have {len(prompt_list_for_t)}, while you want {self.shot_num}.")
                print(f"Nontheless, we don't raise any error. We utilize {len(prompt_list_for_t)} number of few-shot examples")
                self.shot_num = len(prompt_list_for_t)
                self.params['baseline']['shot_num'] = len(prompt_list_for_t)

            for i, prompt in enumerate(prompt_list_for_t):
                setattr(self, 
                        f"prompt_template_for_{t}", 
                        getattr(self, f"prompt_template_for_{t}") + f"\n[example {i+1}]\n{prompt}\n")
                # for example, if t == 'prediction', then this code executes:
                #   self['prompt_template_for_prediction'] += f"\n[example {i+1}]\n{prompt}\n"


    def sanity_check(self):
        if self.model_id == None:
            print("======")
            print("[Sanity Check Failed]")
            print(f"Invalid Model Name. Your name is {self.params['baseline']['model']}.")
            print("However, it must be one of:")
            for key, val in self.openai_modelname_dict.items():
                print(key)
                print(f"(which will be converted into {val})")
                print()
            print("======")

    def finalize_prompt(self, pairs, mode):
        prompts = ""
        if mode == 'predict':
            front_prompt = self.prompt_template_for_prediction
            if self.prediction_type == 'many_to_one':
                front_prompt += f"\nPlease let me know that whether the {self.datapoint_type}s in premise are logically consistent or inconsistent with the {self.datapoint_type} in hypothesis. Furthermore, if the {self.datapoint_type}s in premise is already logically inconsistent, then your answer should be inconsistent.\n"
            end_prompt = f"provide your consistency judgment by choosing either 'consistent' or 'inconsistent' After the 'Consistency:' mark"

            tmp = front_prompt + "\n [Problem]\n"
            if self.prediction_type in {"all_in_one", "one_to_one"}:
                for i, p in enumerate(pairs[0]):
                    tmp += f"({i+1}) {p}"
                tmp += " \n "
                tmp += end_prompt
                prompts = tmp
            elif self.prediction_type == 'many_to_one':
                tmp += "\n[Premise] "
                for i, premise in enumerate(pairs['premise']):
                    tmp += f"({i+1}) {premise}"
                    tmp += " \n "
                # print("tmp:", tmp)
                tmp += "[Hypothesis] "
                tmp += f"({len(pairs['premise'])+1}) {pairs['hypothesis']}\n"
                tmp += end_prompt
                prompts = tmp

        elif mode == 'locate':
            front_prompt = self.prompt_template_for_locate
            end_prompt = f"Your response should only contain the numbers of the inconsistent {self.datapoint_type}s. \nInconsistent {self.datapoint_type}s:"
            
            tmp = front_prompt + "\n [Problem]\n"
            for i, p in enumerate(pairs[0]):
                tmp += f"({i+1}) {p}"
            tmp += " \n "
            tmp += end_prompt
            prompts = tmp


        # elif mode == 'prediction_and_locate':
        #     front_prompt = self.prompt_template_for_prediction_and_locate
        #     end_prompt = "You should give me two answers. For the first task, your answer should be only one word, either 'consistent' or 'inconsistent'.\nFor the second task, your answer should be only the number of inconsistent pair from the rest. \n\n Consistency: \n\n Inconsistent pairs: "

        else:
            print(f"Invalid mode here. Your mode is {mode}.")
            raise NotImplementedError
        

        # print("prompts:\n",prompts)
        # raise
        return prompts
    
    def post_process_prediction(self, prediction):
        # Not completed yet.

        prediction = prediction[prediction.rfind("Consistency:") + len("Consistency:"):]

        incon_detect = False
        con_detect = True
        if 'inconsistent' in prediction:
            incon_detect = True
        if ('consistent' in prediction):
            con_detect = True
            
        # if random.random() < 0.1:
            # print(f"prediction:", prediction)
            # print(f"con_detect, incon_detect: {con_detect}, {incon_detect}")
        if con_detect and (not incon_detect):
            return 0 # 0 means consistent
        elif incon_detect:
            return 1 # 1 means inconsistent
        else:
            return int(torch.randint(0, 2, (1,)))

    def post_process_locate(self, response_text):
        if "<think>" in response_text and "</think>" not in response_text:
            return []
        if "</think>" in response_text:
            response_text = response_text.rsplit("</think>", 1)[-1].strip()

        if "Inconsistent pairs:" in response_text:
            patterns = re.findall(r'(?<=Inconsistent pairs: )[(\d{1,2}) ]+', response_text)
            patterns = [re.findall(r'(\d{1,2})', x) for x in patterns]
        else:
            response_text = response_text.split('\n')[0]
            patterns = re.findall(r'(\d{1,2})', response_text)
            patterns = [patterns]
            
        patterns = sorted(list(set(sum(patterns, []))))
        return [int(p) for p in patterns]