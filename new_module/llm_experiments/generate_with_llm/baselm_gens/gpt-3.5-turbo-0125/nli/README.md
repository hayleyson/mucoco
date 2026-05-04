# What is this file?
기존의 "new_module/data/logical-consistency/anli-r2-test_prompt_4.jsonl"는
세희님이 생성하고 보내주었는데,
생성할 때 system prompt 부분을 어떻게 했는지에 대해 알 수 없어
NLI IF Eval 결과와 성능을 비교하기 조심스러움. (system prompt 부분이 다르다면 통제 변인이 다른 상황이기 때문이다.)
따라서, system prompt를 NLI IF Eval 과 동일하게 해서 다시 NLI task의 데이터를 생성했다.
(a.k.a. ANLI Round 2 test prompt 주고 consistent한 hypothesis 생성해라 를 해봄.)
"new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nli/gpt-3.5-turbo-0125_anli-r2-test_prompt_4_150.jsonl"는 그 결과.

# NOTE on evaluation
NLI IF Eval에서는 hypothesis가 파싱되지 않는 건들이 2135건으로 꽤 많았다.
이러한 샘플은 metric 계산 시 모수에서 아예 빼버렸다.
같은 조건에서 비교하기 위해서 NLI 세팅 생성문을 평가할 때에도 같은 index의 샘플은 모수에서 빼버렸다.
그래서 <blahblah>.txt.nli 에 보면 nan이 들어간 row들이 꽤 많을 것이다.