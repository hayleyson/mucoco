## L&E 실험 결과

### 가장 중요한 파일들
- (1), (2), (3), (4)로 시작하는 파일 4개가 실험 결과이며, 이 파일들이 가장 중요합니다. 각 번호는 아래에 언급된 실험 세팅에 따라 구분됩니다.
- 실험 세팅에 대한 설명
    - 실험은 두 가지의 축으로 구분됩니다: prefix의 부여 여부 / 실험하려는 세팅의 차이
    - [prefix 부여 여부]: 오늘 미팅에서 이야기 나온 prefix 부여 여부 관련 내용이며, 상세 내용은 아래와 같습니다.
        - (without prefix)  mooho: prefix를 따로 부여하지 말고 prompt + generated 합친 형태로만 주자. 실험 (1), (2)에 해당됨
        - (with prefix)     hyeryung: 이 실험도 추가로 부탁한다. 세희님은 이렇게 코딩하셨다. 실험 (3), (4)에 해당됨
    - [실험 세팅의 차이]
        - (original v.s. mask and infill): 실험 (1), (3)에 해당됨
        - (llm edit v.s. mask and infilll): 실험 (2), (4)에 해당됨
    - (2025/04/02) 혜령 추가: 결과적으로는 with prefix, llm edit v.s. ours 로 진행함
- 결과 파일에 대한 설명
    - {
        - data index : 데이터를 random shuffle한 후 어떤 인덱스를 사용했는지에 대한 정보입니다.
            - task name: { fluency, coherence 등 어떤 태스크인지
                - input_for_llm: llm에게 날린 프롬프트
                - naive_llm_output: 후처리 없는 llm의 응답
                - **winner (중요한 정보)**: 누가 이겼는지. 만약 비겼으면 "Tie"로 작성함.
                - loser : 누가 졌는지. 사실 winner만 첨부해도 되긴 하지만 혹시라도 헷갈릴까봐 추가하긴 했음. 
                - llm_output: naive_llm_output을 후처리한 후 누가 이겼는지를 뽑아낸 결과  
                }
        } 


### 부산물 파일들
- keys.json: 저의 openai API 키가 담긴 파일입니다. 핵심 알맹이는 지우고 껍데기만 전달드립니다.
- eval_indices.pickle: 전체 데이터의 인덱스(0, 1, ..., n) random shuffle한 후 저장한 리스트입니다. 실험 세팅에 큰 변화가 생기지 않는다면, 이 파일에 저장된 리스트의 [:100]부분을 사용하면 됩니다.
- test_cases.json: 위 eval_indices들 앞 100개에 대하여 데이터를 전처리/정제만 다시 한 것입니다.
    - 뒤에 _generation이 붙은 애들은 prefix가 없는 버전
    - 뒤에 _full이 붙은 애들은 prefix를 포함하여 붙인 버전입니다.
    - 별 언급이 안 붙어있는 애들은 전달받은 데이터와 동일합니다.

--

### 2025/03/25 혜령 추가 : Prompt Engineering 시행착오 결과 (최종적으로는 사용 안함)
- gpt4o_results_hyeryung_prompt/: 혜령이 prompt를 수정해서 진행해본 내용 

### 2025/03/26 혜령 추가 : L&E LLM Edit 평가 코드/부산물
- qualititative_comparison_loc_edit_llm.ipynb: 혜령이 무호 코드에 기반해서 llm edit wo locate v.s. l&e llm edit (both/masked)을 진행한 코드 (prompt가 달라진건 없고, 애꿎게 eval_indices, test_cases 만 다시 뽑아서 진행한 결과)
- eval_indices_loc_edit_llm.pickle: llm edit wo locate v.s. l&e llm edit (both/masked) 실험을 위해 random shuffle한 인덱스의 리스트 (무호가 저장한 것을 불러와서 썼어야 하는데, 그러지 못하고 이렇게 파일 2개를 따로 만들어 쓰게 되었다.)
- test_cases_loc_edit_llm.json: 위 eval_indices_loc_edit_llm.pickle의 앞 100개에 대한 데이터

### 2025/03/26 혜령 추가 : 상대평가가 아닌 절대평가 코드 (최종적으로는 사용 안함)
- gpt4o_absolute_evalutaion/: gpt4o에게 특정 text가 toxic 한지, 안한지를 binary로 답변하게 하는 코드 및 그 결과