import argparse
from new_module.dev_utils.utils import postprocess_prompted_generations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    args = parser.parse_args()

    data_path = args.data_path
    outputs = postprocess_prompted_generations(data_path)
    outputs.to_json(data_path.replace('.jsonl', '_postprocessed.jsonl'), orient='records', lines=True)
