import json
import argparse


def convert(input_path, output_path, contexts):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    out = []
    for item in data:
        question = item['question']
        paragraphs = item['paragraphs']  # 四個 index
        # 找正確答案是第幾個選項
        try:
          label = paragraphs.index(item['relevant'])

          # 對應選項文字
          endings = [contexts[idx] for idx in paragraphs]
          # 組裝
          new_item = {
              'id': item['id'],
              'question': question,
              'ending0': endings[0],
              'ending1': endings[1],
              'ending2': endings[2],
              'ending3': endings[3],
              'label': label
          }
          out.append(new_item)

        except Exception as e:
          # 對應選項文字
          endings = [contexts[idx] for idx in paragraphs]
          # 組裝
          new_item = {
              'id': item['id'],
              'question': question,
              'ending0': endings[0],
              'ending1': endings[1],
              'ending2': endings[2],
              'ending3': endings[3],
          }
          out.append(new_item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Step 1 MC Preprocessing")
    parser.add_argument(
            "--context_file",
            type=str,
            default="data/context.json",
            help="context json")
    parser.add_argument(
            "--test_file",
            type=str,
            default="data/test.json",
            help="test input json"
            )
    parser.add_argument(
            "--output_file",
            type=str,
            default="data/test_step1.json",
            help="test output json"
            )

    args = parser.parse_args()


    with open(args.context_file, 'r', encoding='utf-8') as f:
        contexts = json.load(f)

    convert(args.test_file, args.output_file, contexts)

if __name__ == "__main__":
    main()

###################
#convert('data/train.json', 'data/train_step1.json', contexts)
#convert('data/valid.json', 'data/valid_step1.json', contexts)
#convert('data/test.json', 'data/test_step1.json', contexts)
