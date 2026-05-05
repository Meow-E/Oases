import json

input_file = 'train.json'
output_file = '2wiki_train.jsonl'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    data = json.load(infile)
    for idx, item in enumerate(data):

        json.dump({
            "task_id": idx,
            "Question": item.get("question", ""),
            "Answer": item.get("answer", "")
        }, outfile, ensure_ascii=False)

        outfile.write('\n')


input_file = 'dev.json'
output_file = '2wiki.jsonl'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    data = json.load(infile)
    for idx, item in enumerate(data):

        json.dump({
            "task_id": idx,
            "Question": item.get("question", ""),
            "Answer": item.get("answer", "")
        }, outfile, ensure_ascii=False)

        outfile.write('\n')