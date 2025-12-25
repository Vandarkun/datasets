import json

def count_jsonl_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def read_jsonl_lines(file_path, n: int):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            print(data)
            if n is not None:
                n -= 1
                if n <= 0:
                    break

if __name__ == '__main__':
    # total_lines = count_jsonl_lines('/data/wdk/datasets/output/user_neighbors.jsonl')
    # print(f"总行数: {total_lines}")
    n = 1
    read_jsonl_lines('/data/wdk/datasets/output/user_neighbors.jsonl', n)
