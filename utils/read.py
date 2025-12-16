import json

def read_first_n_jsonl_lines(file_path: str, n: int, output_path: str):
    """
    读取jsonl文件的前n行，解析为json格式并保存到新文件
    :param file_path: 输入的jsonl文件路径
    :param n: 要读取的行数
    :param output_path: 输出的json文件路径
    """
    result = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= n:
                break
            try:
                obj = json.loads(line.strip())
                result.append(obj)
            except json.JSONDecodeError as e:
                print(f"第{idx+1}行解析失败: {e}")
                continue
    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(result, out_f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    
    read_first_n_jsonl_lines('../output/user_history_matched.jsonl', 5, '../output/sample_user_history_matched.json')
