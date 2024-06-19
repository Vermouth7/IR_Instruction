import json
import os
import random
import re
import time
from http import HTTPStatus

import dashscope
from openai import OpenAI

queries = [
    "How do you make a travel plan for a two-week trip?",
    "What steps do you take to plan a successful project launch for a new product?",
    "How do you create a study plan for preparing for final exams in college?",
    "What is your approach to planning a weekly menu and grocery shopping list?",
    "What are the key elements of planning a birthday party for 50 guests?",
    "How do you organize a moving plan for relocating to a new city?",
    "How do you formulate a fitness plan to train for a marathon?",
    "What is your strategy for planning a home renovation project?",
    "How do you create a financial plan for saving for a down payment on a house?",
    "How do you structure a plan to learn a new language?"
]

def extract_number(input_string):
    match = re.search(r'\d+', input_string)
    if match:
        return int(match.group())
    else:
        return None

def write_to_jsonl(content, file_path):
    # Determine the starting id
    start_id = 0
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if lines:
                last_line = json.loads(lines[-1])
                start_id = extract_number(last_line['_id']) + 1

    # Write the strings to the JSONL file
    with open(file_path, 'a') as f:
        for i, text in enumerate(content, start=start_id):
            record = {"_id": i, "title": "", "text": text}
            f.write(json.dumps(record) + '\n')

def call_with_messages_qwen(query):
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 
                 'content': query}]
# multi-round talk
    response = dashscope.Generation.call(
        model='qwen-max',
        messages=messages,
        result_format='message',  
        max_tokens=2000,
    )
    if response.status_code == HTTPStatus.OK:
        return response['output']['choices'][0]['message']['content'], response.usage['output_tokens']
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
        return None,None

def tokenizer(content):
    response = dashscope.Tokenization.call(
        model='qwen-max',
        messages=[{'role': 'user', 'content': content}],
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
    if response.status_code == HTTPStatus.OK:
        # print('Result is: %s' % response)
        return response.usage['input_tokens']
    else:
        print('Failed request_id: %s, status_code: %s, code: %s, message:%s' %
              (response.request_id, response.status_code, response.code,
               response.message))
        return None

def multi_round(query):
    result=list()
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 
                 'content': "I need you to make the appropriate plan based on the following query: {}. Please generate 5 relevant steps in detail and each of them should be more than 200 tokens in length. Each article must be preceded by a serial number surrounded by '[' ']'. It is required that each output article must be plain text without a title or formatting such as bolding".format(query)}]
    response = dashscope.Generation.call(model="qwen-max",
                               messages=messages,
                               # 将输出设置为"message"格式
                               result_format='message')
    if response.status_code == HTTPStatus.OK:
        content=response['output']['choices'][0]['message']['content']
        temp=re.split('\[\d{1,2}\]', content)
        temp=[s for s in temp if s]
        result.extend(temp)

        # 将assistant的回复添加到messages列表中
        messages.append({'role': response.output.choices[0]['message']['role'],
                         'content': response.output.choices[0]['message']['content']})
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
        # 如果响应失败，将最后一条user message从messages列表里删除，确保user/assistant消息交替出现
        messages = messages[:-1]
    # 将新一轮的user问题添加到messages列表中
    messages.append({'role': 'user', 'content': 'Please generate another 5 steps different from the previous content and follow the previous requirements.'})
    # 进行第二轮模型的响应
    response = dashscope.Generation.call(model="qwen-max",
                               messages=messages,
                               result_format='message',  # 将输出设置为"message"格式
                               )
    if response.status_code == HTTPStatus.OK:
        # print(response['output']['choices'][0]['message']['content'])
        content=response['output']['choices'][0]['message']['content']
        temp=re.split('\[\d{1,2}\]', content)
        temp=[s for s in temp if s]
        result.extend(temp)

        for i in range(len(result)):
            num_token=tokenizer(result[i])
            if num_token:
                while num_token<200:
                    new_doc,num_token=call_with_messages_qwen("Expand the following sentence to a length of no less than 200 tokens without changing the original meaning. "+result[i])
                    if new_doc != None:
                        result[i]=new_doc
        
        write_to_jsonl(content=result,file_path='corpus.jsonl')
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
        




# Function to read a jsonl file
def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def read_qrels(file_path):
    qrels = {}
    with open(file_path, 'r') as file:
        next(file)  # Skip the header
        for line in file:
            query_id, corpus_id, score = line.strip().split('\t')
            if query_id not in qrels:
                qrels[query_id] = []
            qrels[query_id].append((corpus_id, int(score)))
    return qrels

def create_dataset():
    queries_file = 'queries.jsonl'
    corpus_file = 'corpus.jsonl'
    qrels_file = 'qrels/test.tsv'
    output_file = 'my_dataset.jsonl'
    queries = read_jsonl(queries_file)
    corpus = read_jsonl(corpus_file)
    qrels = read_qrels(qrels_file)

    # Create a dictionary for fast corpus lookup
    corpus_dict = {str(extract_number(doc['_id'])): doc['text'] for doc in corpus}
    # print(corpus_dict)
    # Ensure we have enough unrelated corpus entries
    if len(corpus) < 20:
        raise ValueError("Not enough corpus entries to generate unrelated entries")

    # Create the new dataset
    with open(output_file, 'w') as file:
        for query in queries:
            query_id = str(query['_id'])
            query_text = query['text']
            
            # Get related corpus entries
            pos_corpus = qrels.get(query_id, [])
            pos_texts = [corpus_dict[str(extract_number(corpus_id))] for corpus_id, score in pos_corpus]
            pos_indexes = [int(extract_number(corpus_id)) for corpus_id, score in pos_corpus]
            pos_scores = [score for corpus_id, score in pos_corpus]
            
            # Select 10 random unrelated corpus entries
            unrelated_corpus_ids = list(set(corpus_dict.keys()) - set([str(idx) for idx in pos_indexes]))
            random_unrelated_ids = random.sample(unrelated_corpus_ids, 90)
            key_texts = pos_texts + [corpus_dict[uid] for uid in random_unrelated_ids]
            key_indexes = pos_indexes + [int(uid) for uid in random_unrelated_ids]
            
            # Create the dataset entry
            entry = {
                "query": query_text,
                "pos": pos_texts,
                "pos_index": pos_indexes,
                "pos_score": pos_scores,
                "query_id": extract_number(query['_id']),
                "key": key_texts,
                "key_index": key_indexes
            }
            
            # Write to the output file
            file.write(json.dumps(entry) + '\n')

    print(f'{output_file} has been created.')

def write_qrels():

    # File paths
    queries_file = 'queries1.jsonl'
    corpus_file = 'corpus1.jsonl'
    output_file = 'qrels.tsv'

    # Read queries and corpus files
    queries = read_jsonl(queries_file)
    corpus = read_jsonl(corpus_file)

    # Create qrels.tsv file
    with open(output_file, 'w') as file:
        # Write header
        file.write('query-id\tcorpus-id\tscore\n')
        
        # Write relevance data
        for query in queries:
            query_id = query['_id']
            for i in range(10):
                corpus_index = extract_number(query['_id']) * 10 + i
                if corpus_index < len(corpus):  # Ensure we don't go out of bounds
                    corpus_id = corpus[corpus_index]['_id']
                    file.write(f'{query_id}\t{corpus_id}\t1\n')

    print(f'{output_file} has been created.')


if __name__ == '__main__':
    # multi_round(queries)
    # for i in range(1,10):
    #     multi_round(queries[i])
    #     time.sleep(30)
    # create_dataset()
    # corpus=read_jsonl('./corpus.jsonl')
    # for i in corpus:
    #     content=i['text']
    #     i.pop('text')
    #     i["title"]=""
    #     i["text"]=content
    # # print(corpus)
    # with open('./corpus1.jsonl', 'w') as f:
    #     for i in corpus:
    #         f.write(json.dumps(i) + '\n')
    create_dataset()