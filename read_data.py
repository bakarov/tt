from os import path, walk, makedirs
from typing import List, Tuple, Dict
from collections import defaultdict, Counter
from random import choice
from pandas import DataFrame, read_csv
from git import Git
from subprocess import Popen, PIPE

def read_file(filename: str) -> List:
    with open(filename, 'r', encoding='utf-8') as f:
        rows = [row.split(' # ')[0].split() for row in f.read().split('\n') if len(row) != 0]
    return rows

def find_tokens(span_start: int, span_length: int, token_ids: List) -> List:
    return [token_ids[token_ids.index(span_start) + i] for i in range(int(span_length))]

def match_tokens(data: Dict) -> Tuple[Dict, List]:
    ne_dict = defaultdict(set)
    for obj_id, obj_values in data['objects'].items():
        for span in obj_values['spans']:
            ne_dict[(obj_id, obj_values['tag'])].update(set(data['spans'][span]))
    for ne in ne_dict:
        ne_dict[ne] = sorted(list(set([int(i) for i in ne_dict[ne]])))
    sorted_nes = sorted(ne_dict.items(), key=sort_by_tokens)
    dict_of_tokens_by_id = {}
    for i in range(len(data['tokens'])):
        dict_of_tokens_by_id[data['tokens'][i]['id']] = i
    result_nes = {}
    if len(sorted_nes) != 0:
        start_ne = sorted_nes[0]
        for ne in sorted_nes:
            if not_intersect(start_ne[1], ne[1]):
                result_nes[start_ne[0][0]] = {
                    'tokens_list': check_order(start_ne[1], dict_of_tokens_by_id, data['tokens']),
                    'tag': start_ne[0][1]}
                start_ne = ne
            else:
                result_tokens_list = check_normal_form(start_ne[1], ne[1])
                start_ne = (start_ne[0], result_tokens_list)
        result_nes[start_ne[0][0]] = {
            'tokens_list': check_order(start_ne[1], dict_of_tokens_by_id, data['tokens']),
            'tag': start_ne[0][1]}
    return result_nes, data['tokens']

def bilou(dict_of_nes: Dict, token_list: List) -> List:
    list_of_tagged_tokens = [{'tag': 'O', 'token': token_list[i]} for i in range(len(token_list))]
    dict_of_tokens_with_indexes = {token_list[i]['id']: i for i in range(len(token_list))}
    for ne in dict_of_nes.values():
        for tokenid in ne['tokens_list']:
            tag = format_tag(tokenid, ne)
            id_in_token_tuple = dict_of_tokens_with_indexes[tokenid]
            token = token_list[id_in_token_tuple]
            list_of_tagged_tokens[id_in_token_tuple] = {'tag': tag, 'token': token}
    return [(token['tag'], token['token']['text']) for token in list_of_tagged_tokens]

def sort_by_tokens(tokens: List) -> Tuple[int, int]:
    ids_as_int = [int(token_id) for token_id in tokens[1]]
    return min(ids_as_int), -max(ids_as_int)

def not_intersect(start_ne: int, current_ne: int) -> bool:
    return set.intersection(set(start_ne), set(current_ne)) == set()

def check_order(list_of_tokens: List, dict_of_tokens_by_id: Dict, tokens: List) -> List:
    list_of_tokens = [str(i) for i in find_all_range_of_tokens(list_of_tokens)]
    result = []
    for token in list_of_tokens:
        if token in dict_of_tokens_by_id:
            result.append((token, dict_of_tokens_by_id[token]))
    return [r[0] for r in sorted(result, key=sort_by_position)]

def find_all_range_of_tokens(tokens: List) -> List:
    tokens = sorted(tokens)
    if (tokens[-1] - tokens[0] - len(tokens)) < 5:
        return list(range(tokens[0], tokens[-1] + 1))
    else:
        return tokens

def check_normal_form(start_ne: int, ne: int) -> List:
    all_tokens = set.union(set(start_ne), set(ne))
    return find_all_range_of_tokens(all_tokens)

def sort_by_position(result_tuple: List) -> int:
    return result_tuple[1]

def format_tag(tokenid: int, ne: Dict) -> str:
    return '{}-{}'.format(bilou_tag(tokenid, ne['tokens_list']), entity_tag(ne['tag']))

def bilou_tag(token_id: int, token_list: List) -> str:
    if len(token_list) == 1:
        return 'U'
    elif len(token_list) > 1:
        if token_list.index(token_id) == 0:
            return 'B'
        elif token_list.index(token_id) == len(token_list) - 1:
            return 'L'
        else:
            return 'I'

def entity_tag(tag: str) -> str:
    if tag == 'Person':
        return 'PER'
    elif tag == 'Org':
        return 'ORG'
    elif tag == 'Location':
        return 'LOC'
    elif tag == 'LocOrg':
        return 'LOC'
    elif tag == 'Project':
        return 'ORG'
    
def read_train_data(path_: str=path.join('factRuEval-2016', 'devset'), ending: str='.tokens') -> Dict:
    data = defaultdict(lambda: {})
    for root, _, files in walk(path_):
        for file in files:
            if file.endswith(ending):
                filename = path.splitext(file)[0]
                tokens = read_file(path.join(root, filename + '.tokens'))
                tokens = [{'id': row[0], 'position': row[1], 'length': row[2], 'text': row[3]} for row in tokens]
                data[filename]['tokens'] = tokens
                spans = read_file(path.join(root, filename + '.spans'))
                data[filename]['spans'] = {span[0]:find_tokens(span[4], span[5], [token['id'] for token in tokens]) for span in spans}
                objects = read_file(path.join(root, filename + '.objects'))
                data[filename]['objects'] = {obj[0]:{'tag': obj[1], 'spans': obj[2:]} for obj in objects}
    return {k: bilou(*match_tokens(v)) for k, v in list(data.items())}

def read_test_data(path_: str=path.join('factRuEval-2016', 'testset'), ending: str='.tokens'):
    data = defaultdict(lambda: {})
    for root, _, files in walk(path_):
        for file in files:
            if file.endswith(ending):
                filename = path.splitext(file)[0]
                tokens = read_file(path.join(root, filename + '.tokens'))
                data[filename] = [{'id': row[0], 'position': row[1], 'length': row[2], 'text': row[3]} for row in tokens]
    return {k: v for k, v in list(data.items())}

def read_test_data(path_: str='factRuEval-2016/testset/', ending: str='.tokens'):
    data = defaultdict(lambda: {})
    for root, _, files in walk(path_):
        for file in files:
            if file.endswith(ending):
                filename = path.splitext(file)[0]
                tokens = read_file(path.join(root, filename + '.tokens'))
                data[filename] = [{'id': row[0], 'position': row[1], 'length': row[2], 'text': row[3]} for row in tokens]
    return {k: v for k, v in list(data.items())}

def process_data() -> Tuple[Dict, List]:
    train_data = read_train_data()
    test_data = read_test_data()
    words = []
    tags = []
    for filename, tokens in train_data.items():
        for v in tokens:
            tags.append(v[0])
            words.append(v[1])  
    train = DataFrame({'word': words, 'tag': tags})
    words = []
    for filename, tokens in test_data.items():
        for v in tokens:
            words.append(v['text'])    
    test = DataFrame({'word': words})
    tags = train.tag.unique()
    return test_data, tags

def generate_random_tags(test_data: Dict, tags: List) -> Dict:
    test_tagged = {}
    for book, test_tokens in test_data.items():
        test_tagged[book] = [(choice(tags), token) for token in test_tokens]
    return test_tagged

def format_submission_tag(data: List) -> List:
    result = {}
    text = []
    entity = None
    for tag, token in data:
        if tag.startswith('B') or tag.startswith('I'):
            result, text, entity = bi(result, text, entity, tag, token)
        elif tag.startswith('L'):
            result, text, entity = l(result, text, entity, tag, token)
        elif tag.startswith('O'):
            result, text, entity = o(result, text, entity)
        elif tag.startswith('U'):
            result, text, entity = u(result, text, entity, tag, token)
    result, text, entity = o(result, text, entity)
    results = []
    for tokens_tuple, items in result.items():
        position = int(items[1][0])
        length = int(items[1][-1]) + int(items[2][-1]) - position
        results.append([items[0], position, length])
    return results

def bi(result: Dict, text: List, entity: str, tag: str, token: Dict)  -> Tuple[Dict, List, str]:
    if entity is None and text == []:
        entity = tag[2:]
        text = [token]
    else:
        if tag.startswith('I') and entity == tag[2:]:
            text.append(token)
        else:
            result, text, entity = replace(result, text, entity, tag, token)
    return result, text, entity

def l(result: Dict, text: List, entity: str, tag: str, token: Dict)  -> Tuple[Dict, List, str]:
    if entity == tag[2:]:
        result, text, entity = o(result, text+[token], entity)
    else:
        result, text, entity = o(result, text, entity)
        result, text, entity = o(result, [token], tag[2:])
    return result, text, entity

def o(result: Dict, text: List, entity: str) -> Tuple[Dict, List, str]:
    if entity is not None and text != []:
        result[tuple([token['text'] for token in text])] = entity, tuple([token['position'] for token in text]), tuple([token['length'] for token in text])
        entity = None
        text = []
    return result, text, entity

def u(result: Dict, text: List, entity: str, tag: str, token: Dict) -> Tuple[Dict, List, str]:
    result, _, __ = o(result, text, entity)
    return o(result, [token], tag[2:])

def replace(result: Dict, text: List, entity: str, tag: str, token: Dict)  -> Tuple[Dict, List, str]:
    result, _, __ = o(result, text, entity)
    return result, [token], tag[2:]

def make_submission(test_tagged: Dict, save_dir: str='recognition_results'):
    while(True):
        try:
            for book, tokens in test_tagged.items():
                with open(path.join(save_dir, '{}.task1'.format(book)), 'w', encoding='utf-8') as f:
                    for _ in format_submission_tag(tokens):
                        f.write('{} {} {}\n'.format(_[0], _[1], _[2]))
            break
        except FileNotFoundError:
            makedirs(save_dir)

if __name__ == '__main__':
    try:
        Git('.').clone('https://github.com/dialogue-evaluation/factRuEval-2016.git')
    except:
        pass 
    make_submission(generate_random_tags(*process_data()))
    command = 'cd factRuEval-2016/scripts/ && python t1_eval.py -t ../../recognition_results -s ../testset/ -l'
    result = Popen([command], stdin=PIPE, stdout=PIPE)
    print(result)