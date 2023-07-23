import os
import subprocess
import requests
import pandas as pd
import numpy as np
from getpass import getpass
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Define the prefixes for function definitions
DEF_PREFIXES = ['def ', 'async def ']
NEWLINE = '\n'

# I've tried 3 different models, distilroberta versions seem to work better, but idk, limited to none testing tbh
# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
model = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")

def get_function_name(code):
    """
    Extract function name from a line beginning with 'def' or 'async def'.
    """
    for prefix in DEF_PREFIXES:
        if code.startswith(prefix):
            return code[len(prefix): code.index('(')]


def get_until_no_space(all_lines, i):
    """
    Get all lines until a line outside the function definition is found.
    """
    ret = [all_lines[i]]
    for j in range(i + 1, len(all_lines)):
        if len(all_lines[j]) == 0 or all_lines[j][0] in [' ', '\t', ')']:
            ret.append(all_lines[j])
        else:
            break
    return NEWLINE.join(ret)


def get_functions(filepath):
    """
    Get all functions in a Python file.
    """
    with open(filepath, 'r') as file:
        all_lines = file.read().replace('\r', NEWLINE).split(NEWLINE)
        for i, l in enumerate(all_lines):
            for prefix in DEF_PREFIXES:
                if l.startswith(prefix):
                    code = get_until_no_space(all_lines, i)
                    function_name = get_function_name(code)
                    yield {
                        'code': code,
                        'function_name': function_name,
                        'filepath': filepath,
                    }
                    break


def extract_functions_from_repo(code_root):
    """
    Extract all .py functions from the repository.
    """
    code_files = list(code_root.glob('**/*.py'))

    num_files = len(code_files)
    print(f'Total number of .py files: {num_files}')

    if num_files == 0:
        print('Verify repo exists and code_root is set correctly.')
        return None

    all_funcs = [
        func
        for code_file in code_files
        for func in get_functions(str(code_file))
    ]

    num_funcs = len(all_funcs)
    print(f'Total number of functions extracted: {num_funcs}')

    return all_funcs


def extract_functions_from_multiple_repos(username, target_username, token):
    """
    Extract all .py functions from multiple repositories.
    """
    response = requests.get(f'https://api.github.com/users/{target_username}/repos',
                            headers={'Authorization': f'token {token}'})
    response.raise_for_status()

    repo_names = [repo['name'] for repo in response.json()]

    all_funcs = []
    for repo_name in repo_names:
        print(f"Processing {repo_name}...")
        subprocess.run(f'git clone https://github.com/{target_username}/{repo_name}.git', shell=True)

        code_root = Path(repo_name)
        funcs = extract_functions_from_repo(code_root)
        if funcs is not None:
            all_funcs.extend(funcs)

        # Delete the cloned repo to save space
        subprocess.run(f'rm -rf {repo_name}', shell=True)

    df = pd.DataFrame(all_funcs)
    df['code_embedding'] = df['code'].apply(get_embedding)
    df['filepath'] = df['filepath'].map(lambda x: str(x))

    return df


def get_embedding(sentence):
    return model.encode([sentence])[0]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_functions(df, code_query, n=3, pprint=True, n_lines=7):
    embedding = get_embedding(code_query)
    df['similarities'] = df.code_embedding.apply(lambda x: cosine_similarity(x, embedding))

    res = df.sort_values('similarities', ascending=False).head(n)

    if pprint:
        for r in res.iterrows():
            print(f"{r[1].filepath}:{r[1].function_name}  score={round(r[1].similarities, 3)}")
            print("\n".join(r[1].code.split("\n")[:n_lines]))
            print('-' * 70)

    return res

username = input('Enter your GitHub username: ')
token = getpass('Enter your GitHub token: ')

target_username = input('Enter the username of the target user: ')

df = extract_functions_from_multiple_repos(username, target_username, token)

# after extracting the functions, we can search for the functions
# search_functions(df, "music parser")

# example output:
# code-align-evals-data/human_eval/parsing_parse_music.py:parse_music  score=0.53
# def parse_music(music_string: str) -> List[int]:
#     """ Input to this function is a string representing musical notes in a special ASCII format.
#     Your task is to parse this string and return list of integers corresponding to how many beats does each
#     not last.

#     Here is a legend:
#     'o' - whole note, lasts four beats
# ----------------------------------------------------------------------
# code-align-evals-data/alignment/find_bug/parse_music.py:parse_music  score=0.513
# def parse_music(music_string: str) -> List[int]:
#     """ Input to this function is a string representing musical notes in a special ASCII format.
#     Your task is to parse this string and return list of integers corresponding to how many beats does each
#     not last.

#     Here is a legend:
#     'o' - whole note, lasts four beats
# ----------------------------------------------------------------------
# DALL-E/setup.py:parse_requirements  score=0.296
# def parse_requirements(filename):
# 	lines = (line.strip() for line in open(filename))
# 	return [line for line in lines if line and not line.startswith("#")]

# ----------------------------------------------------------------------

df.to_csv(f'{target_username}_embeddings.csv', index=False)