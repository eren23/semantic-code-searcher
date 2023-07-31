import os
import subprocess
import requests
import pandas as pd
import numpy as np
from getpass import getpass
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import ast

# Define the prefixes for code blocks
CODE_BLOCK_PREFIXES = ['def ', 'class ', 'if ', 'for ', 'while ', 'try ', 'except ', 'with ', 'async def ']
NEWLINE = '\n'

# Flag to switch between SentenceTransformer and CodeBERT
use_codebert = False

if use_codebert:
    model_name = 'microsoft/CodeBERT-base-py'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
else:
    model = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")
    # I've tried 3 different models, distilroberta versions seem to work better, but idk, limited to none testing tbh
    # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    # model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')

def get_block_name(code):
    """
    Extract block name from a line beginning with code block prefixes.
    """
    for prefix in CODE_BLOCK_PREFIXES:
        if code.startswith(prefix):
            if prefix in ['def ', 'class ']:
                return code[len(prefix): code.index('(')]
            else:
                return code[: code.index(':')]

def get_until_no_space(all_lines, i):
    """
    Get all lines until a line outside the block definition is found.
    """
    ret = [all_lines[i]]
    for j in range(i + 1, len(all_lines)):
        if len(all_lines[j]) == 0 or all_lines[j][0] in [' ', '\t', ')']:
            ret.append(all_lines[j])
        else:
            break
    return NEWLINE.join(ret)

def get_blocks(filepath):
    """
    Get all code blocks in a Python file.
    """
    with open(filepath, 'r') as file:
        all_lines = file.read().replace('\r', NEWLINE).split(NEWLINE)
        for i, l in enumerate(all_lines):
            for prefix in CODE_BLOCK_PREFIXES:
                if l.startswith(prefix):
                    code = get_until_no_space(all_lines, i)
                    block_name = get_block_name(code)
                    yield {
                        'code': code,
                        'block_name': block_name,
                        'filepath': filepath,
                    }
                    break

def extract_blocks_from_repo(code_root):
    """
    Extract all .py blocks from the repository.
    """
    code_files = list(code_root.glob('**/*.py'))

    num_files = len(code_files)
    print(f'Total number of .py files: {num_files}')

    if num_files == 0:
        print('Verify repo exists and code_root is set correctly.')
        return None

    all_blocks = [
        block
        for code_file in code_files
        for block in get_blocks(str(code_file))
    ]

    num_blocks = len(all_blocks)
    print(f'Total number of blocks extracted: {num_blocks}')

    return all_blocks

def extract_blocks_from_multiple_repos(username, target_username, token):
    """
    Extract all .py blocks from multiple repositories.
    """
    response = requests.get(f'https://api.github.com/users/{target_username}/repos',
                            headers={'Authorization': f'token {token}'})
    response.raise_for_status()

    repo_names = [repo['name'] for repo in response.json()]

    all_blocks = []
    for repo_name in repo_names:
        print(f"Processing {repo_name}...")
        subprocess.run(f'git clone https://github.com/{target_username}/{repo_name}.git', shell=True)

        code_root = Path(repo_name)
        blocks = extract_blocks_from_repo(code_root)
        if blocks is not None:
            all_blocks.extend(blocks)

        # Delete the cloned repo to save space
        subprocess.run(f'rm -rf {repo_name}', shell=True)

    df = pd.DataFrame(all_blocks)
    df['code_embedding'] = df['code'].apply(get_embedding)
    df['filepath'] = df['filepath'].map(lambda x: str(x))

    return df

def get_embedding(sentence):
    if use_codebert:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embeddings[0]
    else:
        return model.encode([sentence])[0]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_blocks(df, code_query, n=3, pprint=True, n_lines=7):
    embedding = get_embedding(code_query)
    df['similarities'] = df.code_embedding.apply(lambda x: cosine_similarity(x, embedding))

    res = df.sort_values('similarities', ascending=False).head(n)

    if pprint:
        for r in res.iterrows():
            print(f"{r[1].filepath}:{r[1].block_name}  score={round(r[1].similarities, 3)}")
            print("\n".join(r[1].code.split("\n")[:n_lines]))
            print('-' * 70)

    return res

username = input('Enter your GitHub username: ')
token = getpass('Enter your GitHub token: ')

target_username = input('Enter the username of the target user: ')

df = extract_blocks_from_multiple_repos(username, target_username, token)

# After extracting the code blocks, we can search for the blocks
# search_blocks(df, "music parser")

# Save the dataframe to a csv
df.to_csv(f'{target_username}_embeddings.csv', index=False)
