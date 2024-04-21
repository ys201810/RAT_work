from openai import OpenAI
import os
import settings
from langchain.tools import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import AsyncHtmlLoader
import tiktoken
from datetime import datetime
from multiprocessing import Process, Queue
from difflib import unified_diff

OPENAI_API_KEY = settings.OPENAI_API_KEY
GOOGLE_CSE_ID = settings.GOOGLE_CSE_ID
GOOGLE_API_KEY = settings.GOOGLE_API_KEY

openai_client = OpenAI(api_key=OPENAI_API_KEY)
JAPANESE_FLG = settings.JAPANESE_FLG
MAKE_BING_QUERY_TIMEOUT = int(settings.MAKE_BING_QUERY_TIMEOUT)
GET_BING_RESULT_TIMEOUT = int(settings.GET_BING_RESULT_TIMEOUT)
REVISE_ANSWER_TIMEOUT = int(settings.REVISE_ANSWER_TIMEOUT)
if JAPANESE_FLG == '1':
    language = '_jp'
else:
    language = ''

with open(f"../prompts/system_prompt{language}.txt", "r") as f:
    chatgpt_system_prompt = f.read()
chatgpt_system_prompt = chatgpt_system_prompt.format(date=datetime.now().strftime('%Y-%m-%d'))


def get_search(query:str="", k:int=1): # get the top-k resources with google
    search = GoogleSearchAPIWrapper(k=k)
    def search_results(query):
        return search.results(query, k)
    tool = Tool(
        name="Google Search Snippets",
        description="Search Google for recent results.",
        func=search_results,
    )
    ref_text = tool.run(query)
    if 'Result' not in ref_text[0].keys():
        return ref_text
    else:
        return None


def get_page_content(link:str):
    loader = AsyncHtmlLoader([link])
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    if len(docs_transformed) > 0:
        return docs_transformed[0].page_content
    else:
        return None


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def chunk_text_by_sentence(text, chunk_size=2048):
    """Chunk the $text into sentences with less than 2k tokens."""
    sentences = text.split('. ')
    chunked_text = []
    curr_chunk = []
    # 各文を追加していき、各チャンクが2kトークン未満であることを保証する。
    for sentence in sentences:
        if num_tokens_from_string(". ".join(curr_chunk)) + num_tokens_from_string(sentence) + 2 <= chunk_size:
            curr_chunk.append(sentence)
        else:
            chunked_text.append(". ".join(curr_chunk))
            curr_chunk = [sentence]
    # 最後のチャンクを追加
    if curr_chunk:
        chunked_text.append(". ".join(curr_chunk))
    return chunked_text[0]


def chunk_text_front(text, chunk_size = 2048):
    '''
    get the first `trunk_size` token of text
    '''
    chunked_text = ""
    tokens = num_tokens_from_string(text)
    if tokens < chunk_size:
        return text
    else:
        ratio = float(chunk_size) / tokens
        char_num = int(len(text) * ratio)
        return text[:char_num]


def chunk_texts(text, chunk_size = 2048):
    '''
    trunk the text into n parts, return a list of text
    [text, text, text]
    '''
    tokens = num_tokens_from_string(text)
    if tokens < chunk_size:
        return [text]
    else:
        texts = []
        n = int(tokens/chunk_size) + 1
        # 各部分の長さを計算
        part_length = len(text) // n
        # 割り切れない場合、最後の部分に追加の文字が含まれる
        extra = len(text) % n
        parts = []
        start = 0

        for i in range(n):
            # extraの数だけの部分には、それぞれ1文字ずつ多く割り当てる
            end = start + part_length + (1 if i < extra else 0)
            parts.append(text[start:end])
            start = end
        return parts

def get_draft(question):
    # Getting the draft answer
    with open(f"../prompts/draft{language}.txt", "r") as f:
        draft_prompt = f.read().replace('\\n', '\n')  # \n\nが\\n\\nで読まれてしまうため、replace

    draft = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": chatgpt_system_prompt
            },
            {
                "role": "user",
                "content": f"{question}" + draft_prompt
            }
        ],
        temperature = 1.0
    ).choices[0].message.content
    return draft


def split_draft(draft, split_char = '\n\n'):
    # draftを複数の段落に分割
    # split_char: '\n\n'
    draft_paragraphs = draft.split(split_char)
    return draft_paragraphs


def get_query(question, answer):
    with open(f"../prompts/query{language}.txt", "r") as f:
        query_prompt = f.read().replace('\\n', '\n')
    query = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": chatgpt_system_prompt
            },
            {
                "role": "user",
                "content": f"##Question: {question}\n\n##Content: {answer}\n\n##Instruction: {query_prompt}"
            }
        ],
        temperature = 1.0
    ).choices[0].message.content
    return query

def get_content(query):
    res = get_search(query, 1)
    if not res:
        print(">>> No good Google Search Result was found")
        return None
    search_results = res[0]
    link = search_results['link'] # title, snippet
    res = get_page_content(link)
    if not res:
        print(f">>> No content was found in {link}")
        return None
    retrieved_text = res
    trunked_texts = chunk_texts(retrieved_text, 1500)
    trunked_texts = [trunked_text.replace('\n', " ") for trunked_text in trunked_texts]
    return trunked_texts

def get_revise_answer(question, answer, content):
    with open(f"../prompts/revise{language}.txt", "r") as f:
        revise_prompt = f.read().replace('\\n', '\n')
    revised_answer = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
                {
                    "role": "system",
                    "content": chatgpt_system_prompt
                },
                {
                    "role": "user",
                    "content": f"##Existing Text in Wiki Web: {content}\n\n##Question: {question}\n\n##Answer: {answer}\n\n##Instruction: {revise_prompt}"
                }
            ],
            temperature = 1.0
    ).choices[0].message.content
    return revised_answer

def get_query_wrapper(q, question, answer):
    result = get_query(question, answer)
    q.put(result)  # 結果をキューに入れる

def get_content_wrapper(q, query):
    result = get_content(query)
    q.put(result)  # 結果をキューに入れる

def get_revise_answer_wrapper(q, question, answer, content):
    result = get_revise_answer(question, answer, content)
    q.put(result)

def run_with_timeout(func, timeout, *args, **kwargs):
    q = Queue()  # プロセス間通信に利用するQueueを作成
    # 渡された関数を実行するためのプロセスを作成し、Queueと他の*args、**kwargsを引数に渡す
    p = Process(target=func, args=(q, *args), kwargs=kwargs)
    p.start()
    # プロセスが完了するかタイムアウトまで待機
    p.join(timeout)
    if p.is_alive():
        print(f"{datetime.now()} [INFO] 関数{str(func)}の実行がタイムアウト({timeout}s)，に達しました。プロセスを終了します...")
        p.terminate()  # プロセスを強制終了
        p.join()  # プロセスが終了したことを確認
        result = None  # タイムアウトの場合は結果がない
    else:
        print(f"{datetime.now()} [INFO] 関数{str(func)}の実行が正常に完了しました")
        result = q.get()  # キューから結果を取得
    return result


def generate_diff_html(text1, text2):
    diff = unified_diff(text1.splitlines(keepends=True),
                        text2.splitlines(keepends=True),
                        fromfile='text1', tofile='text2')

    diff_html = ""
    for line in diff:
        if line.startswith('+'):
            diff_html += f"<div style='color:green;'>{line.rstrip()}</div>"
        elif line.startswith('-'):
            diff_html += f"<div style='color:red;'>{line.rstrip()}</div>"
        elif line.startswith('@'):
            diff_html += f"<div style='color:blue;'>{line.rstrip()}</div>"
        else:
            diff_html += f"{line.rstrip()}<br>"
    return diff_html

newline_char = '\n'

def rat(question):
    print(f"{datetime.now()} [INFO] ドラフト作成中...")
    draft = get_draft(question)
    print(f"{datetime.now()} [INFO] ドラフト作成完了")
    print(f"##################### DRAFT #######################")
    print(draft)
    print(f"#####################  END  #######################")

    print(f"{datetime.now()} [INFO] ドラフトを処理中...")
    draft_paragraphs = split_draft(draft)
    print(f"{datetime.now()} [INFO] ドラフトが{len(draft_paragraphs)}のパーツに分割されました。")
    answer = ""
    for i, p in enumerate(draft_paragraphs):
        print(str(i)*80)
        print(f"{datetime.now()} [INFO] 第{i+1}/{len(draft_paragraphs)}番目のパーツを修正中...")
        answer = answer + '\n\n' + p
        # print(f"[{i}/{len(draft_paragraphs)}] Original Answer:\n{answer.replace(newline_char, ' ')}")

        # query = get_query(question, answer)
        print(f"{datetime.now()} [INFO] 対応するQueryを生成中...")
        res = run_with_timeout(get_query_wrapper, MAKE_BING_QUERY_TIMEOUT, question, answer)
        if not res:
            print(f"{datetime.now()} [INFO] 後続処理をスキップ...")
            continue
        else:
            query = res
        print(f">>> {i}/{len(draft_paragraphs)} Query: {query.replace(newline_char, ' ')}")

        print(f"{datetime.now()} [INFO] webの内容を取得中...")
        # content = get_content(query)
        res = run_with_timeout(get_content_wrapper, GET_BING_RESULT_TIMEOUT, query)
        if not res:
            print(f"{datetime.now()} [INFO] 後続処理をスキップ...")
            continue
        else:
            content = res

        for j, c in enumerate(content):
            if  j > 2:
                break
            print(f"{datetime.now()} [INFO] ウェブの内容に基づいて対応する回答を修正中...[{j}/{min(len(content),3)}]")
            # answer = get_revise_answer(question, answer, c)
            res = run_with_timeout(get_revise_answer_wrapper, REVISE_ANSWER_TIMEOUT, question, answer, c)
            if not res:
                print(f"{datetime.now()} [INFO] 後続処理をスキップ...")
                continue
            else:
                diff_html = generate_diff_html(answer, res)
                # display(HTML(diff_html))
                answer = res
            print(f"{datetime.now()} [INFO] 回答の修正完了[{j}/{min(len(content),3)}]")
        # print(f"[{i}/{len(draft_paragraphs)}] REVISED ANSWER:\n {answer.replace(newline_char, ' ')}")
        # print()
    return draft, answer


def main():
    question = "冨樫義博を説明してください。"  #"Introduce Jin-Yong's Life."
    draft, answer = rat(question)
    print(answer)


if __name__ == "__main__":
    main()
