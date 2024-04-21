# RAT_work
[RAT](https://craftjarvis.github.io/RAT/)を試す。  
[ざっくりメモ](./docs/memo.md)

## ドキュメント

## システム構成図

## 利用ツール
- OPENAI API
- Google Custom Search API

## 環境構築の手順
### 1. モジュールインストール
```
$ cd RAT_work
$ poetry install
```


### 2. env_files/.env.workを作成
以下のように記述。  

```
OPENAI_API_KEY={your_openai_api_key}
GOOGLE_CSE_ID={your_google_cse_id}
GOOGLE_API_KEY={your_google_api_key}
JAPANESE_FLG=1  # 日本語の場合は1、英語の場合は0
MAKE_BING_QUERY_TIMEOUT=5  # Bing検索クエリ生成のタイムアウト
GET_BING_RESULT_TIMEOUT=3  # Bing検索結果取得のタイムアウト
REVISE_ANSWER_TIMEOUT=10   # 回答の修正のタイムアウト
```

### 3. 実行
```
$ cd python_file
$ poetry run python work.py
```

## ディレクトリ構成
```
RAT_work/
  ┣ assets/  # READMEなどに利用する画像ファイルなど
  ┣ docs/
  ┃  ┗ memo.md  # 詳細なメモ
  ┣ env_file/
  ┃  ┗ .env.work  # 環境変数
  ┣ prompts/
  ┃  ┣ draft.txt  # [ENGLISH]質問に対し、段階的に回答するためのドラフト作成用プロンプト
  ┃  ┣ draft_jp.txt  # [JAPANESE]質問に対し、段階的に回答するためのドラフト作成用プロンプト
  ┃  ┣ query.txt  # [ENGLISH]段階的な回答の正確性を検証するためのBing検索キーワード作成用プロンプト
  ┃  ┣ query_jp.txt  # [JAPANESE]段階的な回答の正確性を検証するためのBing検索キーワード作成用プロンプト
  ┃  ┣ revise.txt  # [ENGLISH]生成した文章の訂正用プロンプト
  ┃  ┣ revise_jp.txt  # [JAPANESE]生成した文章の訂正用プロンプト
  ┃  ┣ system_prompt.txt  # [ENGLISH]生成した文章の訂正用プロンプト
  ┃  ┗ system_prompt_jp.txt  # [JAPANESE]生成した文章の訂正用プロンプト
  ┗ python_file/
     ┣ settings.py  # 環境変数の読み込み
     ┗ work.py      # メイン処理
```

## 実行例
### 入力
```
deep learningの発展の歴史を教えてください。
```

### 出力
```
2024-04-20 22:49:08.574388 [INFO] ドラフト作成中...
2024-04-20 22:49:21.784116 [INFO] ドラフト作成完了
##################### DRAFT #######################
深層学習の発展の歴史について、重要なマイルストーンとして挙げられるいくつかの時期があります。

最初に、1950年代から1960年代にかけて、パーセプトロンやバックプロパゲーションといった初期のニューラルネットワークのアイデアが提示されました。これらは、後に深層学習の発展に影響を与える重要な概念となりました。

その後、1980年代から1990年代にかけて、ニューラルネットワークが再び注目されました。特に、バックプロパゲーションアルゴリズムの改善や畳み込みニューラルネットワーク（CNN）の導入などが行われ、画像認識などのタスクで成果が上がりました。

さらに、2000年代以降、計算能力の向上や大規模なデータセットの利用などが進み、深層学習のブームが訪れました。深層学習は、畳み込みニューラルネットワーク（CNN）、リカレントニューラルネットワーク（RNN）、長短期記憶（LSTM）など、さまざまなアーキテクチャやアルゴリズムの発展を経て発展してきました。

最近では、大規模なトランスフォーマー・モデルの成功や、強化学習の応用など、深層学習はさらなる進化を遂げています。特に自然言語処理や画像認識などの領域で、深層学習は驚異的な成果を上げています。

このようにして、深層学習は数十年にわたる研究と開発を経て、今日では多岐にわたる領域で広く活用されている技術となっています。
#####################  END  #######################
2024-04-20 22:49:21.784244 [INFO] ドラフトを処理中...
2024-04-20 22:49:21.784269 [INFO] ドラフトが6のパーツに分割されました。
00000000000000000000000000000000000000000000000000000000000000000000000000000000
2024-04-20 22:49:21.784325 [INFO] 第1/6番目のパーツを修正中...
2024-04-20 22:49:21.784346 [INFO] 対応するQueryを生成中...
2024-04-20 22:49:25.667591 [INFO] 関数<function get_query_wrapper at 0x109fd48b0>の実行が正常に完了しました
>>> 0/6 Query: ## Summary:  深層学習の発展にはいくつかの重要なマイルストーンがあります。  ## Query: 深層学習の発展の歴史とマイルストーン
2024-04-20 22:49:25.670728 [INFO] webの内容を取得中...
Fetching pages:   0%|                                                                                                                                                       | 0/1 [00:00<?, ?it/s]2024-04-20 22:49:30.678550 [INFO] 関数<function get_content_wrapper at 0x109fd4940>の実行がタイムアウト(5s)，に達しました。プロセスを終了します...
2024-04-20 22:49:30.689661 [INFO] 後続処理をスキップ...
11111111111111111111111111111111111111111111111111111111111111111111111111111111
2024-04-20 22:49:30.689863 [INFO] 第2/6番目のパーツを修正中...
2024-04-20 22:49:30.690012 [INFO] 対応するQueryを生成中...
2024-04-20 22:49:34.874804 [INFO] 関数<function get_query_wrapper at 0x109fd48b0>の実行が正常に完了しました
>>> 1/6 Query: "深層学習の歴史と初期のニューラルネットワークの重要な概念は何ですか？"    1950年代から1960年代にかけての初期のニューラルネットワークのアイデアとその深層学習への影響。
2024-04-20 22:49:34.877031 [INFO] webの内容を取得中...
>>> No good Google Search Result was found
2024-04-20 22:49:37.365419 [INFO] 関数<function get_content_wrapper at 0x109fd4940>の実行が正常に完了しました
2024-04-20 22:49:37.367650 [INFO] 後続処理をスキップ...
22222222222222222222222222222222222222222222222222222222222222222222222222222222
2024-04-20 22:49:37.367863 [INFO] 第3/6番目のパーツを修正中...
2024-04-20 22:49:37.367945 [INFO] 対応するQueryを生成中...
2024-04-20 22:49:40.391218 [INFO] 関数<function get_query_wrapper at 0x109fd48b0>の実行が正常に完了しました
>>> 2/6 Query: 深層学習の歴史とマイルストーン.currentTimeMillis("2024-04-20")
2024-04-20 22:49:40.394072 [INFO] webの内容を取得中...
>>> No good Google Search Result was found
2024-04-20 22:49:43.347441 [INFO] 関数<function get_content_wrapper at 0x109fd4940>の実行が正常に完了しました
2024-04-20 22:49:43.349902 [INFO] 後続処理をスキップ...
33333333333333333333333333333333333333333333333333333333333333333333333333333333
2024-04-20 22:49:43.350266 [INFO] 第4/6番目のパーツを修正中...
2024-04-20 22:49:43.350495 [INFO] 対応するQueryを生成中...
2024-04-20 22:49:46.547340 [INFO] 関数<function get_query_wrapper at 0x109fd48b0>の実行が正常に完了しました
>>> 3/6 Query: 深層学習の発展の歴史 クエリ: 深層学習の発展史
2024-04-20 22:49:46.550298 [INFO] webの内容を取得中...
Fetching pages:   0%|                                                                                                                                                       | 0/1 [00:00<?, ?it/s]2024-04-20 22:49:51.559875 [INFO] 関数<function get_content_wrapper at 0x109fd4940>の実行がタイムアウト(5s)，に達しました。プロセスを終了します...
2024-04-20 22:49:51.571695 [INFO] 後続処理をスキップ...
44444444444444444444444444444444444444444444444444444444444444444444444444444444
2024-04-20 22:49:51.571814 [INFO] 第5/6番目のパーツを修正中...
2024-04-20 22:49:51.572390 [INFO] 対応するQueryを生成中...
2024-04-20 22:49:55.775308 [INFO] 関数<function get_query_wrapper at 0x109fd48b0>の実行が正常に完了しました
>>> 4/6 Query: 深層学習の歴史と最近の進化について教えてください。深層学習の最新の進展に関する情報を見つけたいです。深層学習の進化やトランスフォーマーモデルの成功についても知りたいです。
2024-04-20 22:49:55.777930 [INFO] webの内容を取得中...
Fetching pages:   0%|                                                                                                                                                       | 0/1 [00:00<?, ?it/s]2024-04-20 22:50:00.784685 [INFO] 関数<function get_content_wrapper at 0x109fd4940>の実行がタイムアウト(5s)，に達しました。プロセスを終了します...
2024-04-20 22:50:00.796591 [INFO] 後続処理をスキップ...
55555555555555555555555555555555555555555555555555555555555555555555555555555555
2024-04-20 22:50:00.796714 [INFO] 第6/6番目のパーツを修正中...
2024-04-20 22:50:00.796913 [INFO] 対応するQueryを生成中...
2024-04-20 22:50:04.579915 [INFO] 関数<function get_query_wrapper at 0x109fd48b0>の実行が正常に完了しました
>>> 5/6 Query: 深層学習の歴史と発展、最新の進化。深層学習の広範な応用分野。深層学習の進化と成功要因。深層学習の進化。
2024-04-20 22:50:04.582383 [INFO] webの内容を取得中...
Fetching pages:   0%|                                                                                                                                                       | 0/1 [00:00<?, ?it/s]Failed to decode content from https://www.soumu.go.jp/johotsusintokei/whitepaper/ja/h28/pdf/n4200000.pdf
Fetching pages: 100%|###############################################################################################################################################| 1/1 [00:00<00:00,  1.70it/s]
2024-04-20 22:50:09.591167 [INFO] 関数<function get_content_wrapper at 0x109fd4940>の実行がタイムアウト(5s)，に達しました。プロセスを終了します...
2024-04-20 22:50:09.602627 [INFO] 後続処理をスキップ...


深層学習の発展の歴史について、重要なマイルストーンとして挙げられるいくつかの時期があります。

最初に、1950年代から1960年代にかけて、パーセプトロンやバックプロパゲーションといった初期のニューラルネットワークのアイデアが提示されました。これらは、後に深層学習の発展に影響を与える重要な概念となりました。

その後、1980年代から1990年代にかけて、ニューラルネットワークが再び注目されました。特に、バックプロパゲーションアルゴリズムの改善や畳み込みニューラルネットワーク（CNN）の導入などが行われ、画像認識などのタスクで成果が上がりました。

さらに、2000年代以降、計算能力の向上や大規模なデータセットの利用などが進み、深層学習のブームが訪れました。深層学習は、畳み込みニューラルネットワーク（CNN）、リカレントニューラルネットワーク（RNN）、長短期記憶（LSTM）など、さまざまなアーキテクチャやアルゴリズムの発展を経て発展してきました。

最近では、大規模なトランスフォーマー・モデルの成功や、強化学習の応用など、深層学習はさらなる進化を遂げています。特に自然言語処理や画像認識などの領域で、深層学習は驚異的な成果を上げています。

このようにして、深層学習は数十年にわたる研究と開発を経て、今日では多岐にわたる領域で広く活用されている技術となっています。
```

## 参考
- [RAT](https://craftjarvis.github.io/RAT/)
- [RATのarxivのまとめ](https://github.com/ys201810/reading_something/blob/master/memo/arxiv/LLM/RAT_Retrieval_Augmented_Thoughts_Elicit_Context-Aware_Reasoning_in_Long-Horizon_Generation.md)

