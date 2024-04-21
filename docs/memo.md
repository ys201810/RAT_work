# RATのメモ
## 処理の流れ
1. 受けた質問に対し、段階的な回答のドラフトを生成する。
2. ドラフトを分割。
3. 分割されたドラフトでループ開始
4. 各文章が正しいかを確認するためのBing検索をするための検索ワードを生成
5. Bing検索を行い、検索結果を取得
6. 検索結果の文章をchunkに分割
7. 分割したchunk数のループ開始
8. chunkの文章を元に、回答の修正を行う


### 1. 受けた質問に対し、段階的な回答のドラフトを生成する。
prompts/draft.txtを使い、段階的な回答を生成。
例えば、質問が「冨樫義博を説明してください。」である場合に作成されるドラフトは以下。

```
冨樫義博は、日本の漫画家であり、代表作に『幽☆遊☆白書』や『HUNTER×HUNTER』などがあります。彼は1966年生まれで、漫画界で非常に影響力のある存在です。彼の作品は独創的でありながらも幅広い読者層に支持されており、特に緻密なストーリーテリングとキャラクター造形が評価されています。

冨樫義博は、その作風や作品の特徴から、日本の漫画界で一風変わった存在として知られています。彼の作品は緻密なストーリー展開やキャラクターの心理描写、意外な展開などが特徴であり、読者を驚かせることでも知られています。そのため、彼の作品は多くのファンから支持を受けています。

また、冨樫義博は作品の長期連載や休載といった点でも知られています。特に『HUNTER×HUNTER』は、作品の質を維持するために何度か長期休載を行うことがあります。このこだわりや信念から、彼の作品には熱烈なファンが多い一方で、連載再開を待ち望む声も多く聞かれます。

総じて、冨樫義博は日本の漫画界において確固たる地位を築いた漫画家であり、彼の作品からは彼自身の緻密な考えや独創性が感じられます。その作品世界には多くの読者が夢中になり、今もなお多くの人々に愛され続けています。
```

### 2. ドラフトを分割
ドラフトは、'\n\n'で分割されているため、'\n\n'でsplitする。

### 3. 分割されたドラフトでループ開始

### 4. 各文章が正しいかを確認するためのBing検索をするための検索ワードを生成
prompts/query.txtを使い、ドラフトとして作成された文章が正確かどうかを検索するためのワードを生成する。

例えば、分割されたドラフトの1つ目が以下の文章である場合、
```
冨樫義博は、日本の漫画家であり、代表作に『幽☆遊☆白書』や『HUNTER×HUNTER』などがあります。彼は1966年生まれで、漫画界で非常に影響力のある存在です。彼の作品は独創的でありながらも幅広い読者層に支持されており、特に緻密なストーリーテリングとキャラクター造形が評価されています。
```

以下のような検索ワードが生成される。
```
冨樫義博は日本の代表的な漫画家であり、作品は緻密なストーリーテリングとキャラクター造形で評価されています。彼の影響力についても教えてください。\n**クエリ:** 冨樫義博の影響力漫画作品
```

### 5. Bing検索を行い、検索結果を取得
GoogleSearchAPIWrapperで検索を行う。  
検索結果は、以下のような形式で取得される。  

```
[
    {
        'title': '冨樫義博：マンガ界の革命家 - ダイエットや趣味情報',
        'link': 'https://xs420383.xsrv.jp/?p=1382',
        'snippet': 'Jul 27, 2023 ... 彼のユニークなストーリーテリングと魅力的なキャラクター ... され、その影響力はさらに拡大している ... さらに、冨樫の多面的なキャラクター造形能力も\xa0...'
    }
]
```

### 6. 検索結果の文章をchunkに分割
検索結果のlinkのページからテキストを取得し、chunkに分割する。  

### 7. 分割したchunk数のループ開始

### 8. chunkの文章を元に、回答の修正を行う
prompts/revise.txtを使い、chunkの文章を元に回答を修正する。
chunk回数これを繰り返すことで、正しい回答を生成する。  

毎回のクエリには、元の質問と前回の回答が含まれており、  
こうすることで大元の質問に正しく回答するのと、過去の回答との整合性を取るようになっている。  
