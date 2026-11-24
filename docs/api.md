# APIとモデル

## インストール

Python 3.10以降、Cython 3、C++17コンパイラを使う。ビルド依存は`pip install .`時に導入される。実行時はPython標準ライブラリと生成した拡張だけを使い、NumPyやGPUは不要。配布名・import名・コマンド名は`dagger`。

```bash
python -m pip install .
```

コンパイラの自動選択が環境に合わない場合は`CC=gcc CXX=g++ python -m pip install .`のように指定する。Linux上のGCCとPython 3.12で実測した。他のPython版は付属CIの対象だが、そのCIをここで実行済みという意味ではない。モデルはlittle endianホストを対象とする。

手元のCPU専用にコンパイルする場合は、GCC/Clang環境で`DAGGER_NATIVE=1 python -m pip install --no-cache-dir .`を使える。これは`-march=native`を追加するオプションであり、生成した拡張を別のCPUへ配布しない。掲載ベンチマークでは指定していない。

## 解析

```python
from dagger import Tagger

tagger = Tagger("kwdlc.dgr")
words = tagger.tokenize("猫が魚を食べた。")
tokens = tagger.analyze("猫が魚を食べた。")
text = tagger.parse("猫が魚を食べた。")
```

`Token`は`surface`, `pos`, `lemma`, `reading`, `start`, `end`, `is_unknown`を持つ。`pos`はJUMANの4階層をカンマで結合した文字列。`start`と`end`はPython文字列のコードポイント単位の半開区間で、UTF-8のバイト位置ではない。Unicode正規化や空白削除は行わない。

未知の語や連結によって作られた語は、原形と読みを`*`にする。辞書にない読みを生成する機能は持たない。数字・英字・カタカナの同種隣接単位を連結し、品詞は最後の単位から引き継ぐ。漢数字を数詞に含め、カタカナは直前と今回の予測単位の合計が18 UTF-8バイト未満のときに連結する。これは連結後の総長上限ではない。半角カナと連続空白の連結はdagger側の拡張。

`analyze`と`tokenize`は制御文字も表層として保持する。`parse`は単文用で、出力形式を壊すTAB・CR・LF・NULを拒否する。改行で区切った複数文はバッチAPIへ渡す。バッチAPIでも行内のTABとNULは拒否する。

## 大量処理

```python
payload = "猫が魚を食べた。\n東京で働く。\n".encode("utf-8")
result = tagger.parse_batch_bytes(payload)             # MeCab互換の形態素列
wakati = tagger.parse_batch_bytes(payload, wakati=True) # 分かち書き
number_of_tokens = tagger.count_batch_bytes(payload)   # 出力を生成しない
```

UTF-8の検証後、複数文の解析と文字列整形をCython内部でまとめて行う。`Token`や語ごとのPython文字列を大量に作らないため、Pythonオブジェクトが不要な用途ではこのAPIを使う。countは解析器のコアを通るが、表層や品詞の出力文字列を作らない。その速度を他の解析器の出力ありCLIと直接比較しない。

空バイト列は0文、空行は1文。LFとCRLFを文区切りとして扱い、最終行に改行がなくても処理する。文頭ごとに直前品詞をBOSへ戻す。分かち書き出力はJaggerと同じく、各語の後ろに空白を付ける。入力の空白は語として保持するため、分かち書き文字列を`.split()`して元の空白構造を復元することはできない。

解析モデルは読み取り専用で、作業バッファは呼出しごとに確保する。同じ`Tagger`を複数スレッドで共有でき、推論とバッチ整形のC++ループではGILを解放する。UTF-8検証、Pythonオブジェクトの組み立て、返却時の処理にはGILが必要なので、スレッド数に比例した速度向上を保証するものではない。

## 学習

```bash
dagger train data/train.mecab --dictionary data/dictionary.csv --model kwdlc.dgr
```

Pythonからは次の関数を使う。

```python
from dagger.training import train

stats = train("data/train.mecab", "kwdlc.dgr",
              dictionary_path="data/dictionary.csv")
```

`max_pattern_chars`で右文脈上限を指定できる。既定値は辞書・正解コーパスの最大単語長。`min_count`は候補の最低頻度で既定値1。`prune=False`は剪定を無効にする検証用オプションで、通常は不要。CLIでは`--max-pattern-chars`、`--min-count`、`--no-prune`が対応する。

## モデル形式と検証

`.dgr`は独自形式で、Jaggerのモデルとは互換でない。ファイルは64バイトのヘッダ、UTF-8 JSONメタデータ、64バイト境界へのパディング、固定長のdouble-arrayノードからなる。ヘッダはmagic・メタデータ長・ノード領域長・SHA-256を持つ。モデル生成は一時ファイルへの書込み後にatomic replaceする。

ノード配列はmmapで読み込む。文字集合を密な整数に変換し、実行時のコードポイント→ID対応表を作る。未知のコードポイントにはID0を割り当てる。品詞条件は文字IDの後ろにある別の記号領域を使う。

読込時に寸法、チェックサム、親チェーン、遷移記号、末端ラベル、UTF-8の切断位置と原形編集位置を検証する。モデルをPythonコードとして実行するpickleは使わない。SHA-256は破損検知であり、配布者の署名や認証を代替しない。メタデータのJSONと語彙文字列はPython/C++上にも展開するため、実行時メモリはファイルサイズだけからは決まらない。

## 検証の実行

```bash
python -m pip install '.[dev]'
python -m pytest -q
```

学習規則、独立した単純照合器との一致、剪定の保存性、Unicode・未知語・文区切り、同一モデルの並列参照、破損モデルの拒否を検証する。性能と精度の再現手順は[benchmarking.md](benchmarking.md)を参照。
