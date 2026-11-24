# 精度評価と速度測定

## 精度

学習に使っていないテスト集合に対して実行する。

```bash
python scripts/evaluate.py \
  --gold data/test.mecab \
  --model model.dgr \
  --output results/accuracy.json
```

外部の解析器にも同じ評価器を使える。

```bash
/path/to/jagger -m /path/to/jagger-model < data/test.txt > results/jagger.mecab
python scripts/evaluate.py \
  --gold data/test.mecab \
  --pred results/jagger.mecab \
  --output results/jagger-accuracy.json
```

入力形式は `表層<TAB>品詞,細分類,活用型,活用形,原形,読み[,その他]` と `EOS`。MeCabの外見に合わせたJUMANの特徴配列であり、IPA辞書・UniDicの列番号をそのまま受け付ける形式ではない。

| 指標 | 正解と一致する必要があるもの |
| --- | --- |
| `segmentation` | トークンの開始位置と終了位置 |
| `pos1` | 開始位置、終了位置、第一階層の品詞 |
| `full_pos` | 開始位置、終了位置、四階層すべての品詞情報 |
| `lemma` | 開始位置、終了位置、原形。品詞は独立に評価 |

位置はUnicodeコードポイントで数える。文単位の平均ではなく、全テスト文の正解数・予測数・正解トークン数を合計するmicro評価を使う。

\[
P=\frac{N_{correct}}{N_{pred}},\quad
R=\frac{N_{correct}}{N_{gold}},\quad
F_1=\frac{2N_{correct}}{N_{pred}+N_{gold}}
\]

「東京 / 大学 / へ」に対する「東京大学 / へ」は、正解が「へ」の1語、予測が2語、正解集合が3語となる。境界だけの一致率とは異なる。文数不一致や、表層を連結して元の文を復元できない出力はエラーにする。誤りを含む文を分母から落とさない。

## 速度

```bash
python benchmarks/benchmark.py \
  --model model.dgr \
  --input data/test.txt \
  --jagger /path/to/jagger \
  --jagger-model /path/to/jagger-model \
  --jagger-version 2026-06-27 \
  --training-corpus data/train.mecab \
  --dictionary data/dictionary.csv \
  --build-note "dagger: record the actual compiler and flags here" \
  --build-note "jagger: record the actual compiler and flags here" \
  --runs 5 --repeat 100 --api-repeat 10 \
  --output results/benchmark.json
```

`--jagger` と `--jagger-model` を省けばdaggerだけを測る。分かち書きだけなら `--mode wakati`、形態素情報つきだけなら `--mode full`、CLIだけなら `--skip-api` を付ける。

Linuxでは `--cpu 0` のようにCPUを固定できる。ただし、コンテナやホストの割当によって利用可能なCPU番号が異なる。JSONの `environment.cpu_affinity` に実際の割当を記録する。

同じ文を反復する目的は、短いテスト集合の測定を安定させることにある。反復は学習には利用しない。`--repeat` はCLI、`--api-repeat` はPython APIに渡すバッチの反復数。APIは結果全体をメモリ上に生成するので、CLIより小さい既定値にしている。

### 区別する計測範囲

| JSONの項目 | 含める処理 | 含めない処理 |
| --- | --- | --- |
| `modes.full.cli` | プロセス起動、モデル読込、入力、推論、形態素出力整形、標準出力への書込、終了 | 学習、反復入力ファイルの準備 |
| `modes.wakati.cli` | 上記と同じ。ただし出力は分かち書き | 品詞等の文字列出力 |
| `dagger_api_materialized_output` | 既にロードしたモデルによるUTF-8入力の解析と出力バイト列の生成 | モデル読込、ファイルI/O、プロセス起動 |
| `dagger_api_count_only` | 既にロードしたモデルによる解析とトークン数の集計 | 出力文字列の生成、I/O、モデル読込 |

**Jaggerとの速度比は同じ出力モードのCLI同士だけで算出する。** カウント専用APIの数値を、文字列を出力するJaggerのCLIに対する高速化率として扱わない。

CLIは両方とも通常ファイルを標準入力へ渡し、標準出力をOSのnullデバイスに流す。事前の出力検証にも同じ通常ファイル入力を使う。出力は捨てられるが、解析器の内部ではトークンの整形と書込処理を行う。Python APIでは返された出力バイト列の確保までを計時し、結果の解放は各計時の後に行う。

各モードで1回ウォームアップしてから、各ラウンドで両解析器を計測する。どちらを先に実行するかは固定シードのシャッフルで決める。5回の生の秒数、中央値、最小・最大、文/秒、語/秒、入力MiB/秒を記録する。OSページキャッシュが温まった状態での測定であり、完全なコールドスタートを示すものではない。RSSは測定していない。

### 出力と条件の確認

速度測定の前に、反復前のテスト集合を両CLIで実際に処理し、出力ハッシュ・文数・語数・相互の差を保存する。形態素出力ではJaggerとのspan/POS/原形一致も記録するが、これは正解コーパスに対する精度ではない。精度は必ず `scripts/evaluate.py` で別に測る。

同じアルゴリズムでも、未知語処理、同頻度時の選択、辞書・コーパスの版、パターン抽出の実装上の差によって出力は変わる。速度を比較するときは、その条件とテスト精度を併記する。モデルが小さいだけで速くなったケースを、アルゴリズムの優位性と混同しない。

JSONには次も保存する。

- Python、Cython、OS、CPU、CPU affinity、Pythonビルドのコンパイラ情報。
- 入力ファイルのSHA-256と、CRLF・末尾改行を正規化して実際に解析した入力のSHA-256。
- daggerのモデル、Jaggerバイナリ・モデルのサイズとハッシュ。
- 指定した場合の学習コーパス・辞書のハッシュ。
- CLIコマンド、反復数、実行順、Jaggerの指定版。

拡張モジュール独自のコンパイルフラグやJaggerのビルドフラグは、自動検出したPythonビルド情報からは分からない。`--build-note` に実際のビルドログで確認したコンパイラとフラグを渡せば、JSONの `build_notes` にそのまま記録する。複数回指定できる。`environment.python_compiler` と `python_config_cc` はPython本体の情報であり、Cython拡張のコンパイラを意味しない。ビルド時のログも保存し、`-march=native` の有無、スレッド数、測定ホストの競合負荷を報告に補足する。

公式論文の別CPUでの数値と、このベンチマークの数値を直接割って高速化率を出してはいけない。同一ホスト・同一入力・同一出力モードの測定結果を使う。
