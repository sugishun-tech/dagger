# 学習データ

Jaggerの公式手順と同じ、京都大学ウェブ文書リードコーパス（KWDLC）とMeCab用JUMAN辞書7.0を利用する。コーパス、辞書、学習済みモデルは配布ZIPに含めず、取得・変換スクリプトを用意した。

```bash
python scripts/prepare_kwdlc.py --output data
```

標準ライブラリだけで実行できる。初回は公開配布元から取得し、アーカイブのSHA-256を検証する。約9 MBのコーパスアーカイブ、約30 MBの辞書アーカイブに加え、展開済みデータと約122 MBのUTF-8辞書CSVを保存する。入力アーカイブの再取得を避けるため、`data/.sources/`をキャッシュとして利用する。

ローカルに取得済みの資源を使う場合は次の形になる。

```bash
python scripts/prepare_kwdlc.py --output data \
  --kwdlc /path/to/KWDLC \
  --dictionary /path/to/mecab-jumandic-7.0-20130310
```

ローカル辞書がEUC-JPの場合は`--dictionary-encoding euc_jp`を付ける。標準のJUMAN辞書7.0はUTF-8で読み込む。カスタム辞書にもMeCab-JUMANの11列CSVと4階層品詞を要求する。IPADICやUniDicは特徴列の定義が異なるため、このオプションにそのまま渡さない。

## 出力

| ファイル | 内容 |
| --- | --- |
| `data/train.mecab` | 学習用正解形態素列 |
| `data/dev.mecab` | 開発用正解形態素列 |
| `data/test.mecab` | 評価用正解形態素列 |
| `data/train.txt`, `data/dev.txt`, `data/test.txt` | 各形態素の表層を連結した、一行一文の入力 |
| `data/dictionary.csv` | 修正済みUTF-8のMeCab-JUMAN辞書 |
| `data/sources.json` | 実際の件数、取得元、commit、変換後のSHA-256 |

形態素列の形式は`表層<TAB>品詞,品詞細分類,活用型,活用形,原形,読み,*`で、文末が`EOS`となる。KWDLCの`未定義語`はJaggerの学習手順と同様に`名詞,普通名詞,*,*`へ変換する。これは学習時の品詞規約であり、未知語の正解を別データから補う処理ではない。訓練、開発、評価の文書は`id/split_for_pas/`の標準分割で分離する。

変換スクリプトはKNP形式の構造と公開データ仕様から独立して実装した。Jaggerの変換スクリプトやC++コードを取り込んでいない。固定したデータに対して、独立変換で生成したコーパス6ファイルと辞書CSVが公式変換結果とバイト単位で一致することを確認した。

## 固定した版と実際の件数

KWDLCは[`c6ae49d29eca4e1134c6676d682adf5ed6a25f5a`](https://github.com/ku-nlp/KWDLC/tree/c6ae49d29eca4e1134c6676d682adf5ed6a25f5a)を利用する。

| 分割 | 文書数 | EOSで数えた文数 | 形態素数 |
| --- | ---: | ---: | ---: |
| train | 3,915 | 12,271 | 194,489 |
| dev | 512 | 1,585 | 22,625 |
| test | 700 | 2,195 | 35,869 |

上表は今回の変換で集計した値であり、KWDLCのREADMEに記載された「文書数×3」の文数と一致しない。実データには元の文をさらに分割したS-IDがあるため、速度の「文/秒」の分母は実ファイルの2,195文を使う。参考値としてREADMEにはtrain 11,745文、dev 1,536文、test 2,100文が記載されている。異なる分母の値をそのまま速度比較しない。

MeCab-JUMAN辞書の版は`7.0-20130310`で、751,185行を利用する。公式に案内されたAuxV.csvの文字化け修正と同じ6行を、CSVの表層・原形・読みを照合して独立に修正する。対応は`で� → です`、`ま� → まい`、`こと� → ことだ`。ダウンロードした原本には手を加えず、変換出力に修正を反映する。

| 資源 | SHA-256 |
| --- | --- |
| KWDLC固定commitのtar.gz | `0aec50c1ae4e01258567cdd5edbc794e9885cd776603b05d7da5462dcda6b81f` |
| MeCab-JUMAN原本tar.gz | `eaf216758edee9a159bc3d02507007318686b9537943268c4565cc1f9ef07f15` |
| 修正・結合後の`dictionary.csv` | `f9a46a68bdf0e7f20f60e576885d6b610832983ac0548d7514719db5c85290cd` |
| 変換後の`train.mecab` | `98905dc44586ef8a8e1869c991be617efdeaa0bb236d4d3cc54a8c2ff498edc2` |
| 変換後の`test.mecab` | `d2003b07a351f96519e7a5a990be7537b51a34cb3d9be1b62e3de478447d6d29` |

## 利用条件と出典

- **KWDLC**: [公式リポジトリのREADME](https://github.com/ku-nlp/KWDLC)を確認する。自然言語処理研究への貢献を目的とした公開資源で、収録ウェブ文書の権利者から許諾を得ていない旨、権利者の要請によって更新した場合は旧版を削除して置き換える旨が記載されている。公開取得可能であることから、無条件に再配布可能とは扱わない。daggerのコードのライセンスがコーパスへ適用されるわけでもない。固定版は測定再現のための指定であり、配布元から削除・更新の要請が出ている場合はそちらを優先する。
- **MeCab-JUMAN**: 原本に同梱された`COPYING`は、University of TokyoによるBSD 3-Clause条件。辞書やその派生物を配布する場合は元の著作権表示、条件、免責文を保存する。[MeCab公式](https://taku910.github.io/mecab/)と[Debian公式配布ページ](https://packages.debian.org/source/sid/mecab-jumandic)を参照。スクリプトの取得先は[Debianの原本アーカイブ](https://deb.debian.org/debian/pool/main/m/mecab-jumandic/mecab-jumandic_7.0-20130310.orig.tar.gz)。
- **Jagger**: 手法、データ選択、比較対象は[公式ページ](https://www.tkl.iis.u-tokyo.ac.jp/~ynaga/jagger/)および[ACL 2023論文](https://aclanthology.org/2023.acl-short.2/)に基づく。文字化け箇所は[公式辞書パッチ](https://www.tkl.iis.u-tokyo.ac.jp/~ynaga/jagger/mecab-jumandic-7.0-20130310.patch)で確認した。公式実装は比較用の別ディレクトリでのみ利用し、daggerには含めていない。
- **京都大学テキストコーパス**: Jaggerのもう一つの評価資源だが、[公式リポジトリ](https://github.com/ku-nlp/KyotoCorpus)は原文を含まず、復元には毎日新聞1995年CD-ROMが必要となる。今回はこちらを利用していない。
- **UD Japanese GSD**: [公式リポジトリ](https://github.com/UniversalDependencies/UD_Japanese-GSD)でCC BY-SA 4.0を掲げる代替候補。ただしREADMEでは注釈のライセンスと原文の著作権を区別しており、JUMANと単位・品詞体系も異なる。今回はJaggerと条件を合わせるためKWDLCを選んだ。

## 比較を再実行する際の条件

速度だけでなく、訓練入力、辞書、未知語規則、精度評価の単位を記録する。学習やハイパーパラメータ選択にはtrainとdevを使い、testを混ぜない。`dagger`とJaggerには同じ`.txt`を与え、同じ分割出力または形態素出力を作らせる。出力を作らないカウント専用APIと、全文字列を生成するCLIの時間を直接比較しない。

今回の公式比較対象はJagger `2026-06-27`、公式tar.gzのSHA-256は`1120266accc8040a7cc3f0b1f2f9afbae85ace7c98a5ab363f9ef6151c289792`。GCC 14.2.0の`g++ -O3 -DNDEBUG`でビルドした。モデル生成時間、ロード時間、入出力時間を含むかどうかは測定モードごとに明記する。異なる機種で公表されたJagger公式の「文/秒」と今回の値を割り算して高速化率にしない。
