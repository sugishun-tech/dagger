# dagger

Jagger方式を独立実装した、Cython製の日本語形態素解析器。Python 3.10以降とC++17コンパイラが必要。

```bash
python -m pip install .
dagger train examples/tiny.mecab --model tiny.dgr
dagger --model tiny.dgr < examples/input.txt
dagger --model tiny.dgr --wakati < examples/input.txt
```

```python
from dagger import Tagger

tagger = Tagger("tiny.dgr")
print(tagger.tokenize("私は猫が好きだ。"))
for token in tagger.analyze("猫が魚を食べた。"):
    print(token.surface, token.pos, token.lemma)
```

本学習は次の手順。付属の6文は動作確認専用。

```bash
python scripts/prepare_kwdlc.py --output data
dagger train data/train.mecab --dictionary data/dictionary.csv --model kwdlc.dgr
dagger --model kwdlc.dgr < data/test.txt
```

[データの利用条件](docs/data.md) / [API・モデル形式](docs/api.md) / [アルゴリズム](docs/algorithm.md) / [測定方法](docs/benchmarking.md)

