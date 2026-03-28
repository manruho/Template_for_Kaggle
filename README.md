<!-- README.md -->

# Kaggle Template Generator
実際にKaggleコンペで回す手順は [docs/kaggle_competition_guide.md](/home/monruho/Work/Project/Kaggle_template/docs/kaggle_competition_guide.md) に日本語でまとめています。

## 使い方

```bash
uv run generate_template.py \
  --competition-slug titanic \
  --task-type binary_classification \
  --metric AUC \
  --model-type lgbm \
  --data-type tabular \
  --kaggle-username your-name \
  --kernel-slug titanic-exp001 \
  --output-dir ./out/titanic
```

`--use-gpu` と `--enable-internet` は必要なときだけ付けてください。

## 生成されるもの

指定した出力先に、以下の13ファイルを生成します。

- `configs/base.yaml`
- `configs/exp001.yaml`
- `src/__init__.py`
- `src/data.py`
- `src/model.py`
- `src/train.py`
- `src/utils.py`
- `run_experiment.py`
- `kernel-metadata.json`
- `push.sh`
- `notebooks/eda.ipynb`
- `.gitignore`
- `README.md`
