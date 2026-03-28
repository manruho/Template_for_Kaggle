<!-- docs/kaggle_competition_guide.md -->

# Kaggleコンペ実行ガイド

このドキュメントは、このテンプレートで実際にKaggleコンペを回すときの手順を日本語でまとめたものです。  
ローカルでの検証から、Kaggle Notebookへのpush、結果取得までを一通り扱います。

## 1. 事前準備

### 1.1 必要なツール

最低限、以下が必要です。

```bash
python3 --version
pip --version
kaggle --version
```

`kaggle` コマンドがなければインストールします。

```bash
pip install kaggle
```

### 1.2 Kaggle APIトークンの設定

1. Kaggleにログイン
2. `Account` ページを開く
3. `Create New API Token` を押す
4. `kaggle.json` を取得する

その後、ローカルに配置します。

```bash
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

接続確認:

```bash
kaggle competitions list
```

## 2. テンプレート生成

例として `titanic` を使う場合は、まずプロジェクト一式を生成します。

```bash
python3 generate_template.py \
  --competition-slug titanic \
  --task-type binary_classification \
  --metric AUC \
  --model-type lgbm \
  --data-type tabular \
  --kaggle-username your-kaggle-id \
  --kernel-slug titanic-exp001 \
  --output-dir ./out/titanic
```

生成後は、対象ディレクトリに移動して作業します。

```bash
cd ./out/titanic
```

## 3. 依存ライブラリの導入

まず共通ライブラリを入れます。

```bash
pip install pandas numpy scikit-learn pyyaml matplotlib seaborn
```

モデルに応じて追加で入れます。

### LightGBM

```bash
pip install lightgbm
```

### XGBoost

```bash
pip install xgboost
```

### CatBoost

```bash
pip install catboost
```

### PyTorch

```bash
pip install torch
```

### TensorFlow

```bash
pip install tensorflow
```

## 4. コンペデータの取得

対象コンペに参加した状態で、データをダウンロードします。

```bash
mkdir -p input/titanic
kaggle competitions download -c titanic -p input/titanic
```

zipを展開します。

```bash
cd input/titanic
unzip -o titanic.zip
cd ../..
```

このテンプレートは、基本的に以下を前提にしています。

- `input/<competition-slug>/train.csv`
- `input/<competition-slug>/test.csv`
- `input/<competition-slug>/sample_submission.csv`

もしファイル名が違うコンペなら、生成先プロジェクトの `configs/base.yaml` を直接調整してください。

## 5. 設定ファイルの調整

生成直後の設定では、ターゲット列名は `target` を前提にしています。  
実際のコンペに合わせて、少なくとも以下は見直してください。

- `configs/base.yaml` の `training.target_column`
- 必要なら `training.group_column`
- `model.params`
- `data.train_path`, `data.test_path`

たとえばTitanicなら、ターゲットは `Survived` です。

```yaml
training:
  target_column: "Survived"
```

ID列や提出用の列名はコンペごとに違うので、必要に応じて `src/data.py` の特徴量除外ロジックも調整します。

## 6. ローカルで実験する

まず1回、ローカルでCVが回るか確認します。

```bash
python run_experiment.py --config configs/exp001.yaml --note "local first run"
```

成功すると、主に以下ができます。

- `experiments/exp001/weights/` にfoldごとのモデル保存
- `logs/experiments.csv` に実験ログ追記

まず確認すべき点:

- `train.csv` と `test.csv` が正しく読めているか
- `target_column` が正しいか
- 文字列列や日付列の前処理で落ちていないか
- 評価指標がそのタスクに合っているか

## 7. Kaggle Notebookとして実行する

このテンプレートでは `kernel-metadata.json` と `push.sh` を使って、Kaggle Notebookへpushする前提です。

### 7.1 metadataの確認

`kernel-metadata.json` の主な項目:

- `id`: `your-kaggle-id/kernel-slug`
- `competition_sources`: 対象コンペのslug
- `enable_gpu`: GPU使用の有無
- `enable_internet`: Internet有効化の有無

注意:

- Code Competitionでは `enable_internet` は `false` にしてください
- GPUが不要なら `enable_gpu` も `false` のままで構いません

### 7.2 push実行

```bash
bash push.sh exp001
```

このスクリプトは以下を行います。

1. `kaggle kernels push -p .`
2. 30秒ごとに `kaggle kernels status` を確認
3. `complete` または `error` になったら `kaggle kernels output` で出力回収
4. 回収先は `experiments/exp001/`

## 8. Kaggle上で失敗しやすい点

### 8.1 ローカルでは動くがKaggleで動かない

原因になりやすいもの:

- requirements相当のライブラリがKaggle環境にない
- `input/` 配下のローカルファイルを直接参照している
- Notebook上のパスとローカルの相対パスがずれている

このテンプレートでは `input/<slug>/...` を前提にしています。  
Kaggle Notebook上では、必要に応じて `/kaggle/input/<slug>/...` に読み替える修正が要る場合があります。

### 8.2 ターゲット列名が違う

生成直後は `target` 固定です。  
実コンペではほぼそのままでは動かないので、最初に `configs/base.yaml` を修正してください。

### 8.3 提出ファイル生成がまだ入っていない

今のテンプレートは、まずCV実験基盤を作ることに寄せています。  
`submission.csv` を自動生成して `kaggle competitions submit` まで行う処理は、必要なら別途追加してください。

## 9. 実運用の流れ

実際には、以下の順で進めると安定します。

1. テンプレート生成
2. データ取得
3. `configs/base.yaml` のターゲット列やパス修正
4. `src/data.py` の前処理をコンペ仕様に合わせて調整
5. ローカルで `exp001` を実行
6. パラメータを変えて `exp002`, `exp003` を増やす
7. KaggleにpushしてNotebook環境で再現確認

## 10. 追加でやるとよいこと

- `requirements.txt` を用意する
- `submission.csv` 生成スクリプトを追加する
- 推論専用の `src/inference.py` を追加する
- 特徴量作成を `src/features.py` に分離する
- OOFやfeature importanceの保存を追加する
