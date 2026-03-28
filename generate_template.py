#!/usr/bin/env python3
# generate_template.py

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TASK_TYPES = {
    "binary_classification",
    "multiclass_classification",
    "regression",
    "ranking",
}
MODEL_TYPES = {
    "lgbm",
    "xgboost",
    "catboost",
    "sklearn",
    "pytorch",
    "tensorflow",
    "ensemble",
}
DATA_TYPES = {
    "tabular",
    "image",
    "text",
    "time_series",
    "multi_modal",
}


@dataclass
class Spec:
    competition_slug: str
    task_type: str
    metric: str
    model_type: str
    data_type: str
    kaggle_username: str
    kernel_slug: str
    use_gpu: bool
    enable_internet: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kaggleコンペ用のPythonプロジェクトテンプレートを生成します。"
    )
    parser.add_argument("--competition-slug", required=True)
    parser.add_argument("--task-type", required=True, choices=sorted(TASK_TYPES))
    parser.add_argument("--metric", required=True)
    parser.add_argument("--model-type", required=True, choices=sorted(MODEL_TYPES))
    parser.add_argument("--data-type", required=True, choices=sorted(DATA_TYPES))
    parser.add_argument("--kaggle-username", required=True)
    parser.add_argument("--kernel-slug", default="my-kernel")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--enable-internet", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="generated_project",
        help="生成先ディレクトリ",
    )
    return parser.parse_args()


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def yaml_lines(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {yaml_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        if not value:
            lines.append(f"{prefix}[]")
            return lines
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {yaml_scalar(item)}")
        return lines
    return [f"{prefix}{yaml_scalar(value)}"]


def to_yaml(data: dict[str, Any]) -> str:
    return "\n".join(yaml_lines(data)) + "\n"


def default_model_params(spec: Spec) -> dict[str, Any]:
    if spec.model_type == "lgbm":
        params: dict[str, Any] = {
            "n_estimators": 2000,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }
        if spec.task_type == "binary_classification":
            params["objective"] = "binary"
        elif spec.task_type == "multiclass_classification":
            params["objective"] = "multiclass"
        else:
            params["objective"] = "regression"
        if spec.use_gpu:
            params["device"] = "gpu"
        return params
    if spec.model_type == "xgboost":
        params = {
            "n_estimators": 2000,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
        }
        if spec.task_type == "binary_classification":
            params["objective"] = "binary:logistic"
        elif spec.task_type == "multiclass_classification":
            params["objective"] = "multi:softprob"
        else:
            params["objective"] = "reg:squarederror"
        if spec.use_gpu:
            params["device"] = "cuda"
        return params
    if spec.model_type == "catboost":
        params = {
            "iterations": 3000,
            "learning_rate": 0.03,
            "depth": 6,
            "loss_function": "Logloss"
            if spec.task_type == "binary_classification"
            else "MultiClass"
            if spec.task_type == "multiclass_classification"
            else "RMSE",
            "eval_metric": spec.metric,
            "random_seed": 42,
            "verbose": 0,
        }
        if spec.use_gpu:
            params["task_type"] = "GPU"
        return params
    if spec.model_type == "sklearn":
        return {
            "n_estimators": 300,
            "max_depth": 8,
            "random_state": 42,
            "n_jobs": -1,
        }
    if spec.model_type == "ensemble":
        return {
            "n_estimators": 300,
            "max_depth": 8,
            "random_state": 42,
        }
    if spec.model_type == "pytorch":
        return {
            "hidden_dim": 256,
            "dropout": 0.2,
            "lr": 1e-3,
            "batch_size": 256,
            "epochs": 10,
            "weight_decay": 1e-5,
        }
    if spec.model_type == "tensorflow":
        return {
            "hidden_units": [256, 128],
            "dropout": 0.2,
            "lr": 1e-3,
            "batch_size": 256,
            "epochs": 10,
        }
    raise ValueError(f"unsupported model type: {spec.model_type}")


def experiment_model_overrides(spec: Spec) -> dict[str, Any]:
    if spec.model_type == "lgbm":
        return {"learning_rate": 0.03, "num_leaves": 127}
    if spec.model_type == "xgboost":
        return {"learning_rate": 0.03, "max_depth": 8}
    if spec.model_type == "catboost":
        return {"learning_rate": 0.05, "depth": 8}
    if spec.model_type == "sklearn":
        return {"n_estimators": 500, "max_depth": 10}
    if spec.model_type == "ensemble":
        return {"n_estimators": 500, "max_depth": 10}
    if spec.model_type == "pytorch":
        return {"hidden_dim": 512, "dropout": 0.3}
    if spec.model_type == "tensorflow":
        return {"hidden_units": [512, 256], "dropout": 0.3}
    raise ValueError(f"unsupported model type: {spec.model_type}")


def base_config(spec: Spec) -> dict[str, Any]:
    return {
        "competition": spec.competition_slug,
        "exp_name": "base",
        "task": {
            "type": spec.task_type,
            "metric": spec.metric,
        },
        "data": {
            "type": spec.data_type,
            "input_dir": f"input/{spec.competition_slug}",
            "train_path": f"input/{spec.competition_slug}/train.csv",
            "test_path": f"input/{spec.competition_slug}/test.csv",
            "sample_submission_path": f"input/{spec.competition_slug}/sample_submission.csv",
            "n_folds": 5,
            "seed": 42,
        },
        "model": {
            "name": spec.model_type,
            "params": default_model_params(spec),
        },
        "training": {
            "target_column": "target",
            "group_column": "group_id",
            "save_weights": True,
            "early_stopping_rounds": 100,
            "num_boost_round": 2000,
        },
        "output": {
            "exp_dir_base": "experiments",
        },
    }


def exp_config(spec: Spec) -> dict[str, Any]:
    return {
        "exp_name": "exp001",
        "model": {
            "params": experiment_model_overrides(spec),
        },
    }


def build_data_py(spec: Spec) -> str:
    default_type = spec.data_type
    return f'''# src/data.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_data(cfg):
    """学習データとテストデータを読み込む。"""
    data_cfg = cfg["data"]
    input_dir = Path(data_cfg["input_dir"])
    train_path = Path(data_cfg.get("train_path", input_dir / "train.csv"))
    test_path = Path(data_cfg.get("test_path", input_dir / "test.csv"))
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def _basic_fillna(frame, target_column):
    for column in frame.columns:
        if column == target_column:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].fillna(frame[column].median())
        else:
            frame[column] = frame[column].fillna("missing")
    return frame


def _convert_text_columns(frame, excluded_columns):
    text_columns = []
    for column in frame.select_dtypes(include=["object"]).columns:
        if column in excluded_columns:
            continue
        frame[column] = frame[column].astype(str).fillna("")
        frame[f"{{column}}_char_len"] = frame[column].str.len()
        frame[f"{{column}}_word_count"] = frame[column].str.split().str.len()
        text_columns.append(column)
    return frame, text_columns


def _convert_datetime_columns(frame, excluded_columns):
    for column in frame.columns:
        if column in excluded_columns:
            continue
        if any(token in column.lower() for token in ["date", "time", "timestamp"]):
            converted = pd.to_datetime(frame[column], errors="coerce")
            if converted.notna().sum() == 0:
                continue
            frame[f"{{column}}_year"] = converted.dt.year.fillna(-1).astype(int)
            frame[f"{{column}}_month"] = converted.dt.month.fillna(-1).astype(int)
            frame[f"{{column}}_day"] = converted.dt.day.fillna(-1).astype(int)
            frame[f"{{column}}_dayofweek"] = converted.dt.dayofweek.fillna(-1).astype(int)
    return frame


def _encode_categories(train, test, excluded_columns):
    for column in train.columns:
        if column in excluded_columns:
            continue
        if train[column].dtype == "object" or str(train[column].dtype) == "category":
            combined = pd.concat([train[column], test[column]], axis=0).astype(str)
            categories = pd.Categorical(combined)
            train[column] = categories.codes[: len(train)]
            test[column] = categories.codes[len(train) :]
    return train, test


def preprocess(train, test, cfg):
    """データ種別に応じた基本前処理を行う。"""
    train = train.copy()
    test = test.copy()
    data_type = cfg.get("data", {{}}).get("type", "{default_type}")
    target_column = cfg.get("training", {{}}).get("target_column", "target")
    group_column = cfg.get("training", {{}}).get("group_column", "group_id")
    excluded_columns = {{target_column, group_column}}

    train = _basic_fillna(train, target_column)
    test = _basic_fillna(test, target_column)
    train = _convert_datetime_columns(train, excluded_columns)
    test = _convert_datetime_columns(test, excluded_columns)

    if data_type in {{"text", "multi_modal"}}:
        train, _ = _convert_text_columns(train, excluded_columns)
        test, _ = _convert_text_columns(test, excluded_columns)

    if data_type == "image":
        image_columns = [
            column
            for column in train.columns
            if any(token in column.lower() for token in ["image", "img", "path", "file"])
        ]
        for column in image_columns:
            train[column] = train[column].astype(str)
            test[column] = test[column].astype(str)
            train[f"{{column}}_path_len"] = train[column].str.len()
            test[f"{{column}}_path_len"] = test[column].str.len()

    if data_type in {{"tabular", "text", "time_series", "multi_modal", "image"}}:
        train, test = _encode_categories(train, test, excluded_columns)

    return train, test


def get_features(train, cfg):
    """学習に使う特徴量カラム名を返す。"""
    target_column = cfg.get("training", {{}}).get("target_column", "target")
    group_column = cfg.get("training", {{}}).get("group_column", "group_id")
    excluded_columns = {{target_column, group_column}}
    for candidate in ["id", "ID", "index"]:
        if candidate in train.columns:
            excluded_columns.add(candidate)
    features = [column for column in train.columns if column not in excluded_columns]
    return features
'''


def build_model_py(spec: Spec) -> str:
    if spec.model_type == "lgbm":
        model_class = (
            "LGBMClassifier"
            if spec.task_type in {"binary_classification", "multiclass_classification"}
            else "LGBMRegressor"
        )
        return f'''# src/model.py

from __future__ import annotations

import pickle
from pathlib import Path

from lightgbm import {model_class}


def get_model(cfg):
    """設定に応じてLightGBMモデルを返す。"""
    params = cfg["model"].get("params", {{}}).copy()
    return {model_class}(**params)


def save_model(model, model_path, cfg=None):
    """pickleでモデルを保存する。"""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fp:
        pickle.dump(model, fp)


def load_model(model_path, cfg=None):
    """pickleからモデルを読み込む。"""
    with Path(model_path).open("rb") as fp:
        return pickle.load(fp)
'''
    if spec.model_type == "xgboost":
        model_class = (
            "XGBClassifier"
            if spec.task_type in {"binary_classification", "multiclass_classification"}
            else "XGBRegressor"
        )
        return f'''# src/model.py

from __future__ import annotations

import pickle
from pathlib import Path

from xgboost import {model_class}


def get_model(cfg):
    """設定に応じてXGBoostモデルを返す。"""
    params = cfg["model"].get("params", {{}}).copy()
    return {model_class}(**params)


def save_model(model, model_path, cfg=None):
    """pickleでモデルを保存する。"""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fp:
        pickle.dump(model, fp)


def load_model(model_path, cfg=None):
    """pickleからモデルを読み込む。"""
    with Path(model_path).open("rb") as fp:
        return pickle.load(fp)
'''
    if spec.model_type == "catboost":
        model_class = (
            "CatBoostClassifier"
            if spec.task_type in {"binary_classification", "multiclass_classification"}
            else "CatBoostRegressor"
        )
        return f'''# src/model.py

from __future__ import annotations

from pathlib import Path

from catboost import {model_class}


def get_model(cfg):
    """設定に応じてCatBoostモデルを返す。"""
    params = cfg["model"].get("params", {{}}).copy()
    return {model_class}(**params)


def save_model(model, model_path, cfg=None):
    """CatBoostの保存形式でモデルを保存する。"""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)


def load_model(model_path, cfg=None):
    """CatBoostモデルを読み込む。"""
    model = get_model(cfg)
    model.load_model(Path(model_path))
    return model
'''
    if spec.model_type == "sklearn":
        model_class = (
            "RandomForestClassifier"
            if spec.task_type in {"binary_classification", "multiclass_classification"}
            else "RandomForestRegressor"
        )
        return f'''# src/model.py

from __future__ import annotations

import pickle
from pathlib import Path

from sklearn.ensemble import {model_class}


def get_model(cfg):
    """設定に応じてscikit-learnモデルを返す。"""
    params = cfg["model"].get("params", {{}}).copy()
    return {model_class}(**params)


def save_model(model, model_path, cfg=None):
    """pickleでモデルを保存する。"""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fp:
        pickle.dump(model, fp)


def load_model(model_path, cfg=None):
    """pickleからモデルを読み込む。"""
    with Path(model_path).open("rb") as fp:
        return pickle.load(fp)
'''
    if spec.model_type == "ensemble":
        estimators = (
            '[("rf", RandomForestClassifier(**params)), ("et", ExtraTreesClassifier(**params))]'
            if spec.task_type in {"binary_classification", "multiclass_classification"}
            else '[("rf", RandomForestRegressor(**params)), ("et", ExtraTreesRegressor(**params))]'
        )
        ensemble_class = (
            "VotingClassifier"
            if spec.task_type in {"binary_classification", "multiclass_classification"}
            else "VotingRegressor"
        )
        imports = (
            "from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier"
            if spec.task_type in {"binary_classification", "multiclass_classification"}
            else "from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, VotingRegressor"
        )
        extra_kwarg = ', voting="soft"' if ensemble_class == "VotingClassifier" else ""
        return f'''# src/model.py

from __future__ import annotations

import pickle
from pathlib import Path

{imports}


def get_model(cfg):
    """複数モデルの平均化アンサンブルを返す。"""
    params = cfg["model"].get("params", {{}}).copy()
    estimators = {estimators}
    return {ensemble_class}(estimators=estimators{extra_kwarg})


def save_model(model, model_path, cfg=None):
    """pickleでモデルを保存する。"""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fp:
        pickle.dump(model, fp)


def load_model(model_path, cfg=None):
    """pickleからモデルを読み込む。"""
    with Path(model_path).open("rb") as fp:
        return pickle.load(fp)
'''
    if spec.model_type == "pytorch":
        out_dim = 1 if spec.task_type != "multiclass_classification" else "num_classes"
        activation = (
            "return self.network(x)"
            if spec.task_type == "multiclass_classification"
            else "return self.network(x).squeeze(1)"
        )
        return f'''# src/model.py

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class TabularMLP(nn.Module):
    """数値特徴量を受け取るシンプルなMLP。"""

    def __init__(self, input_dim, hidden_dim, dropout, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        {activation}


def get_model(cfg, input_dim, num_classes=1):
    """設定に応じてPyTorchモデルを返す。"""
    params = cfg["model"].get("params", {{}})
    output_dim = {out_dim}
    return TabularMLP(
        input_dim=input_dim,
        hidden_dim=params.get("hidden_dim", 256),
        dropout=params.get("dropout", 0.2),
        output_dim=output_dim,
    )


def save_model(model, model_path, cfg=None):
    """state_dictを保存する。"""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)


def load_model(model_path, cfg, input_dim, num_classes=1):
    """state_dictを読み込む。"""
    model = get_model(cfg, input_dim=input_dim, num_classes=num_classes)
    model.load_state_dict(torch.load(Path(model_path), map_location="cpu"))
    model.eval()
    return model
'''
    if spec.model_type == "tensorflow":
        output_units = (
            "num_classes"
            if spec.task_type == "multiclass_classification"
            else "1"
        )
        output_activation = (
            '"softmax"'
            if spec.task_type == "multiclass_classification"
            else '"sigmoid"'
            if spec.task_type == "binary_classification"
            else '"linear"'
        )
        loss = (
            '"sparse_categorical_crossentropy"'
            if spec.task_type == "multiclass_classification"
            else '"binary_crossentropy"'
            if spec.task_type == "binary_classification"
            else '"mse"'
        )
        return f'''# src/model.py

from __future__ import annotations

from pathlib import Path

import tensorflow as tf


def get_model(cfg, input_dim, num_classes=1):
    """設定に応じてTensorFlowモデルを返す。"""
    params = cfg["model"].get("params", {{}})
    hidden_units = params.get("hidden_units", [256, 128])
    dropout = params.get("dropout", 0.2)
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense({output_units}, activation={output_activation})(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params.get("lr", 1e-3)),
        loss={loss},
    )
    return model


def save_model(model, model_path, cfg=None):
    """Keras形式でモデルを保存する。"""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)


def load_model(model_path, cfg=None, input_dim=None, num_classes=1):
    """Kerasモデルを読み込む。"""
    return tf.keras.models.load_model(Path(model_path))
'''
    raise ValueError(f"unsupported model type: {spec.model_type}")


def build_train_py(spec: Spec) -> str:
    header = '''# src/train.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

from src.data import get_features, load_data, preprocess
'''
    if spec.model_type in {"lgbm", "xgboost", "catboost", "sklearn", "ensemble"}:
        imports = 'from src.model import get_model, save_model\n'
    elif spec.model_type in {"pytorch", "tensorflow"}:
        imports = 'from src.model import get_model, save_model\nfrom src.utils import seed_everything\n'
    else:
        imports = 'from src.model import get_model, save_model\n'

    common = f'''

TASK_TYPE = "{spec.task_type}"
MODEL_TYPE = "{spec.model_type}"


def _mapk(y_true, y_score, k=10):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score)
    topk = y_true[order][:k]
    hits = 0
    score = 0.0
    for rank, label in enumerate(topk, start=1):
        if label:
            hits += 1
            score += hits / rank
    denominator = max(1, min(int(y_true.sum()), k))
    return score / denominator


def evaluate(y_true, y_pred, metric):
    """指定された評価指標でスコアを計算する。"""
    metric = metric.lower()
    if metric == "auc":
        return roc_auc_score(y_true, y_pred)
    if metric == "rmse":
        return mean_squared_error(y_true, y_pred, squared=False)
    if metric == "logloss":
        return log_loss(y_true, y_pred)
    if metric == "accuracy":
        if np.asarray(y_pred).ndim == 2:
            pred_label = np.argmax(y_pred, axis=1)
        else:
            pred_label = (np.asarray(y_pred) >= 0.5).astype(int)
        return accuracy_score(y_true, pred_label)
    if metric == "f1":
        if np.asarray(y_pred).ndim == 2:
            pred_label = np.argmax(y_pred, axis=1)
            return f1_score(y_true, pred_label, average="macro")
        pred_label = (np.asarray(y_pred) >= 0.5).astype(int)
        return f1_score(y_true, pred_label)
    if metric in {{"map@k", "mapk"}}:
        return _mapk(y_true, y_pred, k=10)
    raise ValueError(f"unsupported metric: {{metric}}")


def _build_splitter(cfg, y):
    n_folds = cfg["data"].get("n_folds", 5)
    seed = cfg["data"].get("seed", 42)
    if TASK_TYPE in {{"binary_classification", "multiclass_classification"}}:
        return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return KFold(n_splits=n_folds, shuffle=True, random_state=seed)


def _predict(model, valid_x):
    if hasattr(model, "predict_proba"):
        prediction = model.predict_proba(valid_x)
        if TASK_TYPE == "binary_classification":
            return prediction[:, 1]
        return prediction
    return model.predict(valid_x)
'''

    if spec.model_type == "lgbm":
        fit_impl = '''

def _fit_model(model, train_x, train_y, valid_x, valid_y, cfg):
    import lightgbm as lgb

    callbacks = [
        lgb.early_stopping(cfg["training"].get("early_stopping_rounds", 100), verbose=False),
        lgb.log_evaluation(0),
    ]
    fit_kwargs = {
        "X": train_x,
        "y": train_y,
        "eval_set": [(valid_x, valid_y)],
        "callbacks": callbacks,
    }
    if TASK_TYPE == "binary_classification":
        fit_kwargs["eval_metric"] = "auc"
    elif TASK_TYPE == "multiclass_classification":
        fit_kwargs["eval_metric"] = "multi_logloss"
    else:
        fit_kwargs["eval_metric"] = "rmse"
    model.fit(**fit_kwargs)
    return model
'''
    elif spec.model_type == "xgboost":
        fit_impl = '''

def _fit_model(model, train_x, train_y, valid_x, valid_y, cfg):
    fit_kwargs = {
        "X": train_x,
        "y": train_y,
        "eval_set": [(valid_x, valid_y)],
        "verbose": False,
    }
    if "early_stopping_rounds" in cfg.get("training", {}):
        fit_kwargs["early_stopping_rounds"] = cfg["training"]["early_stopping_rounds"]
    model.fit(**fit_kwargs)
    return model
'''
    elif spec.model_type == "catboost":
        fit_impl = '''

def _fit_model(model, train_x, train_y, valid_x, valid_y, cfg):
    model.fit(train_x, train_y, eval_set=(valid_x, valid_y), verbose=False)
    return model
'''
    elif spec.model_type in {"sklearn", "ensemble"}:
        fit_impl = '''

def _fit_model(model, train_x, train_y, valid_x, valid_y, cfg):
    del valid_x, valid_y, cfg
    model.fit(train_x, train_y)
    return model
'''
    elif spec.model_type == "pytorch":
        fit_impl = '''

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _predict(model, valid_x):
    model.eval()
    valid_tensor = torch.tensor(valid_x.values, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(valid_tensor).cpu().numpy()
    if TASK_TYPE == "binary_classification":
        return 1.0 / (1.0 + np.exp(-prediction))
    if TASK_TYPE == "multiclass_classification":
        logits = prediction - prediction.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)
    return prediction


def _fit_model(model, train_x, train_y, valid_x, valid_y, cfg):
    params = cfg["model"].get("params", {})
    batch_size = params.get("batch_size", 256)
    epochs = params.get("epochs", 10)
    lr = params.get("lr", 1e-3)
    weight_decay = params.get("weight_decay", 1e-5)

    train_dataset = TensorDataset(
        torch.tensor(train_x.values, dtype=torch.float32),
        torch.tensor(train_y.values, dtype=torch.long if TASK_TYPE == "multiclass_classification" else torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if TASK_TYPE == "binary_classification":
        criterion = nn.BCEWithLogitsLoss()
    elif TASK_TYPE == "multiclass_classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            if TASK_TYPE == "binary_classification":
                loss = criterion(logits, batch_y.float())
            elif TASK_TYPE == "multiclass_classification":
                loss = criterion(logits, batch_y.long())
            else:
                loss = criterion(logits, batch_y.float())
            loss.backward()
            optimizer.step()
    return model
'''
    elif spec.model_type == "tensorflow":
        fit_impl = '''

def _predict(model, valid_x):
    prediction = model.predict(valid_x.values, verbose=0)
    if TASK_TYPE == "binary_classification":
        return prediction.reshape(-1)
    return prediction


def _fit_model(model, train_x, train_y, valid_x, valid_y, cfg):
    import tensorflow as tf

    params = cfg["model"].get("params", {})
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=3,
            restore_best_weights=True,
            monitor="val_loss",
        )
    ]
    model.fit(
        train_x.values,
        train_y.values,
        validation_data=(valid_x.values, valid_y.values),
        epochs=params.get("epochs", 10),
        batch_size=params.get("batch_size", 256),
        verbose=0,
        callbacks=callbacks,
    )
    return model
'''
    else:
        raise ValueError(f"unsupported model type: {spec.model_type}")

    if spec.model_type in {"pytorch", "tensorflow"}:
        model_setup = '''

def _build_model(cfg, train_x, train_y):
    if TASK_TYPE == "multiclass_classification":
        num_classes = int(pd.Series(train_y).nunique())
    else:
        num_classes = 1
    return get_model(cfg, input_dim=train_x.shape[1], num_classes=num_classes)
'''
        model_suffix = ".pt" if spec.model_type == "pytorch" else ".keras"
    else:
        model_setup = '''

def _build_model(cfg, train_x, train_y):
    del train_x, train_y
    return get_model(cfg)
'''
        model_suffix = ".cbm" if spec.model_type == "catboost" else ".pkl"

    run_cv = f'''
{model_setup}


def run_cv(cfg, exp_dir):
    """交差検証を実行して平均スコアを返す。"""
    exp_dir = Path(exp_dir)
    weight_dir = exp_dir / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)

    train, test = load_data(cfg)
    train, test = preprocess(train, test, cfg)
    features = get_features(train, cfg)
    target_column = cfg["training"].get("target_column", "target")
    metric = cfg["task"]["metric"]
    train_x = train[features]
    train_y = train[target_column]
    splitter = _build_splitter(cfg, train_y)

    fold_scores = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train_x, train_y), start=1):
        fold_train_x = train_x.iloc[train_idx]
        fold_valid_x = train_x.iloc[valid_idx]
        fold_train_y = train_y.iloc[train_idx]
        fold_valid_y = train_y.iloc[valid_idx]

        model = _build_model(cfg, fold_train_x, fold_train_y)
        model = _fit_model(model, fold_train_x, fold_train_y, fold_valid_x, fold_valid_y, cfg)
        valid_pred = _predict(model, fold_valid_x)
        fold_score = evaluate(fold_valid_y, valid_pred, metric)
        fold_scores.append(fold_score)
        save_model(model, weight_dir / f"fold{{fold}}{model_suffix}", cfg)

    return float(np.mean(fold_scores))
'''
    return header + imports + common + fit_impl + run_cv


def build_utils_py() -> str:
    return '''# src/utils.py

from __future__ import annotations

import contextlib
import copy
import random
import time
from pathlib import Path

import numpy as np
import yaml


def seed_everything(seed):
    """乱数シードを固定する。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


@contextlib.contextmanager
def timer(name="process"):
    """処理時間を計測するコンテキストマネージャ。"""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"[timer] {name}: {elapsed:.2f}s")


def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(base_path, exp_path):
    """base設定と実験設定をdeep mergeして返す。"""
    with Path(base_path).open("r", encoding="utf-8") as fp:
        base_cfg = yaml.safe_load(fp)
    with Path(exp_path).open("r", encoding="utf-8") as fp:
        exp_cfg = yaml.safe_load(fp)
    return _deep_merge(base_cfg, exp_cfg)
'''


def build_run_experiment_py() -> str:
    return '''# run_experiment.py

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from src.train import run_cv
from src.utils import load_config, seed_everything, timer


def parse_args():
    parser = argparse.ArgumentParser(description="Kaggle実験を実行する。")
    parser.add_argument("--config", required=True, help="configs/exp001.yaml のような実験設定")
    parser.add_argument("--note", default="", help="実験メモ")
    return parser.parse_args()


def append_experiment_log(log_path, row):
    """実験ログCSVに追記する。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    exists = log_path.exists()
    with log_path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["exp_name", "datetime", "cv_score", "model", "params", "note"],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    cfg = load_config("configs/base.yaml", args.config)
    seed_everything(cfg["data"].get("seed", 42))

    exp_name = cfg["exp_name"]
    exp_dir = Path(cfg["output"]["exp_dir_base"]) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    with timer(exp_name):
        cv_score = run_cv(cfg, exp_dir)

    append_experiment_log(
        Path("logs/experiments.csv"),
        {
            "exp_name": exp_name,
            "datetime": datetime.now().isoformat(timespec="seconds"),
            "cv_score": cv_score,
            "model": cfg["model"]["name"],
            "params": str(cfg["model"].get("params", {})),
            "note": args.note,
        },
    )
    print(f"cv_score={cv_score:.6f}")


if __name__ == "__main__":
    main()
'''


def build_kernel_metadata_json(spec: Spec) -> str:
    data = {
        "id": f"{spec.kaggle_username}/{spec.kernel_slug}",
        "title": f"{spec.competition_slug} - experiment",
        "code_file": "run_experiment.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": spec.use_gpu,
        "enable_tpu": False,
        "enable_internet": spec.enable_internet,
        "competition_sources": [spec.competition_slug],
    }
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def build_push_sh() -> str:
    return '''#!/usr/bin/env bash
# push.sh

set -euo pipefail

EXP="${1:-exp001}"
KERNEL_ID="$(python - <<'PY'
import json
with open("kernel-metadata.json", "r", encoding="utf-8") as fp:
    data = json.load(fp)
print(data["id"])
PY
)"

kaggle kernels push -p .

while true; do
  STATUS="$(kaggle kernels status "${KERNEL_ID}" | awk 'NR==2 {print $2}')"
  echo "status=${STATUS}"
  if [[ "${STATUS}" == "complete" || "${STATUS}" == "error" ]]; then
    mkdir -p "experiments/${EXP}"
    kaggle kernels output "${KERNEL_ID}" -p "experiments/${EXP}"
    break
  fi
  sleep 30
done
'''


def notebook_cells(spec: Spec) -> list[dict[str, Any]]:
    metric_title = {
        "binary_classification": 'train["target"].value_counts().plot(kind="bar", title="Target Distribution")',
        "multiclass_classification": 'train["target"].value_counts().plot(kind="bar", title="Target Distribution")',
        "regression": 'train["target"].plot(kind="hist", bins=30, title="Target Distribution")',
        "ranking": 'train["target"].plot(kind="hist", bins=30, title="Target Distribution")',
    }[spec.task_type]
    cells = [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# notebooks/eda.ipynb\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from pathlib import Path\n",
                'sns.set_theme(style="whitegrid")\n',
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f'input_dir = Path("input/{spec.competition_slug}")\n',
                'train = pd.read_csv(input_dir / "train.csv")\n',
                'test = pd.read_csv(input_dir / "test.csv")\n',
                "print(train.shape)\n",
                "print(test.shape)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "display(train.isna().sum().sort_values(ascending=False).head(20))\n",
                "display(test.isna().sum().sort_values(ascending=False).head(20))\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(8, 4))\n",
                f"{metric_title}\n",
                "plt.tight_layout()\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "numeric_columns = train.select_dtypes(include=['number']).columns.tolist()\n",
                'if "target" in numeric_columns:\n',
                '    numeric_columns.remove("target")\n',
                "display(train[numeric_columns].describe().T.head(30))\n",
            ],
        },
    ]
    return cells


def build_notebook(spec: Spec) -> str:
    notebook = {
        "cells": notebook_cells(spec),
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, ensure_ascii=False, indent=2) + "\n"


def build_gitignore() -> str:
    return '''# .gitignore

experiments/*/weights/
logs/
__pycache__/
*.pyc
.ipynb_checkpoints/
input/
'''


def build_readme(spec: Spec) -> str:
    return f'''<!-- README.md -->

# {spec.competition_slug}

- タスク: {spec.task_type}
- 評価指標: {spec.metric}

## フォルダ構成

```text
.
├── configs/
├── notebooks/
├── src/
├── kernel-metadata.json
├── push.sh
└── run_experiment.py
```

## セットアップ

1. 仮想環境を作成して有効化します。

```bash
uv venv
source .venv/bin/activate
```

2. 必要なライブラリをインストールします。

```bash
uv pip install pandas numpy scikit-learn pyyaml matplotlib seaborn
```

3. モデルに応じた追加ライブラリを入れます。

```bash
uv pip install lightgbm xgboost catboost torch tensorflow
```

4. Kaggle APIを設定します。

```bash
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## 実験実行

```bash
uv run python run_experiment.py --config configs/exp001.yaml --note "first run"
```

## Kaggleへpush

```bash
bash push.sh exp001
```
'''


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_files(spec: Spec) -> dict[str, str]:
    return {
        "configs/base.yaml": "# configs/base.yaml\n\n" + to_yaml(base_config(spec)),
        "configs/exp001.yaml": "# configs/exp001.yaml\n\n" + to_yaml(exp_config(spec)),
        "src/__init__.py": "",
        "src/data.py": build_data_py(spec),
        "src/model.py": build_model_py(spec),
        "src/train.py": build_train_py(spec),
        "src/utils.py": build_utils_py(),
        "run_experiment.py": build_run_experiment_py(),
        "kernel-metadata.json": build_kernel_metadata_json(spec),
        "push.sh": build_push_sh(),
        "notebooks/eda.ipynb": build_notebook(spec),
        ".gitignore": build_gitignore(),
        "README.md": build_readme(spec),
    }


def main() -> None:
    args = parse_args()
    spec = Spec(
        competition_slug=args.competition_slug,
        task_type=args.task_type,
        metric=args.metric,
        model_type=args.model_type,
        data_type=args.data_type,
        kaggle_username=args.kaggle_username,
        kernel_slug=args.kernel_slug,
        use_gpu=args.use_gpu,
        enable_internet=args.enable_internet,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for relative_path, content in build_files(spec).items():
        write_file(output_dir / relative_path, content)

    print(f"generated: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
