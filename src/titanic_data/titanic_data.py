import json
from logging import config, getLogger
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from deepdiff import DeepDiff
from numpy import ndarray
from pandas import DataFrame

from src.utils.ml_utils import to_dummy

FEATURE_COLS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
DUMMY_COLS = ["Pclass", "Sex"]
DIR_HOME = Path(__file__).parent.parent.parent

with open(DIR_HOME / "loggingconf/loggingconf.json") as f:
    config.dictConfig(json.load(f))

logger = getLogger("formatLogger")


class TitanicData:
    DATA_TYPE_MAPPING = {
        "PassengerId": str,
        "Survived": int,
        "Pclass": int,
        "Name": str,
        "Sex": str,
        "Age": int,
        "SibSp": int,
        "Parch": int,
        "Ticket": str,
        "Fare": float,
        "Cabin": str,
        "Embarked": str,
    }
    CABIN_MAPPING = {
        "Z": 0,  # 欠損値扱い
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6,
        "G": 7,
        "T": 8,
    }

    @classmethod
    def get_cabin_index(cls, cabin_initials: str) -> int:
        try:
            return cls.CABIN_MAPPING[cabin_initials]
        except KeyError:
            return 0

    @staticmethod
    def read_data(csv_filepath: Path) -> DataFrame:
        # point: 読み込む機能は別で分けておくことで、ファイルを用意しなくともクラスインスタンスを生成できるようにし、
        # テストしやすくする
        try:
            df = pd.read_csv(csv_filepath)
            logger.info(f"{csv_filepath}から{len(df)}件のデータを読み込みました。")
        except FileNotFoundError:
            logger.error(f"{csv_filepath}が見つかりません。")
            raise FileNotFoundError(f"{csv_filepath}が見つかりません。")
        return df

    def __init__(self, df: DataFrame):
        # point: private変数としてデータを持つことで、データに対する処理が分散しない
        self.__df = df

        # point: データ件数や読み込んだ内容などの大切な情報はloggingで出力する
        logger.info(self.__df.head())

        # point: データの整合性をassert文で確認する
        # 列がちゃんとあるか確認
        assert not DeepDiff(
            self.__df.columns.tolist(),
            list(self.DATA_TYPE_MAPPING.keys()),
            ignore_order=True,
        ), f"列名が一致しません。\n{self.__df.columns.tolist()} v.s. \n{list(self.DATA_TYPE_MAPPING.keys())}"

        assert len(self.__df) > 0, "データが空です。"

        # point: __init__メソッド内では多くのことをやらない
        # クラスを再利用したいときに、__init__メソッドの書き換えが発生しやすくなるため。
        ...

    # point: 可視化用の処理などもクラス内に定義する。
    # 可視化処理はデータ仕様と結びついているので、結局共通化できないことが多いので汎用化は諦める。
    def draw_age_hist(self, savepath: Path) -> None:
        """年齢のヒストグラムを描画する。"""
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title("Histogram of Age")
        ax.set_xlabel("Age")
        ax.set_ylabel("Frequency")
        sns.histplot(self.__df["Age"], ax=ax)
        logger.info(f"{savepath}に年齢のヒストグラムを保存しました。")
        fig.savefig(savepath)
        plt.clf()
        plt.close()
        return

    def remove_missing_Embarked(self) -> DataFrame:
        """Embarked列の欠損値を削除する。クラス変数__dfを更新する。
        Returns:
            DataFrame: 削除後のDataFrame
        """
        num_before = len(self.__df)
        self.__df.dropna(subset=["Embarked"], inplace=True)
        logger.info(f"欠損値を削除しました。{num_before} -> {len(self.__df)}")
        # point: 想定外の動きは止まるようにしておく
        assert (
            len(self.__df) / num_before > 0.5
        ), "Embarkedをdropしたら50%以上削除されてしまいました。"
        return self.__df

    def to_training_dataset(self) -> tuple[ndarray[float, Any], ndarray[float, Any]]:
        vaild_df = self.__df[FEATURE_COLS]
        # x = vaild_df.astype(self.DATA_TYPE_MAPPING)
        dummied_df = to_dummy(vaild_df[FEATURE_COLS], DUMMY_COLS)

        x = dummied_df.values
        y = self.__df["Survived"]
        assert len(x) == len(y)
        return x, y

    def calc_cabin_feature(self) -> np.ndarray[int, Any]:
        """Cabin(部屋番号)の頭文字を特徴量として追加する。

        Returns:
            np.ndarray[int, Any]: Cacinの頭文字をone-hot vectorに変換したもの
        """
        # 欠損値をZで埋め、頭文字を数値に変換
        cabin_initials_digits = np.array(
            self.__df["Cabin"].fillna("Z").apply(lambda x: self.get_cabin_index(x[0])),
            dtype=int,
        )
        logger.info("Cabinの頭文字を数値に変換しました。")
        one_hot_vector_list = []
        for cabin_initials in self.CABIN_MAPPING:
            # one-hot vectorを作る
            one_hot_vector_list.append(
                np.where(
                    cabin_initials_digits == self.CABIN_MAPPING[cabin_initials], 1, 0
                )
            )

        assert len(one_hot_vector_list) == len(self.CABIN_MAPPING)

        return np.array(one_hot_vector_list).T
