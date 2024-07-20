from pathlib import Path

from src.titanic_data.titanic_data import TitanicData

DIR_HOME = Path(__file__).parent.parent.parent


def main():
    df = TitanicData.read_data(DIR_HOME / "titanic_data/train.csv")
    titanic_data = TitanicData(df)
    titanic_data.draw_age_hist(DIR_HOME / "figs/age_histgram.png")
    one_hot_vector = titanic_data.calc_cabin_feature()
    print(one_hot_vector)


if __name__ == "__main__":
    main()
