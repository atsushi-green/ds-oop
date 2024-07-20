import numpy as np
import pytest  # noqa


from src.titanic_data.titanic_data import TitanicData


class TestTitanicData:
    @pytest.mark.parametrize(
        "cabin_list, expected",
        [
            (
                [
                    "A123",
                    "B45",
                    "C78",
                    "D90",
                    "E10",
                    "F35",
                    "G56",
                    "T12",
                    "存在しないキャビン",
                    "Z",
                    None,
                ],
                np.array(
                    [
                        [0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                ),
            ),
        ],
    )
    def test_calc_cabin_feature(
        self,
        cabin_list: list[str],
        expected: list[int],
        make_random_titanic_df: callable,
    ):
        # arrange
        df = make_random_titanic_df(len(cabin_list))
        df["Cabin"] = cabin_list
        training_data = TitanicData(df)

        # act
        ans = training_data.calc_cabin_feature()

        # assert
        assert np.all(ans == np.array(expected))
