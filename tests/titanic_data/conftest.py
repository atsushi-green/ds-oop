import random

import pytest
from pandas import DataFrame

SURVIVED_CANDIDATES = [0, 1]
PCLASS_CANDIDATES = [1, 2, 3, None]
SEX_CANDIDATES = ["male", "female", ""]
AGE_CANDIDATES = [i for i in range(0, 100)]
SIBSP_CANDIDATES = [i for i in range(0, 10)]
PARCH_CANDIDATES = [i for i in range(0, 10)]
EMBARKED_CANDIDATES = ["S", "C", "Q", None]


@pytest.fixture
def make_random_titanic_df():
    def _make_random_titanic_df(num_data: int):
        data = {
            "PassengerId": range(1, num_data + 1),
            "Survived": random.choices((SURVIVED_CANDIDATES), k=num_data),
            "Pclass": random.choices((PCLASS_CANDIDATES), k=num_data),
            "Name": [f"Name_{i}" for i in range(1, num_data + 1)],
            "Sex": [random.choice(SEX_CANDIDATES) for _ in range(num_data)],
            "Age": random.choices((AGE_CANDIDATES), k=num_data),
            "SibSp": random.choices((SIBSP_CANDIDATES), k=num_data),
            "Parch": random.choices((PARCH_CANDIDATES), k=num_data),
            "Ticket": [f"Ticket_{i}" for i in range(1, num_data + 1)],
            "Fare": [random.uniform(0, 100) for _ in range(num_data)],
            "Cabin": [f"Cabin_{i}" for i in range(1, num_data + 1)],
            "Embarked": [random.choice(EMBARKED_CANDIDATES) for _ in range(num_data)],
        }
        return DataFrame(data)

    yield _make_random_titanic_df
