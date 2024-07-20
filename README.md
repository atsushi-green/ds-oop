# Object-Orientation Programming for Data Sience
By applying some of the ideas of object orientation to data science source code, we aim to improve the following items.

- readability
- Reusability
- Maintainability
- Reliability

article: https://zenn.dev/zenkigen_tech/articles/f15988969d9c3f
<div align="center">
  <img src="https://img.shields.io/badge/rye-0.32-F26649?logo=Rye" alt="rye">
  <img src="https://img.shields.io/badge/python-3.11-F26649?logo=python" alt="python">
  <img src="https://img.shields.io/badge/scikitlearn-1.4.2-F26649?logo=scikitlearn" alt="rye">
  
</div>

# setup
Install [Rye](https://rye-up.com/) and run the following command.

```bash
rye sync
```

# Data
Download train.csv and test.csv from https://www.kaggle.com/c/titanic and save them under titanic_data directory.

# Execution

## Data Processing
```bash
# For Rye users:
rye run python src/titanic_data
```

## Unit test
```bash
rye run pytest tests
```


