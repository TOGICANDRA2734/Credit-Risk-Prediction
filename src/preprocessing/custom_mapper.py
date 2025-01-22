import pandas as pd


def custom_label_encoder(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    MAPPER_VALUE = {
        "N": 0,
        "Y": 1
    }

    for col in params["label_encoder_columns"]:
        data[col] = data[col].replace(MAPPER_VALUE)

    return data
