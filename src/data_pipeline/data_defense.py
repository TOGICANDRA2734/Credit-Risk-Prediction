import pandas as pd


def data_defense_checker(input_data: pd.DataFrame, params: dict) -> None:
    try:
        print("===== Start Data Defense Checker =====")
        
        # Check data types
        object_cols = input_data[params["features"]].select_dtypes("object").columns.to_list()
        int_cols = input_data[params["features"]].select_dtypes("int64").columns.to_list()
        float_cols = input_data[params["features"]].select_dtypes("float64").columns.to_list()
        
        # Fix the object columns comparison
        expected_object_cols = set(params["object_columns"])
        
        assert set(object_cols) == expected_object_cols, f"Mismatch in object columns. Expected: {expected_object_cols}, Got: {set(object_cols)}"
        assert set(int_cols) == set(params["int64_columns"]), f"Mismatch in integer columns. Expected: {set(params['int64_columns'])}, Got: {set(int_cols)}"
        
        # Check specific column values
        # Only check yes/no for relevant columns
        binary_columns = ['cb_person_default_on_file']
        for col in binary_columns:
            assert set(input_data[col]).issubset(set(params["value_status"])), f"Invalid values in {col}"
        
        # Check numeric ranges (you can add specific range checks here)
        assert (input_data['person_age'] > 0).all(), "Invalid age values"
        assert (input_data['person_income'] >= 0).all(), "Invalid income values"
        assert (input_data['loan_amnt'] > 0).all(), "Invalid loan amount values"
        # assert (input_data['loan_int_rate'] >= 0).all(), "Invalid interest rate values"
        
        print("All data defense checks passed successfully!")
        
    except AssertionError as e:
        print(f"Data validation failed: {str(e)}")
        raise Exception("Failed Data Defense Checker") from e
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise Exception("Failed Data Defense Checker") from e
    finally:
        print("===== Finish Data Defense Checker =====")
