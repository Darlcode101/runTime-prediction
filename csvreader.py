import pandas as pd

data = pd.read_csv("joined.csv", low_memory=False)

small = data[
    [
        "athlete_id",
        "event_name",
        "result_time",
        "date_of_event",
        "gender"
    ]
]

print(small.head(20).to_string())