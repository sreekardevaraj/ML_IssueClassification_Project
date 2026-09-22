import pandas as pd

df = pd.read_csv("data/cases_5000.csv")
skew = pd.concat([
    df[df.label == "network"].head(400),
    df[df.label != "network"].sample(300, random_state=1),
])
skew["text"] = skew["text"] + " Additional verbose context that was not in the reference data."
skew.to_csv("data/new_cases.csv", index=False)
print("wrote", len(skew), "rows")
