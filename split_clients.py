import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"D:\deepfake_training_gpu\face_train.csv")

real_df = df[df["label"] == 0]
fake_df = df[df["label"] == 1]

real_c1, real_temp = train_test_split(
    real_df,
    test_size=2/3,
    random_state=42
)

real_c2, real_c3 = train_test_split(
    real_temp,
    test_size=0.5,
    random_state=42
)

fake_c1, fake_temp = train_test_split(
    fake_df,
    test_size=2/3,
    random_state=42
)

fake_c2, fake_c3 = train_test_split(
    fake_temp,
    test_size=0.5,
    random_state=42
)

client1 = pd.concat([real_c1, fake_c1]).sample(frac=1, random_state=42)
client2 = pd.concat([real_c2, fake_c2]).sample(frac=1, random_state=42)
client3 = pd.concat([real_c3, fake_c3]).sample(frac=1, random_state=42)

client1.to_csv("face_train_client1.csv", index=False)
client2.to_csv("face_train_client2.csv", index=False)
client3.to_csv("face_train_client3.csv", index=False)

print("Client CSV files created!")
print("Client1:", len(client1))
print("Client2:", len(client2))
print("Client3:", len(client3))