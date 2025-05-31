
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from google.colab import files

print("⬆️ Upload final_pokemon.csv")
pokemon_upload = files.upload()

print("⬆️ Upload final_combats.csv or train.csv")
combat_upload = files.upload()

pokemon_filename = list(pokemon_upload.keys())[0]
combat_filename = list(combat_upload.keys())[0]

pokemon_data = pd.read_csv(pokemon_filename)
combat_data = pd.read_csv(combat_filename)


pokemon_data.set_index("#", inplace=True)


def get_pokemon_stats(poke_id):
    stats = pokemon_data.loc[poke_id][["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]]
    return stats.values.astype(np.int64)


X = []
y = []

for _, row in combat_data.iterrows():
    try:
        p1_stats = get_pokemon_stats(row["First_pokemon"])
        p2_stats = get_pokemon_stats(row["Second_pokemon"])
    except KeyError:
        continue

    features = np.concatenate([p1_stats, p2_stats])
    X.append(features)
    y.append(0 if row["Winner"] == row["First_pokemon"] else 1)

X = np.array(X)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(12,)))
model.add(Dense(32, activation='relu')),
model.add(Dense(16, activation='relu')),
model.add(Dense(8, activation='relu')),
model.add(Dense(4, activation='relu')),
model.add(Dense(2, activation='relu')),
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)


loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")
