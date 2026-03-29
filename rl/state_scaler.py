import numpy as np
import pickle
import joblib
from sklearn.preprocessing import MinMaxScaler


def normalize_replay_buffer(
    buffer_path="models/rl/replay_buffer.pkl",
    output_path="models/rl/replay_buffer_normalized.pkl",
    scaler_path="models/rl/state_scaler.pkl"
):

    # Load buffer
    with open(buffer_path, "rb") as f:
        buffer = pickle.load(f)

    states, actions, rewards, next_states, dones = zip(*buffer)

    states = np.array(states)
    next_states = np.array(next_states)

    # Fit scaler ONLY on states
    scaler = MinMaxScaler()
    scaler.fit(states)

    # Transform
    states_norm = scaler.transform(states)
    next_states_norm = scaler.transform(next_states)

    # Rebuild buffer
    buffer_norm = list(zip(states_norm, actions, rewards, next_states_norm, dones))

    # Save normalized buffer
    with open(output_path, "wb") as f:
        pickle.dump(buffer_norm, f)

    # Save scaler
    joblib.dump(scaler, scaler_path)

    print("✅ Replay buffer normalized and saved")
    print(f"→ {output_path}")
    print(f"→ {scaler_path}")


if __name__ == "__main__":
    normalize_replay_buffer()