import joblib

file_path = "metadata.joblib"   # same folder as this script

# 1. Load existing metadata (binary → Python dict)
metadata = joblib.load(file_path)
print("Before:", metadata)

# 2. Change threshold
metadata["threshold"] = 0.7     # or 0.7

# 3. Save back to the same binary file
joblib.dump(metadata, file_path)

# 4. Confirm
updated = joblib.load(file_path)
print("After:", updated)
print("New threshold:", updated["threshold"])