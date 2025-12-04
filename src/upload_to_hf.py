from huggingface_hub import HfApi



# https://huggingface.co/settings/tokens token generate
token = ""

# API object
api = HfApi()

# name of repo  
repo_id = "emann123/face-reg-model"

# path to your local model file
model_path = r"D:\Projects_Ml\Face_reg\src\models\best_emotion_model.keras"

# upload models  to Hugging Face
api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="best_emotion_model.keras",
    repo_id=repo_id,
    repo_type="model",
    token=token
)

print("✅ Model uploaded successfully!")
