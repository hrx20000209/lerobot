from huggingface_hub import HfApi

api = HfApi()

repo_id = "hrx2000/Lerobot_Pi05_Blue_Block"

api.upload_folder(
    folder_path="output_lerobot_train/grab_blue_block/pi05/checkpoints/last",
    repo_id=repo_id,
    repo_type="model",
)

api.create_tag(repo_id, tag="v0.1.3", repo_type="model")