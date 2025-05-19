from huggingface_hub import create_repo, upload_folder
from datetime import datetime
# vytvoří nový veřejný nebo soukromý repozitář

#nahraje celý adresář
datetime_now = datetime.now().strftime("%Y-%m-%d")

#create_repo("robe-error-detector", private=False)
upload_folder(
    folder_path="./robe-error-detector",
    repo_id="Hahacko03/robe-error-detector",
    ignore_patterns=["checkpoint*", "trainer_state.json", "*.log", "*.tmp"],
    commit_message=f"RobeCzech based model for error detection in Czech sentences - uploaded {datetime_now}",
)
#create_repo("robe-mask-corrector", private=False)
upload_folder(
    folder_path="./robe-mask-corrector",
    repo_id="Hahacko03/robe-mask-corrector",
    ignore_patterns=["checkpoint*", "trainer_state.json", "*.log", "*.tmp"],
    commit_message=f"RobeCzech based model for error correction in Czech sentences - uploaded {datetime_now}",
)