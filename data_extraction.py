import tarfile
import json


tar_path = "test.tar"
output_dir = "mini_dataset"  

with tarfile.open(tar_path, "r") as tar_ref:
    tar_ref.extract("valid/meta.json", path="temp_meta")
    
with open("temp_meta/valid/meta.json", "r") as f:
    meta = json.load(f)

single_object_videos = []
for video_id, video_info in meta["videos"].items():
    if len(video_info["objects"]) == 1:
        single_object_videos.append(video_id)
    if len(single_object_videos) >= 300:
        break

with tarfile.open(tar_path, "r") as tar_ref:
    members_to_extract = []
    
    for video_id in single_object_videos:
        jpeg_prefix = f"valid/JPEGImages/{video_id}/"
        ann_prefix = f"valid/Annotations/{video_id}/"
        
        for member in tar_ref.getmembers():
            if member.name.startswith((jpeg_prefix, ann_prefix)):
                members_to_extract.append(member)
    
    tar_ref.extractall(path=output_dir, members=members_to_extract)    