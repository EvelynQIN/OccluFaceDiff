import os  
import sys 

def download_from_folder(asset_path, access_token, to_path, download_video=False):
    """
    Args:
        asset_path: path to the fild ids txt (get from gdown --folder)
        folder_id: google drive folder id
        access_token: bear token from google api
    """
    with open(asset_path, 'r') as fp:
        lines = fp.readlines()
    authorization_keys = "Authorization: Bearer " + access_token
    for line in lines:
        line_list = line.strip().split(" ")
        if line_list[0] == "Retrieving":
            subject_id = line_list[-1]
            print(f"-----------Processing {subject_id}")
            folder_path = os.path.join(to_path, subject_id)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        else:
            file_id, file_name = line_list[-2:]
            print(f"{file_id} _ {file_name}")
            if not download_video and 'video' in file_name:
                continue
            file_dist = os.path.join(folder_path, file_name)
            if os.path.exists(file_dist):
                continue
            command = f"curl -H \"{authorization_keys}\" https://www.googleapis.com/drive/v3/files/{file_id}?alt=media -o {file_dist}"
            os.system(command)

def unzip_assets(data_folder, asset_name='audio'):
    for subject in os.scandir(data_folder):
        if subject.name == 'meta':
            continue
        asset_path = os.path.join(subject.path, f"{asset_name}.tar")
        to_folder = os.path.join(subject.path, asset_name)
        if os.path.exists(to_folder):
            continue 
        os.system(
            f"tar -xvf {asset_path} -C {subject.path}"
        )
        os.system(f"rm {asset_path}")
        
if __name__ == "__main__":
    asset_path = "dataset/MEAD/meta/files_id.txt"
    access_token = "ya29.a0AXooCgvdWwY-PLjM0SG9A3OM99wj9yqVnbskHkd0QhSvUwrvjyGyg-7nimFNQp6v1G3b7fTAiSBriCtm5hOQ1Aga42fYvhEYVF8-OUtnuOk66diNK57waMppWTtevDx-q_-yYnod2bOnfd5PTk2Cv7Hh9IWRFjYVWhvEaCgYKATYSARMSFQHGX2MirMjNEzdQgfJIiWGQ17P7KQ0171"
    to_path = "dataset/MEAD"
    download_video = False
    # download_from_folder(asset_path, access_token, to_path, download_video)
    
    unzip_assets(to_path, asset_name="audio")
    # unzip_assets(to_path, asset_name="video")