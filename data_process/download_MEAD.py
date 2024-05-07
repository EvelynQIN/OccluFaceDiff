import os  
import sys 
""" Work around to solve google drive download issue
Go to OAuth 2.0 Playground https://developers.google.com/oauthplayground/
In the Select the Scope box, paste https://www.googleapis.com/auth/drive.readonly
Click Authorize APIs and then Exchange authorization code for tokens
Copy the Access token
"""
def download_from_folder(asset_path, access_token, to_path, download_asset='video'):
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
            if os.path.exists(f"{folder_path}/video"):
                continue
            file_id, file_name = line_list[-2:]
            print(f"{file_id} _ {file_name}")
            if download_asset not in file_name:
                continue
            file_dist = os.path.join(folder_path, file_name)
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
        if asset_name == 'video':
            os.system(
                f"tar -xvf {asset_path} -C {subject.path} video/front"
            )
        else:
            os.system(
                f"tar -xvf {asset_path} -C {subject.path}"
            )
        # os.system(f"rm {asset_path}")
        
if __name__ == "__main__":
    asset_path = "dataset/MEAD/meta/files_id.txt"
    access_token = "ya29.a0AXooCgslAoaGwIMBmq2dA0v4Pjs09yvxFIiPKNZZh_EJbquRaCeUumXJGtAqyytUiynteULYfqSPHVjUFAwZD8ECkoI0Ui2_ejparlWgxcNhFJ2s5lN_2I-SuR4aiQn5IVRvGWI3YqjQd6jQax74Gk9S-KixuhYYr7hdaCgYKAZQSARMSFQHGX2Mi_Fy2gGCHDJRaCFLcFglVQw0171"
    to_path = "dataset/MEAD"
    download_asset = 'video'
    download_from_folder(asset_path, access_token, to_path, download_asset)
    unzip_assets(to_path, asset_name=download_asset)