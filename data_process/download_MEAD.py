import os  
import sys 

def download_from_folder(asset_path, access_token, to_path):
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
            file_dist = os.path.join(folder_path, file_name)
            if os.path.exists(file_dist):
                continue
            command = f"curl -H \"{authorization_keys}\" https://www.googleapis.com/drive/v3/files/{file_id}?alt=media -o {file_dist}"
            os.system(command)
            
if __name__ == "__main__":
    asset_path = "dataset/MEAD/meta/files_id.txt"
    access_token = "ya29.a0Ad52N38fbUINDibY_MwG_FtrhT5ZhCI9hYIMheCbvsiAH4USPcMaxV3nSFlNoOLYOOfg386T50P4o7wgImv1H8IpgBtkmngNzCeRwgMNLJbJaexAD6LdkFXIuqaLDhkCI2BOy1Bf5QeH3ECoU1i3wChCMv9nUCRhmm93aCgYKAe0SARMSFQHGX2MiymoexiE_De_g412ivuI1TA0171"
    to_path = "dataset/MEAD"
    download_from_folder(asset_path, access_token, to_path)
    
    