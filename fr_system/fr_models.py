import torch
import os
from fr_system.facenet_pytorch import InceptionResnetV1

from transformers import AutoModel
from huggingface_hub import hf_hub_download
import shutil
import sys


# helpfer function to download huggingface repo and use model
def download(repo_id, path, HF_TOKEN=None):
    os.makedirs(path, exist_ok=True)
    files_path = os.path.join(path, 'files.txt')
    if not os.path.exists(files_path):
        hf_hub_download(repo_id, 'files.txt', token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)
    with open(os.path.join(path, 'files.txt'), 'r') as f:
        files = f.read().split('\n')
    for file in [f for f in files if f] + ['config.json', 'wrapper.py', 'model.safetensors']:
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path):
            hf_hub_download(repo_id, file, token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)

            
# helpfer function to download huggingface repo and use model
def load_model_from_local_path(path, HF_TOKEN=None):
    cwd = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, path)
    model = AutoModel.from_pretrained(path, trust_remote_code=True, token=HF_TOKEN)
    os.chdir(cwd)
    sys.path.pop(0)
    return model


# helpfer function to download huggingface repo and use model
def load_model_by_repo_id(repo_id, save_path, HF_TOKEN=None, force_download=False):
    if force_download:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
    download(repo_id, save_path, HF_TOKEN)
    return load_model_from_local_path(save_path, HF_TOKEN)

# models_dir = os.path.abspath(__file__)
# models_dir = models_dir.split("anonymisation")[0] + "models"
# print(models_dir)
# print(f"[FR_MODELS IMPORT] {models_dir}")
# exit()
class FR_model(torch.nn.Module):
    def __init__(self, model_name, device: str = 'cuda', models_dir="models"):
        super(FR_model, self).__init__()

        if model_name in ["arcface_r50", "arcface_r34"]:
            from fr_system.arcface_torch import get_model
            model_name = model_name.split("_")[-1]
            self.model = get_model(model_name, dropout=0, fp16=False)
            state_dict = torch.load(os.path.join(models_dir,
             f"ms1mv3_arcface_{model_name}_fp16", "backbone.pth"), map_location='cpu')
            self.model.load_state_dict(state_dict, strict=True, assign=True)
            del state_dict
            self.input_size = 112
        elif model_name == "facenet_vggface2":
            self.model = InceptionResnetV1(pretrained='vggface2', classify=False)
            self.input_size = 160
        elif model_name == "facenet_casia":
            self.model = InceptionResnetV1(pretrained='casia-webface', classify=False)
            self.input_size = 160
        elif model_name == "adaface_vit":
            HF_TOKEN = 'put your huggingface token here if the model is private'
            path = os.path.expanduser('~/.cvlface_cache/minchul/cvlface_adaface_vit_base_webface4m')
            repo_id = 'minchul/cvlface_adaface_vit_base_webface4m'
            self.model = load_model_by_repo_id(repo_id, path, HF_TOKEN)
            self.input_size = 112

        self.model.to(device)
        self.model.eval()
        
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model_name = "arcface_r50"  # Change to the desired model name
    # model_name = "facenet_vggface2"
    fr_model = FR_model(model_name)
    fr_model.eval()
    print(f"Loaded {model_name} model successfully.")
    
    # Example input tensor
    example_input = torch.randn(1, 3, 112, 112)  # Batch size of 1, 3 channels, 112x112 image
    output = fr_model(example_input)
    print("Output shape:", output.shape)  # Should match the expected output shape of the model

    #is normalized
    print("Is output normalized?", torch.allclose(output.norm(dim=1), torch.ones(output.size(0)), atol=1e-6))
    print(output.norm(dim=1))








    # arcface = get_model("arcface_r50", fp16=True)
    # arcface.load_state_dict(torch.load("C:\\Users\\Pol\\Documents\\code\\anonymisation\\models\\ms1mv3_arcface_r50_fp16\\backbone.pth"), strict=True)
    # arcface.eval().cuda()



    # # For a model pretrained on VGGFace2
    # model = InceptionResnetV1(pretrained='vggface2').eval()

    # # For a model pretrained on CASIA-Webface
    # model = InceptionResnetV1(pretrained='casia-webface').eval()
