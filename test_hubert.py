from fairseq import checkpoint_utils, tasks, utils
import os
from fairseq.dataclass.configs import GenerationConfig
from jiwer import wer, cer
import tempfile
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from argparse import Namespace
import argparse

separator = Separator(phone='-', word=' ')
backend = EspeakBackend('en-us', words_mismatch='ignore', with_stress=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mead_25fps', help='dataset name')

    args = parser.parse_args()

    ckpt_path = "pretrained/av_hubert/self_large_vox_433h.pt" # download this from https://facebookresearch.github.io/av_hubert/

    utils.import_user_module(Namespace(user_dir='external/av_hubert/avhubert'))

    modalities = ["video"]
    gen_subset = "test"
    gen_cfg = GenerationConfig(beam=1)
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    models = [model.eval().cuda() for model in models]
    saved_cfg.task.modalities = modalities