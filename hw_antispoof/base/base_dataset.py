import logging
import random
from typing import List

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from hw_antispoof.base.base_text_encoder import BaseTextEncoder
from hw_antispoof.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            config_parser: ConfigParser,
            wave_augs=None,
            spec_augs=None,
            limit=None,
            max_audio_length=None,
            max_text_length=None,
    ):
        self.config_parser = config_parser
        self.wave_augs = wave_augs
        self.spec_augs = spec_augs
        self.log_spec = config_parser["preprocessing"]["log_spec"]
        max_text_length = None
        self._assert_index_is_valid(index)
        index = self._filter_records_from_dataset(index, max_audio_length, max_text_length, limit)
        # it's a good idea to sort index by audio length
        # It would be easier to write length-based batch samplers later
        index = self._sort_index(index)
        self._index: List[dict] = index

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio_wave = self.load_audio(audio_path)

        label = data_dict["label"]
        return {
            "audio": audio_wave,
            "label": label,
            "duration": audio_wave.size(1) / self.config_parser["preprocessing"]["sr"],
            "audio_path": audio_path,
            "audio_len": data_dict['audio_len'],
            "true_label": data_dict["true_label"]
        }

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path, num_frames=64000)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        while audio_tensor.shape[-1] < 64000:
            pad_len = 64000 - audio_tensor.shape[-1]
            to_pad = audio_tensor[:, 0:pad_len]
            audio_tensor = torch.cat([audio_tensor, to_pad], dim=1)
        return audio_tensor[:, 0:64000]

    def process_wave(self, audio_tensor_wave: Tensor):
        with torch.no_grad():
            if self.wave_augs is not None:
                audio_tensor_wave = self.wave_augs(audio_tensor_wave)
            wave2spec = self.config_parser.init_obj(
                self.config_parser["preprocessing"]["spectrogram"],
                torchaudio.transforms,
            )
            audio_tensor_spec = wave2spec(audio_tensor_wave)
            if self.spec_augs is not None:
                audio_tensor_spec = self.spec_augs(audio_tensor_spec)
            if self.log_spec:
                audio_tensor_spec = torch.log(audio_tensor_spec + 1e-5)
            return audio_tensor_wave, audio_tensor_spec

    @staticmethod
    def _filter_records_from_dataset(
            index: list, max_audio_length, max_text_length, limit
    ) -> list:
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = np.array([el["audio_len"] for el in index]) >= max_audio_length
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        initial_size = len(index)
        if max_text_length is not None:
            exceeds_text_length = (
                    np.array(
                        [len(BaseTextEncoder.normalize_text(el["text"])) for el in index]
                    )
                    >= max_text_length
            )
            _total = exceeds_text_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_text_length} characters. Excluding them."
            )
        else:
            exceeds_text_length = False

        records_to_filter = exceeds_text_length | exceeds_audio_length

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total}({_total / initial_size:.1%}) records  from dataset"
            )

        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "audio_len" in entry, (
                "Each dataset item should include field 'audio_len'"
                " - duration of audio (in seconds)."
            )
            assert "path" in entry, (
                "Each dataset item should include field 'path'" " - path to audio file."
            )
