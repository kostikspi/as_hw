import json
import logging
import os
import shutil
from curses.ascii import isascii
from pathlib import Path

import torchaudio
from hw_antispoof.base.base_dataset import BaseDataset
from hw_antispoof.utils import ROOT_PATH
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ASVDataset(BaseDataset):
    def __init__(self, part, data_dir=None, index_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = Path(data_dir)
        self._index_dir = Path(index_dir)
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, part):
        index_path = self._index_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / "LA" / f"ASVspoof2019_LA_{part}" / "flac"

        flac_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".flac") for f in filenames]):
                flac_dirs.add(dirpath)
        for flac_dir in tqdm(
                list(flac_dirs), desc=f"Preparing ASV folders: {part}"
        ):
            flac_dir = Path(flac_dir)
            part_ext = "trn" if part == "train" else "trl"
            protocol_path = self._data_dir / "LA" / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{part}.{part_ext}.txt"
            # trans_path = list(protocol_path.glob("*.txt"))[0]
            with protocol_path.open() as f:
                for line in f:
                    w_id = line.split()[1]
                    w_speaker = line.split()[0]
                    w_label = line.split()[-1].strip()

                    if w_label == 'bonafide':
                        label = [0, 1]
                    else:
                        label = [1, 0]
                    flac_path = flac_dir / f"{w_id}.flac"
                    if not flac_path.exists():  # elem in another part
                        continue
                    t_info = torchaudio.info(str(flac_path))
                    length = t_info.num_frames / t_info.sample_rate
                    index.append(
                        {
                            "path": str(flac_path.absolute().resolve()),
                            "speaker_id": w_speaker,
                            "audio_len": length,
                            "label": label,
                            "true_label": w_label
                        }
                    )
        return index
