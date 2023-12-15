import logging
from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

MAX_LEN = 64000


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    # result_batch = {}
    result_batch = {"wave": [], "label": [], "true_label": [],
                    "audio_len": [], "audio_path": []}

    for item in dataset_items:
        if item["audio"].shape[-1] < MAX_LEN:
            pad_len = MAX_LEN - item["audio"].shape[-1]
            to_pad = item["audio"][:, 0:pad_len]
            result_batch["wave"].append(torch.cat([item["audio"], to_pad], dim=1))
        else:
            result_batch["wave"].append(item["audio"])
        result_batch["label"].append(item["label"])
        result_batch["audio_len"].append(item["audio_len"])
        result_batch["true_label"].append(item["true_label"])
        # result_batch["spectrogram"].append(item["spectrogram"].squeeze(0).T)
        # result_batch["text_encoded"].append(item["text_encoded"].squeeze(0))
        # result_batch["spectrogram_length"].append(item["spectrogram"].shape[2])
        # result_batch["text_encoded_length"].append(item["text_encoded"].shape[1])
        # result_batch["text"].append(item["text"])
        result_batch["audio_path"].append(item["audio_path"])

    result_batch["wave"] = torch.stack(result_batch["wave"])
    result_batch["label"] = torch.tensor(result_batch["label"], dtype=torch.long)
    # result_batch["spectrogram"] = pad_sequence(result_batch["spectrogram"], batch_first=True).mT
    # result_batch["text_encoded"] = pad_sequence(result_batch["text_encoded"], batch_first=True)
    # result_batch["spectrogram_length"] = torch.tensor(result_batch["spectrogram_length"], dtype=torch.long)
    # result_batch["text_encoded_length"] = torch.tensor(result_batch["text_encoded_length"], dtype=torch.long)

    return result_batch
