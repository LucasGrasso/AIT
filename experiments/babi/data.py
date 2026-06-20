import glob
import os
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PAD, UNK = "<pad>", "<unk>"  # ids 0 and 1


def tokenize(text):
    text = text.lower().strip()
    text = re.sub(r"([.?!,])", r" \1 ", text)
    return text.split()


def parse_file(path):
    """Parse one bAbI task file into (story+question tokens, answer, n_support).

    Story id resetting to "1" starts a new story; context accumulates until a
    question line (tab-separated: question, answer, supporting-fact ids).
    """
    examples, story = [], []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            sid_str, rest = line.split(" ", 1)
            if sid_str == "1":
                story = []
            if "\t" in rest:
                q, answer, support = rest.split("\t")
                tokens = [w for sent in story for w in sent] + tokenize(q)
                examples.append((tokens, answer.strip(), len(support.split())))
            else:
                story.append(tokenize(rest))
    return examples


def task_files(data_dir, tasks, split):
    out = []
    for t in tasks:
        hits = sorted(glob.glob(os.path.join(data_dir, f"qa{t}_*_{split}.txt")))
        if not hits:
            raise FileNotFoundError(f"no qa{t}_*_{split}.txt in {data_dir}")
        out.extend(hits)
    return out


def build_vocab(data_dir, tasks):
    """Deterministic vocab over the TRAIN split so the runner and the analysis
    script agree on token ids. id 0=PAD, 1=UNK, then words sorted."""
    words = set()
    for path in task_files(data_dir, tasks, "train"):
        for tokens, answer, _ in parse_file(path):
            words.update(tokens)
            words.add(answer)
    vocab = {PAD: 0, UNK: 1}
    for w in sorted(words):
        vocab[w] = len(vocab)
    return vocab


def encode(examples, vocab, max_len):
    unk = vocab[UNK]
    X, Y, S = [], [], []
    for tokens, answer, n_support in examples:
        ids = [vocab.get(w, unk) for w in tokens][-max_len:]
        ids = ids + [0] * (max_len - len(ids))  # right-pad with PAD=0
        X.append(ids)
        Y.append(vocab.get(answer, unk))
        S.append(n_support)
    return (
        np.asarray(X, dtype=np.int64),
        np.asarray(Y, dtype=np.int64),
        np.asarray(S, dtype=np.int64),
    )


def load_split(data_dir, tasks, vocab, split, max_len):
    examples = []
    for path in task_files(data_dir, tasks, split):
        examples.extend(parse_file(path))
    return encode(examples, vocab, max_len)


def get_loaders(batch_size, data_dir, tasks=(1, 2, 3), max_len=120, seed=0):
    vocab = build_vocab(data_dir, tasks)
    xtr, ytr, _ = load_split(data_dir, tasks, vocab, "train", max_len)
    xte, yte, _ = load_split(data_dir, tasks, vocab, "test", max_len)
    train = TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr))
    test = TensorDataset(torch.from_numpy(xte), torch.from_numpy(yte))
    g = torch.Generator().manual_seed(seed)
    return (
        DataLoader(train, batch_size, shuffle=True, drop_last=True, generator=g),
        DataLoader(test, batch_size, shuffle=False),
        len(vocab),
    )
