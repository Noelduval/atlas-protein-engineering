"""Candidate-targeted entry point for pinned official ThermoMPNN-D inference.

This module deliberately imports and calls the checkout's model loader,
featurizer, ProteinMPNN encoder, ``SSMDataset``, and ``run_double`` function.
It changes only construction of the mutation tensors so a bounded adaptive run
does not enumerate every amino-acid pair at every nearby residue pair.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
TOKEN = re.compile(r"^(?P<wt>[A-Z])(?P<position>\d+)(?P<mut>[A-Z])$")


def _request_tensors(
    requests: pd.DataFrame, sequence: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    required = {"variant_id", "mutation_set"}
    missing = required.difference(requests.columns)
    if missing:
        raise ValueError(f"Targeted request table is missing columns: {sorted(missing)}")
    positions: list[list[int]] = []
    wildtypes: list[list[int]] = []
    mutants: list[list[int]] = []
    labels: list[str] = []
    for row in requests.itertuples(index=False):
        tokens = str(row.mutation_set).replace(";", ":").split(":")
        if len(tokens) != 2:
            raise ValueError(f"Expected a double mutant, received {row.mutation_set}")
        parsed: list[tuple[int, str, str]] = []
        for token in tokens:
            match = TOKEN.fullmatch(token)
            if match is None:
                raise ValueError(f"Invalid DP622 mutation label: {token}")
            position = int(match.group("position"))
            wt = match.group("wt")
            mut = match.group("mut")
            if not 1 <= position <= len(sequence):
                raise ValueError(f"Mutation position is outside the model sequence: {token}")
            observed = sequence[position - 1]
            if observed != wt:
                raise ValueError(
                    f"Mutation wild type does not match model sequence: {token}; found {observed}"
                )
            if mut == wt or mut not in ALPHABET[:20]:
                raise ValueError(f"Mutation must specify a non-wild-type standard amino acid: {token}")
            parsed.append((position - 1, wt, mut))
        parsed.sort(key=lambda item: item[0])
        if parsed[0][0] == parsed[1][0]:
            raise ValueError(f"Double mutant repeats one position: {row.mutation_set}")
        positions.append([item[0] for item in parsed])
        wildtypes.append([ALPHABET.index(item[1]) for item in parsed])
        mutants.append([ALPHABET.index(item[2]) for item in parsed])
        labels.append(
            ":".join(f"{wt}{position + 1}{mut}" for position, wt, mut in parsed)
        )
    return (
        torch.tensor(positions, dtype=torch.long),
        torch.tensor(wildtypes, dtype=torch.long),
        torch.tensor(mutants, dtype=torch.long),
        labels,
    )


def run(args: argparse.Namespace) -> None:
    checkout = Path(args.repo).resolve()
    sys.path.insert(0, str(checkout))

    # Imported only after checkout precedence is established.  These are all
    # functions/classes from the pinned official ThermoMPNN-D source tree.
    import v2_ssm  # type: ignore[import-not-found]
    from thermompnn.datasets.dataset_utils import Mutation  # type: ignore[import-not-found]
    from thermompnn.datasets.v2_datasets import tied_featurize_mut  # type: ignore[import-not-found]
    from thermompnn.ssm_utils import get_config, get_model, load_pdb  # type: ignore[import-not-found]

    requests = pd.read_csv(args.requests)
    pdb = load_pdb(args.pdb, ["A"])
    positions, wildtypes, mutants, labels = _request_tensors(requests, pdb["seq"])
    cfg = get_config("epistatic")
    model = get_model("epistatic", cfg)
    model.eval()
    model.cuda()
    pdb["mutation"] = Mutation([0], ["A"], ["A"], [0.0], "")

    batch = tied_featurize_mut([pdb])
    (
        X,
        S,
        mask,
        lengths,
        chain_M,
        chain_encoding_all,
        residue_idx,
        _mut_positions,
        _mut_wildtype_aas,
        _mut_mutant_aas,
        mut_ddgs,
        _atom_mask,
    ) = batch
    device = "cuda"
    X = torch.nan_to_num(X.to(device), nan=0.0)
    S = S.to(device)
    mask = mask.to(device)
    _lengths = torch.tensor(lengths).to(device)
    chain_M = chain_M.to(device)
    chain_encoding_all = chain_encoding_all.to(device)
    residue_idx = residue_idx.to(device)
    _mut_ddgs = mut_ddgs.to(device)

    with torch.no_grad():
        all_mpnn_hid, mpnn_embed, _, mpnn_edges = model.prot_mpnn(
            X, S, mask, chain_M, residue_idx, chain_encoding_all
        )
        dataset = v2_ssm.SSMDataset(positions, wildtypes, mutants)
        loader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=0,
        )
        predictions = v2_ssm.run_double(
            all_mpnn_hid,
            mpnn_embed,
            cfg,
            loader,
            args.batch_size,
            model,
            X,
            mask,
            mpnn_edges,
        )

    values = np.atleast_1d(predictions).astype(float)
    if len(values) != len(labels):
        raise RuntimeError(
            f"ThermoMPNN-D returned {len(values)} scores for {len(labels)} requests"
        )
    pd.DataFrame(
        {
            "variant_id": requests["variant_id"].astype(str),
            "Mutation": labels,
            "ddG (kcal/mol)": values,
        }
    ).to_csv(args.out, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
