#!/usr/bin/env python
"""Prepare Polyvore dataset in LMDB format."""
import argparse
import logging
import os

import lmdb
from tqdm import tqdm

import utils

LOGGER = logging.getLogger(__name__)


def create_lmdb(dataset, image_dir):
    LOGGER.info("Creating LMDB to %s", dataset)
    image_list = utils.check.list_files(image_dir)
    env = lmdb.open(dataset, map_size=2 ** 40)
    # open json file
    with env.begin(write=True) as txn:
        for image_name in tqdm(image_list):
            fn = os.path.join(image_dir, image_name)
            with open(fn, "rb") as f:
                img_data = f.read()
                txn.put(image_name.encode("ascii"), img_data)
    env.close()
    LOGGER.info("Converted Polyvore to LDMB")


if __name__ == "__main__":

    LOGGER.info("Log to file %s", utils.config_log())
    parser = argparse.ArgumentParser(description="Create LMDB")
    parser.add_argument(
        "src",
        default="data/polyvore/images/291x291",
        type=str,
        help="image folder for polyvore dataset",
    )
    parser.add_argument("dst", type=str, help="folder to save lmdb")

    args = parser.parse_args()
    create_lmdb(args.dst, args.src)
    exit(0)
