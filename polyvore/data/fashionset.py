import logging
import os
import pickle

import lmdb
import numpy as np
import pandas as pd
import six
import torch
import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import utils

from .transforms import get_img_trans

LOGGER = logging.getLogger(__name__)


def open_lmdb(path):
    return lmdb.open(
        path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False
    )


def load_semantic_data(semantic_fn):
    """Load semantic data."""
    data_fn = os.path.join(semantic_fn)
    with open(data_fn, "rb") as f:
        s2v = pickle.load(f)
    return s2v


def read_image_list(image_list_fn):
    """Return image list for each fashion category.

    the image name of n-th item in c-th category in image_list[c][n]
    """
    files = image_list_fn
    return [open(fn).read().splitlines() for fn in files]


def read_item_list(posi_tpl, variable_length):
    """Read items used in outfit list.

    For variable length outfits, the 2ed item list contains "-1"s to maintain the
    ratio of outfit with differrnt length.
    """
    outfits = posi_tpl[:, 1:]
    if variable_length:
        # all items
        main_top = list(set(outfits[:, 0]))
        sub_top = set(outfits[:, 1])
        # remove index -1
        sub_top.discard(-1)
        sub_top = list(sub_top)
        bot = list(set(outfits[:, 2]))
        sho = list(set(outfits[:, 3]))
        # portion of outfits that have one top item
        p = sum(outfits[:, 1] == -1) / len(outfits)
        sub_top = sub_top + [-1] * int(len(sub_top) * p / (1 - p))
        item_list = [main_top, sub_top, bot, sho]
        return item_list
    # for fixed length outfit, just return the list
    item_sets = [set(items) for items in outfits.transpose()]
    return [list(items) for items in item_sets]


class Datum(object):
    """Abstract Class for Polyvore dataset."""

    def __init__(
        self,
        image_list,  # item list for each category
        variable_length=False,
        use_semantic=False,
        semantic=None,
        use_visual=False,
        image_dir="",
        lmdb_env=None,
        transforms=None,
    ):

        if variable_length:
            # regard the variable-length outfits as four-category
            self.cate_map = [0, 0, 1, 2]
            self.cate_name = ["top", "top", "bottom", "shoe"]
        else:
            # the normal outfits
            self.cate_map = [0, 1, 2]
            self.cate_name = ["top", "bottom", "shoe"]
        self.image_list = image_list
        self.use_semantic = use_semantic
        self.semantic = semantic
        self.use_visual = use_visual
        self.image_dir = image_dir
        self.lmdb_env = lmdb_env
        self.transforms = transforms

    def load_image(self, c, n):
        """PIL loader for loading image.

        Return
        ------
        img: the image of n-th item in c-the category, type of PIL.Image.
        """
        img_name = self.image_list[c][n]
        # read with lmdb format
        if self.lmdb_env:
            with self.lmdb_env.begin(write=False) as txn:
                imgbuf = txn.get(img_name.encode())
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        else:
            # read from raw image
            path = os.path.join(self.image_dir, img_name)
            with open(path, "rb") as f:
                img = Image.open(f).convert("RGB")
        return img

    def load_semantics(self, c, n):
        """Load semantic embedding.

        Return
        ------
        vec: the semantic vector of n-th item in c-the category,
            type of torch.FloatTensor.
        """
        img_name = self.image_list[c][n]
        vec = self.semantic[img_name]
        return torch.from_numpy(vec.astype(np.float32))

    def semantic_data(self, indices):
        """Load semantic data of one outfit."""
        vecs = []
        # for simplicity, fill up top item for variable-length outfit
        if indices[1] == -1:
            indices[1] = indices[0]
        for idx, cate in zip(indices, self.cate_map):
            v = self.load_semantics(cate, idx)
            vecs.append(v)
        return vecs

    def visual_data(self, indices):
        """Load image data of ne outfit."""
        images = []
        # for simplicity, fill up top item for variable-length outfit
        if indices[1] == -1:
            indices[1] = indices[0]
        for idx, cate in zip(indices, self.cate_map):
            img = self.load_image(cate, idx)
            if self.transforms:
                img = self.transforms(img)
            images.append(img)
        return images

    def get(self, tpl):
        """Convert a tuple to torch.FloatTensor."""
        if self.use_semantic and self.use_visual:
            tpl_s = self.semantic_data(tpl)
            tpl_v = self.visual_data(tpl)
            return tpl_v, tpl_s
        if self.use_visual:
            return self.visual_data(tpl)
        if self.use_semantic:
            return self.semantic_data(tpl)
        return tpl


class PolyvoreDataset(Dataset):
    def __init__(self, param, transforms=None):
        self.param = param
        image_list = read_image_list(param.image_list_fn)
        if param.use_semantic:
            semantic = load_semantic_data(param.semantic_fn)
        else:
            semantic = None
        lmdb_env = open_lmdb(param.lmdb_dir) if param.use_lmdb else None
        self.datum = Datum(
            image_list,
            variable_length=param.variable_length,
            use_semantic=param.use_semantic,
            semantic=semantic,
            use_visual=param.use_visual,
            image_dir=param.image_dir,
            lmdb_env=lmdb_env,
            transforms=transforms,
        )
        self.image_list = image_list
        # load tuples
        self.posi_tpl = np.array(pd.read_csv(param.posi_fn, dtype=np.int))
        self.nega_tpl = np.array(pd.read_csv(param.nega_fn, dtype=np.int))
        # load item list
        self.item_list = read_item_list(self.posi_tpl, param.variable_length)
        self.item_size = [len(ilist) for ilist in self.item_list]
        # probability for hard negative samples
        self.hard_ratio = 0.8
        # the ratio between negative outfits and positive outfits
        self.ratio = self.ratio_fix = len(self.nega_tpl) // len(self.posi_tpl)
        # split user ids and positive tuple ids
        self.uidxs = self.posi_tpl[:, 0]
        self.posi = self.posi_tpl[:, 1:]
        # negative tuple ids
        self.nega = self.nega_fix = self.nega_tpl[:, 1:]
        self.num_posi = len(self.posi)
        self.num_users = len(set(self.uidxs))
        # id list and size for each category
        self.posi_set = set(map(tuple, self.posi))
        # number of positive outfits for each user
        self.num_posi_per_user = np.bincount(self.uidxs)
        self.set_data_mode(param.data_mode)
        self.set_nega_mode(param.nega_mode)
        self.summary()

    def get_image_path(self, c, n):
        image_name = self.image_list[c][n]
        return os.path.join(self.param.image_dir, image_name)

    def get_names(self, tpl):
        # duplicate the first top if the second is empty
        if tpl[1] == -1:
            tpl[1] = tpl[0]
        names = []
        for cate, idx in zip(self.param.cate_map, tpl):
            names.append(self.image_list[cate][idx])
        return names

    def get_outfits_list(self):
        """Return the outfits list (both positive and negative for each user)."""
        tpls = [[] for u in range(self.num_users)]
        for u, tpl in zip(self.uidxs, self.posi):
            tpls[u].append(tpl)
        for i, tpl in enumerate(self.nega):
            u = self.uidxs[i // self.ratio]
            tpls[u].append(tpl)
        return tpls

    def save_tuples(self, posi_fn, nega_fn):
        """Save current tuple list to file."""
        cols = ["user"] + self.param.cate_name
        # positive outfit tuples
        uidxs = self.uidxs.reshape(-1, 1)
        posi = pd.DataFrame(np.hstack((uidxs, self.posi)), columns=cols)
        posi.to_csv(posi_fn, index=False)
        # negative outfit tuples
        uidxs = np.repeat(self.uidxs, self.ratio).reshape(-1, 1)
        nega = pd.DataFrame(np.hstack((uidxs, self.nega)), columns=cols)
        nega.to_csv(nega_fn, index=False)

    def summary(self):
        LOGGER.info("Summary for %s data set", utils.colour(self.param.phase))
        LOGGER.info("Number of users: %s", utils.colour(self.num_users))
        LOGGER.info(
            "Number of items: %s", utils.colour(",".join(map(str, self.item_size)))
        )
        LOGGER.info("Number of positive outfits: %s", utils.colour(self.num_posi))

    def set_nega_mode(self, mode):
        """Set negative outfits mode."""
        assert mode in [
            "RandomOnline",
            "RandomFix",
            "HardOnline",
            "HardFix",  # not implemented
        ], "Unknown negative mode."
        if self.param.data_mode == "PosiOnly":
            LOGGER.warning(
                "Current data-mode is %s. " "The negative mode will be ignored!",
                utils.colour(self.param.data_mode, "Red"),
            )
        else:
            LOGGER.info("Set negative mode to %s.", utils.colour(mode))
            self.param.nega_mode = mode
            self.make_nega()

    def set_data_mode(self, mode):
        """Set data mode."""
        assert mode in [
            "TupleOnly",
            "PosiOnly",
            "NegaOnly",
            "PairWise",
            "TripleWise",  # not implemented
        ], ("Unknown data mode: %s" % mode)
        LOGGER.info("Set data mode to %s.", utils.colour(mode))
        self.param.data_mode = mode

    # TODO: change name
    def set_prob_hard(self, p):
        """Set the proportion for hard negative examples."""
        if self.param.data_mode == "PosiOnly":
            LOGGER.warning(
                "Current data-mode is %s. " "The proportion will be ignored!",
                utils.colour(self.param.data_mode, "Red"),
            )
        elif self.param.nega_mode != "HardOnline":
            LOGGER.warning(
                "Current negative-sample mode is %s. "
                "The proportion will be ignored!",
                utils.colour(self.param.nega_mode, "Red"),
            )
        else:
            self.phard = p
            LOGGER.info(
                "Set the proportion of hard negative outfits to %s",
                utils.colour("%.3f" % p),
            )

    def _random_nega(self, ratio=1):
        """Make negative outfits randomly (RandomOnline).

        Parameter
        ---------
        ratio: ratio of negative tuples vs positive tuples

        Return
        ------
        nega: Randomly mixed item from each category.
        """
        row, col = self.posi.shape
        nrow = row * ratio
        drow = 2 * nrow
        nega = np.empty((drow, col), dtype=np.int)
        for i in range(col):
            nega[:, i] = np.random.choice(self.item_list[i], drow)
        idx = []
        for i, tpl in enumerate(nega):
            if tuple(tpl) in self.posi_set:
                continue
            else:
                idx.append(i)
            if len(idx) == nrow:
                break
        return nega.take(idx, axis=0)

    def _hard_nega(self, ratio=1):
        """Make hard negative outfits (HardOnline)."""
        idxs = []
        num_nega_per_user = self.num_posi_per_user * ratio
        # randomly select positive outfits for each user
        random_idx_per_user = [
            np.random.randint(self.num_posi, size=num * 3) for num in num_nega_per_user
        ]
        for uid in range(self.num_users):
            num_nega = num_nega_per_user[uid]
            random_idx = random_idx_per_user[uid]
            # valid index for current user
            valid = (self.uidxs[random_idx] != uid).nonzero()
            random_idx = random_idx[valid]
            # if the number does not meet the requirement
            while len(random_idx) < num_nega:
                random_idx = np.random.randint(self.num_posi, size=num_nega * 5)
                valid = (self.uidxs[random_idx] != uid).nonzero()
                random_idx = random_idx[valid]
            idxs.append(random_idx[:num_nega])
        idxs = np.hstack(idxs)
        return self.posi.take(idxs, axis=0)

    def make_nega(self, ratio=1):
        """Make negative outfits according to its mode and ratio."""
        LOGGER.info("Make negative outfit for mode %s ", self.param.nega_mode)
        if self.param.nega_mode == "RandomOnline":
            self.nega = self._random_nega(ratio)
            self.ratio = ratio
            LOGGER.info("Random negative outfits ratio: %d", self.ratio)
        if self.param.nega_mode == "HardOnline":
            rand_nega = self._random_nega(ratio)
            hard_nega = self._hard_nega(ratio)
            assert rand_nega.shape == hard_nega.shape
            nega = np.vstack((rand_nega, hard_nega))
            size = self.uidxs.size * ratio
            rand_idx = np.arange(size)
            hard_idx = np.arange(size) + size
            indicator = np.random.uniform(size=size) < self.hard_ratio
            idx = np.where(indicator, hard_idx, rand_idx)
            phard = indicator.sum() / indicator.size
            idx = np.arange(size) + indicator * size
            self.nega = nega.take(idx, axis=0)
            assert len(self.nega) == (len(self.posi) * ratio)
            self.ratio = ratio
            LOGGER.info("Hard negative portion %.2f", phard)
        if self.param.nega_mode.endswith("Fix"):
            # use negative outfits in file
            self.nega = self.nega_fix
            self.ratio = self.ratio_fix
            LOGGER.info("Fix negative outfits ratio: %d", self.ratio)
        LOGGER.info("Done making negative outfits!")

    # TODO: delete method
    def tpl_to_tensor(self, tpl):
        """Public the API for get outfit by index tuple."""
        return self.datum.get(tpl)

    def _TupleOnly(self, index):
        """Get single outfit."""
        tpl = self.posi[index]
        return (self.datum.get(tpl), tpl)

    def _PosiOnly(self, index):
        """Get single outfit."""
        uidx, tpl = self.uidxs[index], self.posi[index]
        return (self.datum.get(tpl), uidx)

    def _NegaOnly(self, index):
        uidx, tpl = self.uidxs[index // self.ratio], self.nega[index]
        return (self.datum.get(tpl), uidx)

    def _PairWise(self, index):
        """Get a pair of outfits."""
        uidx = self.uidxs[index // self.ratio]
        posi_tpl = self.posi[index // self.ratio]
        nega_tpl = self.nega[index]
        return (self.datum.get(posi_tpl), self.datum.get(nega_tpl), uidx)

    def __getitem__(self, index):
        """Get one tuple of examples by index."""
        return dict(
            TupleOnly=self._TupleOnly,
            PairWise=self._PairWise,
            PosiOnly=self._PosiOnly,
            NegaOnly=self._NegaOnly,
        )[self.param.data_mode](index)

    def __len__(self):
        """Return the size of dataset."""
        return dict(
            TupleOnly=self.num_posi,
            PosiOnly=self.num_posi,  # all positive tuples
            NegaOnly=self.ratio * self.num_posi,  # all negative tuples
            PairWise=self.ratio * self.num_posi,
        )[self.param.data_mode]


# TODO: Check PolyvoreLoader
class PolyvoreLoader(object):
    """Class for Polyvore data loader."""

    def __init__(self, param):
        """Initialize a loader for Polyvore."""
        LOGGER.info(
            "Loading data (%s) in phase (%s)",
            utils.colour(param.data_set),
            utils.colour(param.phase),
        )
        LOGGER.info(
            "Data loader configuration: batch size (%s) " "number of workers (%s)",
            utils.colour(param.batch_size),
            utils.colour(param.num_workers),
        )
        transforms = get_img_trans(param.phase, param.image_size)
        self.dataset = PolyvoreDataset(param, transforms)
        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=param.batch_size,
            num_workers=param.num_workers,
            shuffle=param.shuffle,
            pin_memory=True,
        )
        self.num_users = self.dataset.num_users

    def __len__(self):
        """Return number of batches."""
        return len(self.loader)

    @property
    def num_batch(self):
        """Get number of batches."""
        return len(self.loader)

    @property
    def num_sample(self):
        """Get number of samples."""
        return len(self.dataset)

    # TODO: change API to make
    def make_nega(self, ratio=1):
        """Prepare negative outfits."""
        self.dataset.make_nega(ratio)
        return self

    def set_nega_mode(self, mode):
        """Set the mode for generating negative outfits."""
        self.dataset.set_nega_mode(mode)
        return self

    def set_data_mode(self, mode):
        """Set the mode for data set."""
        self.dataset.set_data_mode(mode)
        return self

    def set_prob_hard(self, p):
        """Set the probability of negative outfits."""
        self.dataset.set_prob_hard(p)
        return self

    def __iter__(self):
        """Return generator."""
        for data in self.loader:
            yield data


# TODO: Check FITBDataset
class FITBDataset(Dataset):
    """Dataset for FITB task.

    Only test data has fitb file.
    """

    def __init__(self, param, transforms=None):
        self.param = param
        image_list = read_image_list(param.image_list_fn)
        if param.use_semantic:
            semantic = load_semantic_data(param.semantic_fn)
        else:
            semantic = None
        lmdb_env = open_lmdb(param.lmdb_dir) if param.use_lmdb else None
        self.datum = Datum(
            image_list,
            variable_length=param.variable_length,
            use_semantic=param.use_semantic,
            semantic=semantic,
            use_visual=param.use_visual,
            image_dir=param.image_dir,
            lmdb_env=lmdb_env,
            transforms=transforms,
        )
        self.image_list = image_list
        self.fitb = np.array(pd.read_csv(param.fitb_fn, dtype=np.int))
        self.num_comparisons = len(self.fitb)
        self.summary()

    def make_fill_in_blank(self, save_fn=None, num_cand=4, fix_cate=None):
        """Make fill-in-the-blank list.

        Parameters
        ----------
        num_cand: number of candidates.
        fix_cate: if given, then only generate list for this category.
        save_fn: save the fill-in-the-blank list.
        """
        if save_fn is None:
            save_fn = self.param.fitb_fn
        if os.path.isfile(save_fn):
            LOGGER.warning(
                "Old fill-in-the-blank file (%s) exists for %s. "
                "Delete it before override.",
                save_fn,
                self.param.phase,
            )
            return None
        outfits = np.array(pd.read_csv(self.param.posi_fn, dtype=np.int))
        item_list = read_item_list(outfits[:, 0], self.param.variable_length)
        cand_tpl = [outfits.copy() for __ in range(num_cand)]
        num_posi = len(outfits)
        num_cate = len(self.param.cate_map)
        # randomly select category for FITB task
        if fix_cate:
            assert fix_cate in range(num_cate)
            cand_cate = np.array([fix_cate] * num_posi)
        else:
            if self.param.variable_length:
                cand_cate = np.random.choice([2, 3], size=num_posi)
            else:
                cand_cate = np.random.choice(num_cate, size=num_posi)
        # randomly select items
        num_item = num_cand * 2
        cand_item = []
        for cate in range(num_cate):
            cand_item.append(np.random.choice(item_list[cate], num_posi * num_item))
        # replace the items in outfits
        for n in tqdm.tqdm(range(num_posi)):
            cate = cand_cate[n]
            cands = cand_item[cate][num_item * n : num_item * (n + 1)]
            # remove the ground-truth item
            cands = list(set(cands) - {outfits[n][cate + 1]})
            # replace
            for i in range(1, num_cand):
                cand_tpl[i][n][cate] = cands[i]
        cand_tpl = np.concatenate(cand_tpl, axis=1)
        cols = (["user"] + self.param.cate_name) * num_cand
        df = pd.DataFrame(cand_tpl, columns=cols)
        df.to_csv(save_fn, index=False)
        return cand_tpl

    def summary(self):
        LOGGER.info("Summary for fill-in-the-blank data set")
        LOGGER.info("Number of outfits: %s", utils.colour(len(self.fitb)))
        LOGGER.info(
            "Number of candidates (include ground-truth): %s",
            utils.colour(self.param.num_cand),
        )

    def __getitem__(self, index):
        """Get one tuple of examples by index."""
        n = index // self.param.num_cand
        i = index % self.param.num_cand
        num = len(self.param.cate_name) + 1
        tpls = self.fitb[n]
        tpl = tpls[num * i + 1 : num * (i + 1)]
        uidx = tpls[num * i]
        return self.datum.get(tpl), uidx

    def __len__(self):
        """Return the size of dataset."""
        return len(self.fitb) * self.param.num_cand


# TODO: Check FITBLoader
class FITBLoader(object):
    """DataLoader warper FITB data."""

    def __init__(self, param):
        """Initialize a loader for FITBDataset."""
        LOGGER = logging.getLogger(__name__)
        LOGGER.info(
            "Loading data (%s) in phase (%s)",
            utils.colour(param.data_set),
            utils.colour(param.phase),
        )
        LOGGER.info(
            "Data loader configuration: batch size (%s) " "number of workers (%s)",
            utils.colour(param.num_cand),
            utils.colour(param.num_workers),
        )
        transforms = get_img_trans(param.phase, param.image_size)
        self.dataset = FITBDataset(param, transforms)
        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=param.num_cand,
            num_workers=param.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def __len__(self):
        """Return number of batches."""
        return len(self.loader)

    @property
    def num_batch(self):
        """Get number of batches."""
        return len(self.loader)

    @property
    def num_sample(self):
        """Get number of samples."""
        return len(self.dataset)

    def __iter__(self):
        """Return generator."""
        for data in self.loader:
            yield data


# --------------------------
# Loader and Dataset Factory
# --------------------------

# TODO: Check
def get_dataset(param):
    name = param.__class__.__name__
    if name == "FITBDataParam":
        return FITBDataset(param)
    if name == "DataParam":
        return PolyvoreDataset(param)
    return None


# TODO: Check
def get_dataloader(param):
    name = param.__class__.__name__
    if name == "FITBDataParam":
        return FITBLoader(param)
    if name == "DataParam":
        return PolyvoreLoader(param)
    return None
