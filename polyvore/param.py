"""Fashion Net Configuration."""
import logging
import os
import warnings
from datetime import datetime

import polyvore.config as cfg

NO_WEIGHTED_HASH = 0
WEIGHTED_HASH_U = 1
WEIGHTED_HASH_I = 2
WEIGHTED_HASH_BOTH = 3

_VAR_LENGTH_DATA = ["tuples_519", "tuples_32"]
_VAR_LENGTH_USER = [519, 32]

LOGGER = logging.getLogger(__name__)


def show(opt, num=0):
    """Print log header."""
    logger = logging.getLogger(__name__)
    indent = "  " * num
    for k, v in opt.items():
        if v is None:
            continue
        if isinstance(v, dict):
            logger.info("%s%s : {{", indent, k)
            show(v, num + 1)
            logger.info("%s}}", indent)
        else:
            logger.info("%s%s : %s,", indent, k, v)


def format_display(opt, num=1):
    """Show hierarchal information for _Param class."""
    indent = "  " * num
    string = ""
    for k, v in opt.items():
        if v is None:
            continue
        if isinstance(v, _Param):
            string += "{}{} : {{\n".format(indent, k)
            string += format_display(v.get_params(), num + 1)
            string += "{}}},\n".format(indent)
        elif isinstance(v, dict):
            string += "{}{} : {{\n".format(indent, k)
            string += format_display(v, num + 1)
            string += "{}}},\n".format(indent)
        elif isinstance(v, list):
            string += "{}{} : ".format(indent, k)
            one_line = ",".join(map(str, v))
            if len(one_line) < 87:
                string += "[" + one_line + "]\n"
            else:
                prefix = "  " + indent
                string += "[\n"
                for i in v:
                    string += "{}{},\n".format(prefix, i)
                string += "{}]\n".format(indent)
        else:
            string += "{}{} : {},\n".format(indent, k, v)
    return string


class _Param(object):
    """Base `Param` class.

    There are two sets of params:
     - `Configurable params` are listed in self.default with default values
     - `Other params` are listed in self.infer which can be inferred.

    Methods
    -------
    - self.get_params: return two sets of params
    - self.log: show two sets of params

    """

    # default configurations
    default = {}
    # param which can be inferred from default
    infer = []

    def __init__(self, **kwargs):
        """Update default paramaters and set all as attributes."""
        if (kwargs.keys() | self.default.keys()) != self.default.keys():
            err_key = list(kwargs.keys() - self.default.keys())
            all_key = list(self.default.keys())
            raise KeyError(
                "Keyword(s) {} is(are) not in configurable list: {}".format(
                    err_key, all_key
                )
            )
        self.default.update(kwargs)
        for attr, value in self.default.items():
            setattr(self, attr, value)
        self.setup()

    def setup(self):
        """Setup configuration."""
        pass

    def log(self):
        print(self)

    def get_params(self):
        params = dict()
        params.update(self.default)
        for key in self.infer:
            params[key] = self.__getattribute__(key)
        return params

    def __setattr__(self, name, value):
        if name in self.default:
            self.default[name] = value
        return super().__setattr__(name, value)

    def __str__(self):
        type_name = type(self).__name__
        return type_name + ":\n" + format_display(self.get_params(), 1)

    def __repr__(self):
        return "ParamClass(# Configrations: %s)" % len(self.default)


# TODO: Check FITBDataParam
class FITBDataParam(_Param):
    default = dict(
        phase="test",
        data_set="tuples_630",  # Polyvore-U dataset
        data_root="data/polyvore",  # data root
        list_fmt="image_list_{}",
        use_semantic=False,
        use_visual=True,
        image_root=None,  # image root if it's saved in another place
        saliency_image=False,  # whether to use saliency image
        image_size=291,
        use_lmdb=True,  # whether to use lmdb data
        num_workers=8,  # number of workers for dataloader
        num_cand=4,  # number of candidates, which equals to batch size
    )
    infer = [
        "data_dir",
        "image_list_fn",
        "image_dir",
        "posi_fn",
        "fitb_fn",
        "semantic_fn",
    ]

    def setup(self):
        self.image_root = self.image_root or self.data_root
        self.variable_length = self.data_set in _VAR_LENGTH_DATA
        if self.variable_length:
            # regard the variable-length outfits as four-category
            self.cate_map = [0, 0, 1, 2]
            self.cate_name = ["top", "top", "bottom", "shoe"]
        else:
            # the normal outfits
            self.cate_map = [0, 1, 2]
            self.cate_name = ["top", "bottom", "shoe"]
        # tricky implementation for FITB by loading one task in one batch.
        # no shuffle
        self.shuffle = False
        if not (self.use_semantic or self.use_visual):
            warnings.warn("Neither semantic nor visual is selected! ", RuntimeWarning)

    @property
    def semantic_fn(self):
        return os.path.join(self.data_root, "sentence_vector/semantic.pkl")

    @property
    def fitb_fn(self):
        fn = os.path.join(self.data_dir, "fill_in_blank_{}".format(self.phase))
        return fn

    @property
    def image_dir(self):
        if self.saliency_image:
            folder = "saliency"
        else:
            folder = "291x291"
        return os.path.join(self.image_root, "images", folder)

    @property
    def lmdb_dir(self):
        return os.path.join(self.image_root, "images/lmdb")

    @property
    def data_dir(self):
        return os.path.join(self.data_root, self.data_set)

    @property
    def image_list_fn(self):
        return [
            os.path.join(self.data_dir, self.list_fmt.format(p)) for p in cfg.CateName
        ]

    @property
    def posi_fmt(self):
        """Infer the file format for positive tuples"""
        return "tuples_{}_posi"

    @property
    def posi_fn(self):
        return os.path.join(self.data_dir, self.posi_fmt.format(self.phase))


class DataParam(_Param):
    """Parameters class for data."""

    default = dict(
        phase="train",
        data_set="tuples_630",  # Polyvore-U dataset
        data_mode="PairWise",  # mode for data
        nega_mode="RandomOnline",  # mode for negative outfits
        data_root="data/polyvore",  # data root
        list_fmt="image_list_{}",
        use_semantic=False,
        use_visual=True,
        image_root=None,  # image root if it's saved in another place
        saliency_image=False,  # whether to use saliency image
        image_size=291,
        use_lmdb=True,  # whether to use lmdb data
        batch_size=64,
        num_workers=8,  # number of workers for dataloader
        shuffle=None,
        fsl=None,
    )
    infer = [
        "image_dir",
        "data_dir",
        "posi_fn",
        "nega_fn",
        "image_list_fn",
        "semantic_fn",
    ]

    def setup(self):
        """Load args."""
        if self.fsl:
            self.data_set += "_fsl"
        self.image_root = self.image_root or self.data_root
        self.variable_length = self.data_set in _VAR_LENGTH_DATA
        if self.variable_length:
            # regard the variable-length outfits as four-category
            self.cate_map = [0, 0, 1, 2]
            self.cate_name = ["top", "top", "bottom", "shoe"]
        else:
            # the normal outfits
            self.cate_map = [0, 1, 2]
            self.cate_name = ["top", "bottom", "shoe"]
        if self.shuffle is None:
            self.shuffle = self.shuffle or (self.phase == "train")
        if (not self.use_semantic) and (not self.use_visual):
            warnings.warn("Neither semantic nor visual is selected! ", RuntimeWarning)

    @property
    def image_dir(self):
        if self.saliency_image:
            folder = "saliency"
        else:
            folder = "291x291"
        return os.path.join(self.image_root, "images", folder)

    @property
    def lmdb_dir(self):
        return os.path.join(self.image_root, "images/lmdb")

    @property
    def data_dir(self):
        return os.path.join(self.data_root, self.data_set)

    @property
    def semantic_fn(self):
        return os.path.join(self.data_root, "sentence_vector/semantic.pkl")

    @property
    def image_list_fn(self):
        return [
            os.path.join(self.data_dir, self.list_fmt.format(p)) for p in cfg.CateName
        ]

    @property
    def posi_fmt(self):
        """Infer the file format for positive tuples"""
        return "tuples_{}_posi"

    @property
    def nega_fmt(self):
        # only used for evaluation
        if self.nega_mode == "HardFix":
            return "tuples_{}_nega_hard"
        return "tuples_{}_nega"

    @property
    def posi_fn(self):
        return os.path.join(self.data_dir, self.posi_fmt.format(self.phase))

    @property
    def nega_fn(self):
        return os.path.join(self.data_dir, self.nega_fmt.format(self.phase))

    @property
    def hard(self):
        if self.nega_mode in ["HardFix", "HardOnline"]:
            return True
        return False


# TODO: Check NetParam
class NetParam(_Param):
    """Parameters class for net."""

    default = dict(
        name="FashionNet",
        num_users=630,
        dim=128,
        single=False,
        binary01=False,
        triplet=False,
        scale_tanh=True,
        backbone="alexnet",
        share_user=False,
        without_binary=False,
        zero_uterm=False,
        zero_iterm=False,
        use_semantic=False,
        use_visual=False,
        hash_types=0,
        margin=None,
        debug=False,
    )

    def setup(self):
        self.variable_length = self.num_users in _VAR_LENGTH_USER
        if self.use_semantic and self.use_visual:
            self.margin = self.margin or 0.1
        if self.num_users in _VAR_LENGTH_USER:
            # regard the variable-length outfits as four-category
            self.cate_map = [0, 0, 1, 2]
            self.variable_length = True
            self.cate_name = ["top", "top", "bottom", "shoe"]
        else:
            # the normal outfits
            self.cate_map = [0, 1, 2]
            self.variable_length = False
            self.cate_name = ["top", "bottom", "shoe"]


# TODO: Check OptimParam
class OptimParam(_Param):
    default = dict(
        name="SGD",
        lr=1e-3,
        weight_decay=0,
        grad_param=None,
        lr_scheduler="StepLR",
        scheduler_param=None,
    )
    infer = ["groups"]

    def setup(self):
        self.groups = self._param_groups(self.lr, self.weight_decay)
        if self.name == "SGD":
            self.grad_param = self._optim_SGD(self.grad_param)
        elif self.name in ["Adam", "Adamax"]:
            self.grad_param = self._optim_Adam(self.grad_param)
        else:
            raise KeyError
        scheduler_param = self.scheduler_param
        if self.lr_scheduler == "StepLR":
            self.lr_param = self._policy_StepLR(scheduler_param)
        elif self.lr_scheduler == "ReduceLROnPlateau":
            self.lr_param = self._policy_ReduceLROnPlateau(scheduler_param)
        else:
            raise KeyError

    def _param_groups(self, lr, weight_decay):
        """Parse setting for multiple group of parameters."""
        # learning rates and weight decay
        if not isinstance(lr, list):
            lr = [lr]
        if not isinstance(weight_decay, list):
            weight_decay = [weight_decay]
        num_lrs, lrs = len(lr), lr
        num_wds, wds = len(weight_decay), weight_decay
        if num_lrs == 1 and num_wds > 1:
            lrs = lrs * num_wds
        elif num_lrs > 1 and num_wds == 1:
            wds = wds * num_lrs
        elif num_lrs > 1 and num_wds > 1:
            assert num_lrs == num_wds, (
                "Number of learning rate doesn't",
                "the number of weight decay.",
            )
        groups = [dict(lr=lr, weight_decay=wd) for lr, wd in zip(lrs, wds)]
        return groups

    def _optim_SGD(self, param=None):
        if param is None:
            param = dict()
        return dict(momentum=param.get("momentum", 0.9))

    def _optim_Adam(self, param=None):
        if param is None:
            param = dict()
        return dict(betas=param.get("betas", (0.9, 0.999)), eps=param.get("eps", 1e-8))

    def _policy_StepLR(self, param=None):
        if param is None:
            param = dict()
        lr_param = dict(
            step_size=param.get("step_size", 30), gamma=param.get("gamma", 0.1)
        )
        return lr_param

    def _policy_ReduceLROnPlateau(self, param=None):
        if param is None:
            param = dict()
        lr_param = dict(
            mode="max",
            cooldown=param.get("cooldown", 10),
            factor=param.get("factor", 0.1),
            patience=param.get("patience", 10),
            threshold=param.get("threshold", 0.0001),
            verbose=True,
        )
        return lr_param


# TODO: Check SolverParam
class SolverParam(_Param):
    """Parameters class for solver."""

    default = dict(
        name=None,
        gpus=[0],
        gamma=0.1,
        visdom_env="main",
        checkpoints="./checkpoints",
        visdom_title="FHN",
        display=10,
        balance_loss=False,
        increase_hard=False,
        epochs=100,
        optim_param=None,
    )

    def setup(self):
        optim_param = self.optim_param or dict()
        self.optim_param = OptimParam(**optim_param)

    def add_timestamp(self, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now().strftime("%m%d%H%M")
        self.visdom_title = self.visdom_title + "." + timestamp


# TODO: Check FashionParam
class FashionParam(_Param):
    """_Param for fashion hash net."""

    default = dict(
        data_param=None,
        train_data_param=None,
        test_data_param=None,
        fitb_data_param=None,
        net_param=None,
        solver_param=None,
        load_trained=None,  # load pre-trained model
        log_file=None,  # lof file
        log_level=None,  # log level
        result_file=None,  # file to save metric
        result_dir=None,  # folder to save result images
        feature_file=None,  # file for saving features
        resume=None,  # resume training
        cold_start=None,  # fine-tune model for new users
        gpus=None,  # gpus
    )

    def setup(self):
        # if set specific configuration fopr train/test
        if not (self.train_data_param is None and self.test_data_param is None):
            param = self.data_param or dict()
            train_param = self.train_data_param or dict()
            test_param = self.test_data_param or dict()
            train_param.update(param)
            test_param.update(param)
            self.train_data_param = DataParam(**train_param)
            self.test_data_param = DataParam(**test_param)
            self.data_param = None
        if self.data_param:
            param = self.data_param
            self.data_param = DataParam(**param)
        if self.fitb_data_param:
            param = self.fitb_data_param
            self.fitb_data_param = FITBDataParam(**param)

        if self.net_param:
            self.net_param = NetParam(**self.net_param)
        if self.solver_param:
            self.solver_param = SolverParam(**self.solver_param)
            self.gpus = self.solver_param.gpus

    def add_timestamp(self, timestamp=None):
        """Add timestamp to log file and solver."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%m%d%H%M")
        if self.log_file:
            self.log_file = "{name}.{time}.log".format(
                name=self.log_file, time=timestamp
            )
        self.solver_param.add_timestamp(timestamp)


__all__ = [
    "show",
    "FashionParam",
    "NetParam",
    "SolverParam",
    "DataParam",
    "FITBDataParam",
]
