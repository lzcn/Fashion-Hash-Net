"""Check utils mod."""
import logging
import os

LOGGER = logging.getLogger(__name__)


def check_dirs(folders, action="check", mode="all", verbose=False):
    """Check whether all folders exist."""
    flags = []
    if action.lower() not in ["check", "mkdir"]:
        raise ValueError("{} not in ['check', 'mkdir']".format(action.lower()))
    if mode.lower() not in ["all", "any"]:
        raise ValueError("{} not in ['all', 'any']".format(mode.lower()))
    ops = {"any": any, "all": all}
    if verbose:
        LOGGER.info("Checked folder(s):")
    if isinstance(folders, str):
        flags.append(_check_dir(folders, action, verbose))
    else:
        for folder in folders:
            flags.append(_check_dir(folder, action, verbose))

    return ops[mode](flags)


def _check_dir(folder, action="check", verbose=False):
    """Check if directory exists and make it when necessary.

    Parameters
    ----------
    folder: folder to be checked
    action: what should be do if the folder does not exists, if action is
            'mkdir', than the Return will also be True
    verbose: For rich info
    Return
    ------
    exists: whether the folder exists

    """
    exists = os.path.isdir(folder)
    if not exists:
        if action == "mkdir":
            # make directories recursively
            os.makedirs(folder)
            exists = True
            LOGGER.info("folder '%s' has been created.", folder)
        if action == "check" and verbose:
            LOGGER.info("folder '%s' does not exist.", folder)
    else:
        if verbose:
            LOGGER.info("folder '%s' exist.", folder)
    return exists


def check_files(file_list, mode="any", verbose=False):
    """Check whether files exist, optional modes are ['all','any']."""
    n_file = len(file_list)
    opt_modes = ["all", "any"]
    ops = {"any": any, "all": all}
    if mode not in opt_modes:
        LOGGER.info("Wrong choice of mode, optional modes %s", opt_modes)
        return False
    exists = [os.path.isfile(fn) for fn in file_list]
    if verbose:
        LOGGER.info("names\t status")
        info = [file_list[i] + "\t" + str(exists[i]) for i in range(n_file)]
        LOGGER.info("\n".join(info))
    return ops[mode](exists)


def check_exists(lists, mode="any", verbose=False):
    """Check whether file(s)/folder(s) exist(s)."""
    n_file = len(lists)
    opt_modes = ["all", "any"]
    ops = {"any": any, "all": all}
    if mode not in opt_modes:
        LOGGER.info("Wrong choice of mode, optional modes %s", opt_modes)
        return False
    exists = [os.path.exists(fn) for fn in lists]
    if verbose:
        LOGGER.info("filename\t status")
        info = [lists[i] + "\t" + str(exists[i]) for i in range(n_file)]
        LOGGER.info("\n".join(info))
    return ops[mode](exists)


def list_files(folder="./", suffix="", recursive=False):
    """List all files.

    Parameters
    ----------
    suffix: filename must end with suffix if given, it can also be a tuple
    recursive: if recursive, return sub-paths
    """
    files = []
    if recursive:
        for path, _, fls in os.walk(folder):
            files += [os.path.join(path, f) for f in fls if f.endswith(suffix)]
    else:
        files = [f for f in os.listdir(folder) if f.endswith(suffix)]
    return files
