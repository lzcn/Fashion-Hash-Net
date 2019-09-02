#!/usr/bin/env python
"""Script for training, evaluate and retrieval."""
import argparse
import logging
import os
import pickle
import shutil
import textwrap

import numpy as np
import torch
import tqdm
import yaml
from torch.nn.parallel import data_parallel

import polyvore
import utils

NAMED_SOLVERS = utils.get_named_class(polyvore.solver)
NAMED_MODELS = utils.get_named_class(polyvore.model)


def get_net(config):
    """Get network."""
    # get net param
    net_param = config.net_param
    LOGGER.info("Initializing %s", config.net_param.name)
    LOGGER.info(net_param)
    # dimension of latent codes
    net = NAMED_MODELS[config.net_param.name](net_param)
    # Load model from pre-trained file
    if config.load_trained:
        # load weights from pre-trained model
        num_devices = torch.cuda.device_count()
        map_location = {"cuda:{}".format(i): "cpu" for i in range(num_devices)}
        LOGGER.info("Loading pre-trained model from %s.", config.load_trained)
        state_dict = torch.load(config.load_trained, map_location=map_location)
        # when new user problem from pre-trained model
        if config.cold_start:
            # TODO: fit with new arch
            # reset the user's embedding
            LOGGER.info("Reset the user embedding")
            # TODO: use more decent way to load pre-trained model for new user
            weight = "user_embedding.encoder.weight"
            state_dict[weight] = torch.zeros(net_param.dim, net_param.num_users)
            net.load_state_dict(state_dict)
            net.user_embedding.init_weights()
        else:
            # load pre-trained model
            net.load_state_dict(state_dict)
    elif config.resume:  # resume training
        LOGGER.info("Training resume from %s.", config.resume)
    else:
        LOGGER.info("Loading weights from backbone %s.", net_param.backbone)
        net.init_weights()
    LOGGER.info("Copying net to GPU-%d", config.gpus[0])
    net.cuda(device=config.gpus[0])
    return net


def update_npz(fn, results):
    if fn is None:
        return
    if os.path.exists(fn):
        pre_results = dict(np.load(fn, allow_pickle=True))
        pre_results.update(results)
        results = pre_results
    np.savez(fn, **results)


def evalute_accuracy(config):
    """Evaluate fashion net for accuracy."""
    # make data loader
    parallel, device = utils.get_device(config.gpus)
    param = config.data_param
    loader = polyvore.data.get_dataloader(param)
    net = get_net(config)
    net.eval()
    # set data mode to pair for testing pair-wise accuracy
    LOGGER.info("Testing for accuracy")
    loader.set_data_mode("PairWise")
    loader.make_nega()
    accuracy = binary = 0.0
    for idx, inputv in enumerate(loader):
        # compute output and loss
        uidx = inputv[-1].view(-1)
        batch_size = uidx.numel()
        inputv = utils.to_device(inputv, device)
        with torch.no_grad():
            if parallel:
                output = data_parallel(net, inputv, config.gpus)
            else:
                output = net(*inputv)
        _, batch_results = net.gather(output)
        batch_accuracy = batch_results["accuracy"]
        batch_binary = batch_results["binary_accuracy"]
        LOGGER.info(
            "Batch [%d]/[%d] Accuracy %.3f Accuracy (Binary Codes) %.3f",
            idx,
            loader.num_batch,
            batch_accuracy,
            batch_binary,
        )
        accuracy += batch_accuracy * batch_size
        binary += batch_binary * batch_size
    accuracy /= loader.num_sample
    binary /= loader.num_sample
    LOGGER.info("Average accuracy: %.3f, Binary Accuracy: %.3f", accuracy, binary)
    # save results
    if net.param.zero_iterm:
        results = dict(uaccuracy=accuracy, ubinary=binary)
    elif net.param.zero_uterm:
        results = dict(iaccuracy=accuracy, ibinary=binary)
    else:
        results = dict(accuracy=accuracy, binary=binary)
    update_npz(config.result_file, results)


def evalute_rank(config):
    """Evaluate fashion net for NDCG an AUC."""

    def outfit_scores():
        """Compute rank scores for data set."""
        num_users = net.param.num_users
        scores = [[] for u in range(num_users)]
        binary = [[] for u in range(num_users)]
        for inputv in tqdm.tqdm(loader, desc="Computing scores"):
            uidx = inputv[-1].view(-1)
            inputv = utils.to_device(inputv, device)
            with torch.no_grad():
                if parallel:
                    output = data_parallel(net, inputv, config.gpus)
                else:
                    output = net(*inputv)
            # save scores for each user
            for n, u in enumerate(uidx):
                scores[u].append(output[0][n].item())
                binary[u].append(output[1][n].item())
        return scores, binary

    parallel, device = utils.get_device(config.gpus)
    LOGGER.info("Testing for NDCG and AUC.")
    print(config.net_param)
    net = get_net(config)
    net.eval()
    data_param = config.data_param
    data_param.shuffle = False
    LOGGER.info("Dataset for positive tuples: %s", data_param)
    loader = polyvore.data.get_dataloader(data_param)
    loader.make_nega()
    loader.set_data_mode("PosiOnly")
    posi_score, posi_binary = outfit_scores()
    LOGGER.info("Compute scores for positive outfits, done!")
    loader.set_data_mode("NegaOnly")
    nega_score, nega_binary = outfit_scores()
    LOGGER.info("Compute scores for negative outfits, done!")
    # compute ndcg
    mean_ndcg, avg_ndcg = utils.metrics.NDCG(posi_score, nega_score)
    mean_ndcg_binary, avg_ndcg_binary = utils.metrics.NDCG(posi_binary, nega_binary)
    aucs, mean_auc = utils.metrics.ROC(posi_score, nega_score)
    aucs_binary, mean_auc_binary = utils.metrics.ROC(posi_binary, nega_binary)
    LOGGER.info(
        "Metric:\n"
        "- average ndcg:%.4f\n"
        "- average ndcg(binary):%.4f\n"
        "- mean auc:%.4f\n"
        "- mean auc(binary):%.4f",
        mean_ndcg.mean(),
        mean_ndcg_binary.mean(),
        mean_auc,
        mean_auc_binary,
    )
    # save results
    results = dict(
        posi_score_binary=posi_binary,
        posi_score=posi_score,
        nega_score_binary=nega_binary,
        nega_score=nega_score,
        mean_ndcg=mean_ndcg,
        avg_ndcg=avg_ndcg,
        mean_ndcg_binary=mean_ndcg_binary,
        avg_ndcg_binary=avg_ndcg_binary,
        aucs=aucs,
        mean_auc=mean_auc,
        aucs_binary=aucs_binary,
        mean_auc_binary=mean_auc_binary,
    )
    update_npz(config.result_file, results)
    # saved ranked outfits
    result_dir = config.result_dir
    if config.result_dir is None:
        return
    assert not data_param.variable_length
    labels = [
        np.array([1] * len(pos) + [0] * len(neg))
        for pos, neg in zip(posi_score, nega_score)
    ]
    outfits = loader.dataset.get_outfits_list()
    sorting = [
        np.argsort(-1.0 * np.array(pos + neg))
        for pos, neg in zip(posi_binary, nega_binary)
    ]
    utils.check.check_dirs(result_dir, action="mkdir")
    ndcg_fn = os.path.join(result_dir, "ndcg.txt")
    label_folder = os.path.join(result_dir, "label")
    outfit_folder = os.path.join(result_dir, "outfit")
    utils.check.check_dirs([label_folder, outfit_folder], action="mkdir")
    np.savetxt(ndcg_fn, mean_ndcg_binary)
    for uid, ranked_idx in tqdm.tqdm(enumerate(sorting), desc="Computing outfits"):
        # u is the user id, rank is the sorting for outfits
        folder = os.path.join(outfit_folder, "user-%03d" % uid)
        utils.check.check_dirs(folder, action="mkdir")
        label_file = os.path.join(label_folder, "user-%03d.txt" % uid)
        # save the rank list for current user
        np.savetxt(label_file, labels[uid][ranked_idx], fmt="%d")
        # rank the outfit according to rank scores
        for n, idx in enumerate(ranked_idx):
            # tpl is the n-th ranked outfit
            tpl = outfits[uid][idx]
            y = labels[uid][idx]
            image_folder = os.path.join(folder, "top-%03d-%d" % (n, y))
            utils.check.check_dirs(image_folder, action="mkdir")
            for cate, item_id in enumerate(tpl):
                src = loader.dataset.get_image_path(cate, item_id)
                dst = os.path.join(image_folder, "%02d.jpg" % cate)
                shutil.copy2(src, dst)
    LOGGER.info("All outfits are save in %s", config.result_dir)


# TODO: Check fitb
def fitb(config):
    parallel, device = utils.get_device(config.gpus)
    data_param = config.fitb_data_param
    LOGGER.info("Get data for FITB questions: %s", data_param)
    loader = polyvore.data.get_dataloader(data_param)
    pbar = tqdm.tqdm(loader, desc="Computing scores")
    net = get_net(config)
    net.eval()
    correct = 0
    cnt = 0
    for inputv in pbar:
        inputv = utils.to_device(inputv, device)
        with torch.no_grad():
            if parallel:
                _, score_b = data_parallel(net, inputv, config.gpus)
            else:
                _, score_b = net(*inputv)
        # the first item is the groud-truth item
        if torch.argmax(score_b).item() == 0:
            correct += 1
        cnt += 1
        pbar.set_description("Accuracy: {:.3f}".format(correct / cnt))
    fitb_acc = correct / cnt
    LOGGER.info("FITB Accuracy %.4f", fitb_acc)
    results = dict(fitb_acc=fitb_acc)
    update_npz(config.result_file, results)


def train(config):
    """Training tasks."""
    # get data for training
    train_param = config.train_data_param or config.data_param
    LOGGER.info("Data set for training: %s", train_param)
    train_loader = polyvore.data.get_dataloader(train_param)
    # set data for validation
    val_param = config.test_data_param or config.data_param
    LOGGER.info("Data set for validation: %s", val_param)
    val_loader = polyvore.data.get_dataloader(val_param)
    # check number of users
    assert val_loader.num_users == train_loader.num_users
    # get net
    net = get_net(config)
    # get solver
    solver_param = config.solver_param
    name = config.solver_param.name
    LOGGER.info("Initialize a solver for training.")
    LOGGER.info("Solver configuration: %s", solver_param)
    solver = NAMED_SOLVERS[name](solver_param, net, train_loader, val_loader)
    # load solver state
    if config.resume:
        solver.resume(config.resume)
    # run
    solver.run()


# TODO: check
def extract_features(config):
    LOGGER.info("Extract features.")
    data_param = config.data_param
    LOGGER.info("Dataset for positive tuples: %s", data_param)
    loader = polyvore.data.get_dataloader(data_param)
    net = get_net(config).eval()
    device = config.gpus[0]
    pbar = tqdm.tqdm(loader, desc="Computing features")
    user_codes = net.get_user_binary_code(device)
    item_codes = dict()
    lambda_i, lambda_u, alpha = net.get_matching_weight()
    for inputv in pbar:
        items, tpls = inputv
        items = utils.to_device(items, device)
        with torch.no_grad():
            features = net.compute_codes(items)
        if data_param.use_semantic and data_param.use_visual:
            feat_v, feat_t = features
            feat_v = [feat.cpu().numpy().astype(np.int8) for feat in feat_v]
            feat_t = [feat.cpu().numpy().astype(np.int8) for feat in feat_t]
        elif data_param.use_semantic:
            feat_t = features
            feat_v = [[] for _ in feat_t]
            feat_t = [feat.cpu().numpy().astype(np.int8) for feat in feat_t]
        elif data_param.use_visual:
            feat_v = features
            feat_v = [feat.cpu().numpy().astype(np.int8) for feat in feat_v]
            feat_t = [[] for _ in feat_v]
        else:
            raise ValueError
        for n, tpl in enumerate(tpls):
            names = loader.dataset.get_names(tpl)
            for c, name in enumerate(names):
                item_codes[name] = [feat_v[c][n], feat_t[c][n]]

    with open(config.feature_file, "wb") as f:
        data = dict(
            user_codes=user_codes,
            item_codes=item_codes,
            lambda_u=lambda_u,
            lambda_i=lambda_i,
            alpha=alpha,
        )
        pickle.dump(data, f)


ACTION_FUNS = {
    "train": train,
    "fitb": fitb,
    "evaluate-accuracy": evalute_accuracy,
    "evaluate-rank": evalute_rank,
    "extract-features": extract_features,
}

LOGGER = logging.getLogger("polyvore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Fashion Hash Net",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Fashion Hash Net Training Script
            --------------------------------
            Actions:
                1. train: train fashion net.
                2. evaluate: evaluate NDCG and accuracy.
                3. retrieval: retrieval for items.
                """
        ),
    )
    actions = ACTION_FUNS.keys()
    parser.add_argument("action", help="|".join(sorted(actions)))
    parser.add_argument("--cfg", help="configuration file.")
    args = parser.parse_args()
    with open(args.cfg, "r") as f:
        kwargs = yaml.load(f, Loader=yaml.FullLoader)
    config = polyvore.param.FashionParam(**kwargs)
    # config.add_timestamp()
    logfile = utils.config_log(stream_level=config.log_level, log_file=config.log_file)
    LOGGER.info("Logging to file %s", logfile)
    LOGGER.info("Fashion param : %s", config)

    if args.action in actions:
        ACTION_FUNS[args.action](config)
        exit(0)
    else:
        LOGGER.info("Action %s is not in %s", args.action, "|".join(actions))
        exit(1)
