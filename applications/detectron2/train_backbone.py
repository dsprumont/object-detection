# Common utility
import os
import argparse
import logging
import time
import torch
import torch.nn.functional as F
from collections import OrderedDict

# Detectron2 dependencies
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.modeling import build_model
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage, CommonMetricPrinter, TensorboardXWriter, JSONWriter
from detectron2.evaluation import inference_on_dataset, print_csv_format

# Custom dependencies
from data.build import build_classification_train_loader, build_classification_test_loader
from modeling.featureExtractor import FeatureExtractor # so META_ARCH_REGISTRY triggers
from modeling.simple import add_simple_resnet_config
from evaluation.evaluator import BinaryClassificationEvaluator

# Dataset dependencies
from fal_syn_dataset import get_cls_dataset


logger = logging.getLogger("detectron2")

def do_train(cfg, model, resume=False):
    """
    training loop.
    """

    # Build optimizer and scheduler from configuration and model
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    # Build checkpointers
    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
    start_iter = (checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)
    max_iter = cfg.SOLVER.MAX_ITER
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )
    # Build writers
    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )
    # Build dataloader
    data_loader = build_classification_train_loader(cfg)

    # training loop
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        start = time.perf_counter()
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):

            data_time = time.perf_counter() - start
            iteration = iteration + 1
            storage.step()

            loss_dict = model(data)

            # compute losses
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalar("data_time", data_time)
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            # backward
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            #validation
            if (
                (cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0)
                or (iteration == max_iter)
            ):
                do_test(cfg, model)

            # logging/checkpoint
            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

            #Try to get an accurate measuremetn of time
            start = time.perf_counter()



def do_test(cfg, model):
    """
    run inference on model:
        preds = model(data)

    accumulate results (ErrorRate, Accuracy, Precision, Recall, Specificity)
        metrics = compute_metrics(preds)

    log aggregated results
    """
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_classification_test_loader(cfg, dataset_name)

        evaluator = BinaryClassificationEvaluator(
            dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name))

        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i

    if len(results) == 1:
        results = list(results.values())[0]
    return results



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_simple_resnet_config(cfg)
    cfg.merge_from_file(args.model_config)

    if args.eval_only==True:
        # cfg.OUTPUT_DIR = "./output_eval"
        cfg.MODEL.WEIGHTS = cfg.OUTPUT_DIR+"/model_final.pth"
        # cfg.DATASETS.TEST = ("xris_fal_syn_test",)
    
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def get_parser():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser(description="Detectron2 train script Classification")
    parser.add_argument(
        "--model-config",
        default="configs/base_feature_extractor.yaml",
        metavar="FILE",
        help="path to model configuration file"
    )
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint directory")
    parser.add_argument("--eval-only", action="store_true", help="Run in evaluation mode")
    return parser
    pass


def main(args):
    """
    Setup, init.
    """    
    cfg = setup(args)
    
    # Add the datasets to the DatasetCatalog
    # TODO: pretty much hardcoded for now, generalization of this will be implemented in the future
    path = 'C:/Users/sprum/Workspace/Anaconda3/TFE/defect-detection/datasets/ADRIC-XRIS-FAL-SYN-SIMP'
    for d in ["train", "val", "test"]:
        DatasetCatalog.register("xris_fal_syn_" + d, lambda d=d: get_cls_dataset(os.path.join(path, d)))
        MetadataCatalog.get("xris_fal_syn_" + d).set(thing_classes=["intact", "defective"])


    # Build the backbone model
    #   cfg:            configuration dict
    #   input_shape:    input_channels (if None, it is deduce from cfg.MODEL.PIXEL_MEAN)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    # Evaluate the model
    if args.eval_only:
        
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    # Train de model
    do_train(cfg, model, resume=False)
    return do_test(cfg, model)


if __name__ == '__main__':
    args = get_parser().parse_args()
    print("Command Line Args:", args)
    main(args)