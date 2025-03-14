import os
import random
import time
import datetime
import numpy as np
import albumentations as A
import torch
from adet.config import get_cfg
from detectron2.utils.logger import setup_logger

from detectron2.utils import comm

from detectron2.engine import default_setup, default_argument_parser
from torch.utils.data import DataLoader

from condseg_stage1 import (
    print_and_save,
    shuffling,
    add_condseg_config,
    epoch_time,
    ConDSegStage1,
    DiceBCELoss,
    load_data,
    train,
    evaluate,
    DATASET,
    MULTI_IMAGE_DATASET
)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def my_seeding(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_condseg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet", color=False)

    return cfg


if __name__ == "__main__":

    args = default_argument_parser().parse_args()

    cfg = setup(args)

    dataset_name = 'ssmg_coco_train'

    val_name = None

    seed = random.randint(0, 10000)

    my_seeding(seed)

    image_size = 512
    size = (image_size, image_size)
    batch_size = 1
    num_epochs = 300
    lr = 1e-5
    early_stopping_patience = 150

    resume_path = None

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"stage1_{dataset_name}_{val_name}_lr{lr}_{current_time}"

    base_dir = "output"
    data_path = "datasets"
    save_dir = os.path.join(base_dir, folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_log_path = os.path.join(save_dir, "train_log.txt")
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")

    train_log = open(train_log_path, "w")
    train_log.write("\n")
    train_log.close()

    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    hyperparameters_str = f"Image Size: {image_size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    hyperparameters_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    hyperparameters_str += f"Seed: {seed}\n"
    print_and_save(train_log_path, hyperparameters_str)

    """ Data augmentation: Transforms """

    transform = A.Compose([
        A.Rotate(limit=90, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ], additional_targets={'image2': 'image'})

    """ Dataset """
    # (train_x, train_y), (valid_x, valid_y) = load_data(data_path,val_name)
    # train_x, train_y = shuffling(train_x, train_y)
    # data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    # print_and_save(train_log_path, data_str)
    train_image_dir = "datasets/mul_ssmg_coco/train2017"
    train_json_path = "datasets/mul_ssmg_coco/annotations/instances_train2017.json"

    val_image_dir = "datasets/mul_ssmg_coco/val2017"
    val_json_path = "datasets/mul_ssmg_coco/annotations/instances_val2017.json"

    """ Dataset and loader """
    train_dataset = MULTI_IMAGE_DATASET(train_image_dir, train_json_path, (image_size, image_size), transform=transform)
    valid_dataset = MULTI_IMAGE_DATASET(val_image_dir, val_json_path, (image_size, image_size), transform=None)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    """ Model """
    device = torch.device('cuda')
    model = ConDSegStage1(cfg)

    if resume_path:
        checkpoint = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(checkpoint)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30, verbose=True)
    loss_fn = DiceBCELoss()
    loss_name = "BCE Dice Loss"
    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    data_str = f"Number of parameters: {num_params / 1000000}M\n"
    print_and_save(train_log_path, data_str)

    """ Training the model """
    best_valid_metrics = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        if valid_metrics[0] > best_valid_metrics:
            data_str = f"Valid mIoU improved from {best_valid_metrics:2.4f} to {valid_metrics[0]:2.4f}. Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[0]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0


        elif valid_metrics[0] < best_valid_metrics:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f} - mIoU: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - mIoU: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        print_and_save(train_log_path, data_str)

        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving from last {early_stopping_patience} continously.\n"
            print_and_save(train_log_path, data_str)
            break
