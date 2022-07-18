import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.select_dataset import get_dataset
from .models.get_model import get_model
from .typings.TrainOptions import TrainOptions
from .utils import utils_image as util
from .utils import utils_option as option
from .utils.get_logger import get_logger


def main(json_path='options/train_fbcnn_color.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt',
        type=str,
        default=json_path,
        help='Path to option JSON file.'
    )

    opt: TrainOptions = option.parse(parser.parse_args().opt, is_train=True)

    for key, path in opt['path'].items():
        if 'pretrained' in key:
            continue

        Path(str(path)).mkdir(exist_ok=True)

    # update opt
    init_iter, init_path_G = option.find_last_checkpoint(
        opt['path']['models'],
        net_type='G'
    )

    opt['path']['pretrained_netG'] = init_path_G
    current_step = init_iter
    border = 0

    option.save(opt)

    # * Logger
    logger_name = 'train'
    logger = get_logger(logger_name)

    # * Seed
    seed = opt['train'].get(
        'manual_seed',
        random.randint(1, 10000)
    )

    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # * Create loaders
    dataset_type = opt['datasets']['train']['dataset_type']
    train_loader = None
    test_loader = None

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = get_dataset(dataset_opt)
            train_size = int(
                math.ceil(len(train_set) / dataset_opt['dataloader_batch_size'])
            )

            logger.info(
                'Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set),
                    train_size
                )
            )

            train_loader = DataLoader(
                train_set,
                batch_size=dataset_opt['dataloader_batch_size'],
                shuffle=dataset_opt['dataloader_shuffle'],
                num_workers=dataset_opt['dataloader_num_workers'],
                drop_last=True,
                pin_memory=True
            )

        elif phase == 'test':
            test_set = get_dataset(dataset_opt)
            test_loader = DataLoader(
                test_set,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                drop_last=False,
                pin_memory=True
            )

        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    if train_loader is None:
        logger.error('Was\'t able to create a dataset loader')
        return

    # * Init model
    model = get_model(opt)
    if opt['merge_bn'] and current_step > opt['merge_bn_startpoint']:
        logger.info('^_^ -----merging bnorm----- ^_^')
        model.merge_bnorm_test()

    logger.info(model.info_network())
    model.init_train()
    logger.info(model.info_params())

    # * Training
    for epoch in range(1000000):  # keep running
        for train_data in train_loader:
            current_step += 1

            if dataset_type == 'dnpatch' and current_step % 20000 == 0:
                train_loader.dataset.update_data()

            model.update_learning_rate(current_step)  # update learning rate
            model.feed_data(train_data)  # feed patch pairs
            model.optimize_parameters(current_step)  # optimize parameters

            # * merge bnorm
            if opt['merge_bn'] and opt['merge_bn_startpoint'] == current_step:
                logger.info('^_^ -----merging bnorm----- ^_^')
                model.merge_bnorm_train()
                model.print_network()

            # * Print info
            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # * Save model
            if current_step % opt['train']['checkpoint_save'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # * Perform test
            if current_step % opt['train']['checkpoint_test'] == 0 and test_loader is not None:
                avg_psnr = 0.0
                avg_ssim = 0.0
                avg_psnrb = 0.0
                idx = 0

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['H_path'][0])
                    img_name, _ = os.path.splitext(image_name_ext)

                    img_dir = Path(opt['path']['images']).joinpath(img_name)
                    img_dir.mkdir()

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])
                    QF = 1-visuals['QF']

                    # save estimated image E
                    save_img_path = img_dir.joinpath(f'{img_name}_{current_step}.png')
                    util.imsave(E_img, save_img_path)

                    # calculate PSNR
                    current_psnr = util.calculate_psnr(
                        E_img, H_img, border=border)

                    avg_psnr += current_psnr

                    # calculate SSIM
                    current_ssim = util.calculate_ssim(
                        E_img, H_img, border=border
                    )

                    avg_ssim += current_ssim

                    # calculate PSNRB
                    current_psnrb = util.calculate_psnrb(
                        H_img, E_img, border=border)
                    avg_psnrb += current_psnrb

                    logger.info(
                        '{:->4d}--> {:>10s} | PSNR : {:<4.2f}dB | SSIM : {:<4.3f}dB | PSNRB : {:<4.2f}dB'.format(
                            idx,
                            image_name_ext,
                            current_psnr,
                            current_ssim,
                            current_psnrb
                        )
                    )

                    logger.info(
                        'predicted quality factor: {:<4.2f}'.format(float(QF))
                    )

                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                avg_psnrb = avg_psnrb / idx

                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB, Average SSIM : {:<.3f}dB, Average PSNRB : {:<.2f}dB\n'.format(
                    epoch, current_step, avg_psnr, avg_ssim, avg_psnrb))

        if epoch % 50 == 0:
            logger.info('Saving the final model.')
            model.save('latest')
        logger.info('End of training.')


if __name__ == '__main__':
    main()
