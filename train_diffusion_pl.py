import argparse
import os
import yaml
from diffusion_pl.diffuser_template_adverse_weather import DenoisingDiffusion
import os
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
 
def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Training Patch-Based Denoising Diffusion Models')
    parser.add_argument("--config", type=str,
                        default="allweather_test1.yml",
                        help="Path to the config file")
    parser.add_argument("--train_type", type=bool,
                        default=True,
                        help="train or test")
    parser.add_argument("--sampling_timesteps", type=int, default=2,
                        help="Number of implicit sampling steps for validation images")
    parser.add_argument("--image_folder", default='./save_images_training', type=str,
                        help="Location to save restored training images")
    parser.add_argument("--image_folder_test", default='./save_images_test', type=str,
                        help="Location to save restored validation images")
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    args = parser.parse_args()
    
    if args.train_type == True:    
        with open(os.path.join("configs", args.config), "r") as f:
            config = yaml.safe_load(f)
            config['sampling_timesteps'] = args.sampling_timesteps
            config['image_folder'] = args.image_folder + '_' + config['data']['dataset']
            config['train_type'] = args.train_type # train
        new_config = dict2namespace(config)
    else: # test
        with open(os.path.join("configs", args.config), "r") as f:
            config = yaml.safe_load(f)
            config['sampling_timesteps'] = args.sampling_timesteps
            config['sampling']['batch_size'] = 1 # test bs=1
            config['image_folder'] = args.image_folder_test + '_' + config['data']['dataset']
            config['train_type'] = args.train_type # test
        new_config = dict2namespace(config)
    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def main():
    args, config = parse_args_and_config()
    # resume training process 
    if config.RESUME == False:
        resume_checkpoint_path = None
    else:
        resume_checkpoint_path = config.resume_checkpoint_path
        
    print("=> creating denoising-diffusion model...")
    diffusion = DenoisingDiffusion(config)

    if not os.path.exists(config.image_folder):
        os.makedirs(config.image_folder)

    output_dir =  './logs'
    logger = TensorBoardLogger(name=config.log_name,save_dir=output_dir)
    
    trainer = pl.Trainer(
        check_val_every_n_epoch=config.Trainer.check_val_every_n_epoch,
        max_steps=config.Trainer.max_steps,
        accelerator=config.Trainer.accelerator,
        devices=config.Trainer.devices,
        precision=config.Trainer.precision,
        accumulate_grad_batches = config.Trainer.accumulate_grad_batches,
        logger=logger,
        strategy='ddp_find_unused_parameters_true',
        enable_progress_bar=True,
        log_every_n_steps=config.Trainer.log_every_n_steps
    )
    if config.train_type == True:
        trainer.fit(diffusion,ckpt_path=resume_checkpoint_path)
    else:
        # pass
        # if you want to evaluate performance directly, you could opt to unannotate following code and add your .ckpt path
        trainer.validate(diffusion,ckpt_path='your path of ckpt')
    
if __name__ == "__main__":
    main()
