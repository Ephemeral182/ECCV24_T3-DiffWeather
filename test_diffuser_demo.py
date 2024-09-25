import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import argparse
import yaml
from pathlib import Path
from collections import OrderedDict 
from torch.utils.data import DataLoader
from utils.Allweather import *
from utils.AGAN_data import *
from utils.RainDS import *
from torchvision.utils import save_image
from Depth_Anything.depth_anything.dpt import DepthAnything
from models_prompt.modules.DenoisingNAFNet_arch import NAFNet
import diffusion_pl.pipeline as pipeline
from tqdm import tqdm

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def parse_args_and_config():
    config_path = Path(__file__).resolve().parent.joinpath("configs")
    parser = argparse.ArgumentParser(description='Test Denoising Diffusion')
    parser.add_argument("--config", type=str,
                        default=str(config_path.joinpath("allweather_demo.yml")),
                        help="Path to the config file")
    parser.add_argument("--model_path", type=str, default="pretrained/T3_DiffWeather_demo.ckpt",
                        help="Path to model")
    parser.add_argument("--sampling_timesteps", type=int, default=2,
                        help="Number of implicit sampling steps for validation image patches")
    parser.add_argument("--image_folder_test", default='./save_images_test', type=str,
                        help="Location to save restored validation images")
    
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        config['sampling_timesteps']=args.sampling_timesteps
        config['image_folder'] = args.image_folder_test + '_' + config['data']['dataset']
        new_config = dict2namespace(config)

    return args, new_config

class DiffusionSampler(torch.nn.Module):
    def __init__(self, config, data_loader, model_path):
        super(DiffusionSampler, self).__init__()
        self.config = config
        if not os.path.exists(config.image_folder):
            os.makedirs(config.image_folder)
        self.model = NAFNet(img_channel=config.model.img_channel,
                    out_channel=config.model.out_channel,
                    width=config.model.width,
                    middle_blk_num=config.model.middle_blk_num,
                    enc_blk_nums=config.model.enc_blk_nums,
                    dec_blk_nums=config.model.dec_blk_nums,
                    is_prompt_pool=True).cuda().eval()
        
        state_dict = torch.load(model_path)
        new_state_dict = OrderedDict()
        for k in state_dict['state_dict']:
                    print(k)
                    if k[:6] != 'model.':
                        continue
                    name = k[6:]
                    print(name)
                    new_state_dict[name] = (state_dict['state_dict'][k])
        self.model.load_state_dict(new_state_dict, strict=True)
        self.depth = DepthAnything.from_pretrained('LiheYoung/depth_anything_vits14').eval().cuda()
        self.DiffSampler = pipeline.SR3Sampler(
            model=self.model,
            scheduler = pipeline.create_SR3scheduler(self.config.diffusion, 'train')
        )
        print("setting timestep to: {}".format(self.config.sampling_timesteps))
        self.DiffSampler.scheduler.set_timesteps(self.config.sampling_timesteps)

        self.data_loader = data_loader
    def closest_multiple_of_14(self,n):
        return round(n / 14.0) * 14
    def sample_test(self):
        
        for i, batch in tqdm(enumerate(self.data_loader), total=len(self.data_loader), desc="Testing"):
            x, id = batch
            target_height = self.closest_multiple_of_14(x.shape[2])
            target_width = self.closest_multiple_of_14(x.shape[3])
            depth, features = self.depth(tfs.Resize([target_height, target_width])(x.cuda()))
            samples_, _ = self.DiffSampler.sample_high_res(x.cuda(), train=False, depth_feature=features[3][0])
            
            output = samples_ + x.cuda()
            
            fn = f"sample_{id[0]}.png"
            result_fp = str(Path(config.image_folder).joinpath(fn))

            save_image(output, result_fp)

if __name__ == "__main__":
    args, config = parse_args_and_config()
    
    test_set = Real(config.data.val_data_dir,train=False,crop=False)
    loader = DataLoader(test_set, batch_size=1, shuffle=False)

    diff_sampler = DiffusionSampler(config, loader, args.model_path)
    diff_sampler.sample_test()