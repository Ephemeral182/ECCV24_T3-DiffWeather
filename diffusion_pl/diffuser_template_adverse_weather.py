import os
import torch

import pytorch_lightning as pl
import torchvision
import torchvision.transforms as tfs
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import lpips
from Depth_Anything.depth_anything.dpt import DepthAnything
from loss.CL1 import PSNRLoss
from models_prompt.modules.DenoisingNAFNet_arch import NAFNet
from utils.Allweather import *
from utils.AGAN_data import *
from utils.imgqual_utils import batch_PSNR, batch_SSIM
from utils.save_image import save_colormapped_image

from diffusion_pl.ema import EMA
import diffusion_pl.pipeline as pipeline
import utils

class DenoisingDiffusion(pl.LightningModule):
    def __init__(self, config):
        super(DenoisingDiffusion,self).__init__()
        self.config = config

        self.loss_psnr = PSNRLoss()
        self.model = NAFNet(img_channel=config.model.img_channel,
                    out_channel=config.model.out_channel,
                    width=config.model.width,
                    middle_blk_num=config.model.middle_blk_num,
                    enc_blk_nums=config.model.enc_blk_nums,
                    dec_blk_nums=config.model.dec_blk_nums,
                    is_prompt_pool=True,
                    )
        self.depth = DepthAnything.from_pretrained('LiheYoung/depth_anything_vits14').eval()
        self.save_path=self.config.image_folder
        self.max_steps = self.config.Trainer.max_steps
        self.epochs = self.config.Trainer.max_epochs
        self.val_crop=True
        
        self.DiffSampler = pipeline.SR3Sampler(
            model=self.model,
            scheduler = pipeline.create_SR3scheduler(self.config.diffusion, 'train')
        )
        self.DiffSampler.scheduler.set_timesteps(self.config.sampling_timesteps)

        self.lpips_fn = lpips.LPIPS(net='alex')
        self.automatic_optimization = True
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.save_hyperparameters()
    def closest_multiple_of_14(self,n):
        return round(n / 14.0) * 14
    def lpips_score_fn(self,x,gt):
        self.lpips_fn.to(self.device)
        x = x.to(self.device)
        gt = gt.to(self.device)
        lp_score = self.lpips_fn(
            gt * 2 - 1, x * 2 - 1
        )
        return torch.mean(lp_score).item()

    def configure_optimizers(self):
        parameters=[
                    {'params':self.model.parameters()},
                    ]
        optimizer = utils.optimize.get_optimizer(self.config, parameters)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_steps//len(self.train_dataloader()), eta_min=self.config.optim.lr* 1e-2)

        self.optimizer = optimizer
        return [optimizer], [scheduler]
    
    def configure_callbacks(self):
        checkpoint_callback = ModelCheckpoint(
            monitor='psnr',
            filename='Test1-epoch{epoch:02d}-PSNR-{psnr:.3f}-SSIM-{ssim:.4f}',
            auto_insert_metric_name=False,   
            every_n_epochs=1,
            save_top_k=6,
            mode = "max",
            save_last=True
        )
        lr_monitor_callback = LearningRateMonitor(logging_interval='step')
        ema_callback = EMA(decay=0.995,every_n_steps=1)
        return [checkpoint_callback,lr_monitor_callback,ema_callback]

    
    def training_step(self, batch, batch_idx):
        x,gt,img_id =  batch
        b,c,h,w = x.shape
        # construct degradation residual
        gt_residual = gt - x

        target_height = self.closest_multiple_of_14(x.shape[2])
        target_width = self.closest_multiple_of_14(x.shape[3])
        depth,features = self.depth(tfs.Resize([target_height,target_width])(x))
        batch = torch.cat([x,gt_residual],dim=1)
        
        bs = batch.shape[0]
        timesteps = torch.randint(0, self.DiffSampler.scheduler.config.num_train_timesteps, (bs,), device=self.device).long()
        # add noise
        noise = torch.randn(batch.shape).to(self.device)
        noisy_images = self.DiffSampler.scheduler.add_noise(batch, timesteps=timesteps, noise=noise)
        # train 
        [residual_train,prompt_loss_train] = self.model(noisy_images,timesteps,depth_feature = features[3][0])
        # diffusion loss
        loss_noise = self.loss_psnr(residual_train + x,gt)
        # training reverse sampling
        sample_residual,prompt_loss = self.DiffSampler.sample_high_res(x,train=True,depth_feature=features[3][0])
        samples = x + sample_residual
        # training psnr
        psnr_train_online = batch_PSNR(samples.detach().float(),gt.float(),ycbcr=True)

        loss_samples_psnr = self.loss_psnr(samples,gt)
        loss_samples = 0
        loss_samples += loss_samples_psnr

        loss_prompt_contrast = (prompt_loss + prompt_loss_train) * 0.5
        loss = loss_noise + loss_samples + loss_prompt_contrast
        self.log("train_loss",loss,prog_bar=True)
        self.log("train_psnr",psnr_train_online,prog_bar=True)
        return {"loss":loss}

    def validation_step(self,batch,batch_idx):
        
        input_x,target,img_id =  batch
        target_height = self.closest_multiple_of_14(input_x.shape[2])
        target_width = self.closest_multiple_of_14(input_x.shape[3])
        depth,features = self.depth(tfs.Resize([target_height,target_width])(input_x))
        # generate degradation residual from noise
        samples_residual,_ = self.DiffSampler.sample_high_res(input_x,train=False,depth_feature=features[3][0])

        samples = samples_residual + input_x
        if self.config.train_type == True:
            if batch_idx==0:
                filename = "sample_{}.png".format(self.current_epoch)
                save_image(samples[:5,...],os.path.join(self.save_path, filename))  
            if batch_idx==0:  
                filename = "sample_degraded_{}.png".format(self.current_epoch)
                save_colormapped_image(samples_residual[:5, ...], os.path.join(self.save_path, filename))
            if batch_idx==0:
                filename = "target_{}.png".format(self.current_epoch)
                save_image(target[:5,...],os.path.join(self.save_path, filename))
        else:
            filename = "sample_{}.png".format((img_id[0]))
            save_image(samples[:1,...],os.path.join(self.save_path, filename))  
            # filename = "sample_degraded_{}.png".format((img_id[0]))  
            # save_image(samples_residual[:1,...],os.path.join(self.save_path, filename))        
            # save_colormapped_image(samples_residual[:5, ...], os.path.join(self.save_path, filename))
            # filename = "target_{}.png".format((img_id[0]))
            # save_image(target[:5,...],os.path.join(self.save_path, filename))
        psnr = batch_PSNR(samples.float(),target.float(),ycbcr=True)
        ssim = batch_SSIM(samples.float(),target.float(),ycbcr=True)
        lpips_score = self.lpips_score_fn(samples.float(),target.float())

        self.log('psnr',psnr,sync_dist=True)
        self.log('ssim',ssim,sync_dist=True)
        self.log('lpips',lpips_score)

        
        return {"psnr":psnr,"ssim":ssim,"lpips":lpips_score}
    
    def train_dataloader(self):
        
        train_set = AllWeather(self.config.data.data_dir,train=True,size=self.config.data.image_size,crop=True)

        train_loader = DataLoader(train_set, 
                                  batch_size=self.config.training.batch_size, 
                                  shuffle=True, 
                                  num_workers=self.config.data.num_workers,
                                  pin_memory=True)

        return train_loader
    
    def val_dataloader(self):
        
        if self.config.data.dataset == 'Test1':
            val_set = Test1(self.config.data.val_data_dir,train=False,size=256,crop=True)
        if self.config.data.dataset == 'Raindrop':
            val_set = AGAN_Dataset(self.config.data.val_data_dir,train=False,size=256,crop=True)
        if self.config.data.dataset == 'Snow100k-S':
            val_set = Snow100kTest(self.config.data.val_data_dir,train=False,size=256,crop=True)
        if self.config.data.dataset == 'Snow100k-L':
            val_set = Snow100kTest(self.config.data.val_data_dir,train=False,size=256,crop=True)

        val_loader = DataLoader(val_set, batch_size=self.config.sampling.batch_size, shuffle=False, num_workers=self.config.data.num_workers,pin_memory=True)

        return val_loader



