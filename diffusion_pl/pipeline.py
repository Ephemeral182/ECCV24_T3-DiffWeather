import torch
from tqdm import tqdm
from functools import partial

from diffusers import DDIMScheduler


class SR3scheduler(DDIMScheduler):
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001,beta_end: float = 0.02, beta_schedule: str = 'linear',diff_chns=3):
        super().__init__(num_train_timesteps, beta_start ,beta_end,beta_schedule,prediction_type= "sample")
        # Initialize other attributes specific to SR3scheduler class
        # ...
        self.diff_chns = diff_chns
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # Only modify the last three channels of the tensor (assuming channels are in the second dimension)
        num_channels = original_samples.shape[1]
        if num_channels > self.diff_chns:
            original_samples_select = original_samples[:, -self.diff_chns:].contiguous()
            noise_select = noise[:, -self.diff_chns:].contiguous()

            noisy_samples_select = sqrt_alpha_prod * original_samples_select + sqrt_one_minus_alpha_prod * noise_select

            noisy_samples = original_samples.clone()
            noisy_samples[:, -self.diff_chns:] = noisy_samples_select
        else:
            noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

        return noisy_samples

def create_SR3scheduler(opt,phase):
    
    steps= opt.num_train_timesteps if phase=="train" else opt.sampling_timesteps
    scheduler=SR3scheduler(
        num_train_timesteps = steps,
        beta_start = opt.beta_start,
        beta_end = opt.beta_end,
        beta_schedule = opt.beta_schedule
    )
    return scheduler
    
class SR3Sampler():
    
    def __init__(self,model: torch.nn.Module,scheduler:SR3scheduler,eta: float =.0):
        self.model = model
        self.scheduler = scheduler
        self.eta = eta
    def sample_high_res(self,low_res: torch.Tensor, conditions=None,clean_structure=None,guidance=None,train=False,cross_feature_embed_=None,depth_feature=None):
        "Using Diffusers built-in samplers"
        device = next(self.model.parameters()).device
        eta = torch.Tensor([self.eta]).to(device)
        HR_image = torch.randn_like(low_res, device=device)
        low_res=low_res.to(device)
        preds = []
        for t in self.scheduler.timesteps:
            # print(t)
            if train == False: # testing process

                self.model.eval()
                with torch.no_grad():
                    if conditions is not None:
                        batch_data = conditions+[clean_structure,low_res,HR_image]
                    else:
                        batch_data=[low_res,HR_image]
                    out_put = self.model(torch.cat(batch_data, dim=1), t, depth_feature=depth_feature)
                    if isinstance(out_put, list) and len(out_put) > 1:
                        [noise, prompt_loss] = out_put
                    else:
                        noise = out_put
                        prompt_loss = 0
                HR_image = self.scheduler.step(model_output = noise,timestep = t,  sample = HR_image).prev_sample
                
                del noise
                torch.cuda.empty_cache()
            else:

                if conditions is not None:
                    batch_data = conditions+[clean_structure,low_res,HR_image]
                else:
                    batch_data=[low_res,HR_image]
                    out_put = self.model(torch.cat(batch_data, dim=1), t, depth_feature=depth_feature)
                    if isinstance(out_put, list) and len(out_put) > 1:
                        [noise, prompt_loss] = out_put
                    else:
                        noise = out_put
                        prompt_loss = 0
                HR_image = self.scheduler.step(model_output = noise,timestep = t,  sample = HR_image).prev_sample
    
                del noise
                torch.cuda.empty_cache()
        return HR_image,prompt_loss

def create_SR3Sampler(model,opt):
    
    scheduler = create_SR3scheduler(opt,"test")
    scheduler.set_timesteps(opt.sampling_timesteps)
    sampler = SR3Sampler(
        model = model,
        scheduler = scheduler,
        eta = opt.eta
    )
    return sampler
