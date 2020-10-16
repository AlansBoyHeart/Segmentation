#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import os, json, argparse, torch, warnings
warnings.filterwarnings("ignore")

import models
import models as module_arch
import evaluation.losses as module_loss
import evaluation.metrics as module_metric
import dataloaders.dataloader as module_data

from utils.logger import Logger
from trainer.trainer import Trainer
from models.PSPNet import PSPNet

#------------------------------------------------------------------------------
#   Get instance
#------------------------------------------------------------------------------
def get_instance(module, name, config, *args):
	return getattr(module, config[name]['type'])(*args, **config[name]['args'])


#------------------------------------------------------------------------------
#   Main function
#------------------------------------------------------------------------------
def main(config, resume):
	train_logger = Logger()
	#
	# Build model architecture
	# model = get_instance(module_arch, 'arch', config)   #取得PSPNet模型，PSPNet的backbone是resnet18模型
	model = models.PSPNet(**config["arch"]['args'])
	# model = torch.nn.DataParallel(model)
	# img_sz = config["train_loader"]["args"]["resize"]
	img_sz = 320
	# model.summary(input_shape=(3, img_sz, img_sz))

	# Setup data_loader instances
	# train_loader = get_instance(module_data, 'train_loader', config).loader
	# valid_loader = get_instance(module_data, 'valid_loader', config).loader
	train_loader = module_data.SegmentationDataLoader(**config["train_loader"]['args']).loader
	valid_loader = module_data.SegmentationDataLoader(**config["valid_loader"]['args']).loader


	# Get function handles of loss and metrics
	# loss = getattr(module_loss, config['loss'])
	loss = module_loss.custom_pspnet_loss
	# metrics = [getattr(module_metric, met) for met in config['metrics']]
	metrics = [module_metric.custom_pspnet_miou]

	# Build optimizer, learning rate scheduler.
	trainable_params = filter(lambda p: p.requires_grad, model.parameters())
	# optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
	optimizer = torch.optim.SGD(trainable_params, **config["optimizer"]['args'])
	# lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **config["lr_scheduler"]['args'])

	# Create trainer and start training
	trainer = Trainer(model, loss, metrics, optimizer, 
					  resume=resume,
					  config=config,
					  data_loader=train_loader,
					  valid_data_loader=valid_loader,
					  lr_scheduler=lr_scheduler,
					  train_logger=train_logger)
	trainer.train()


#------------------------------------------------------------------------------
#   Main execution
#------------------------------------------------------------------------------
if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"
	# Argument parsing
	parser = argparse.ArgumentParser(description='Train model')

	parser.add_argument('-c', '--config', default="./config/config_PSPNet.json", type=str,
						   help='config file path (default: None)')

	parser.add_argument('-r', '--resume', default=None, type=str,
						   help='path to latest checkpoint (default: None)')

	parser.add_argument('-d', '--device', default=True, type=str,
						   help='indices of GPUs to enable (default: all)')
 
	args = parser.parse_args()


	# Load config file
	if args.config:
		config = json.load(open(args.config))
		path = os.path.join(config['trainer']['save_dir'], config['name'])


	# Load config file from checkpoint, in case new config file is not given.
	# Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
	elif args.resume:
		config = torch.load(args.resume)['config']


	# AssertionError
	else:
		raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")
	

	# Set visible devices
	if args.device:
		os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


	# Run the main function
	main(config, args.resume)
