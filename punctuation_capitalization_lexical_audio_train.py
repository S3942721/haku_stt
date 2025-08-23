"""
Punctuation and Capitalization Lexical Audio Training Script
Converted from NeMo tutorial notebook to standalone Python script

This script trains a model that predicts punctuation and capitalization using both 
text and audio features for each word in a sentence to make ASR output more readable.
"""

import os
import argparse
import wget
import torch
import lightning.pytorch as pl
from omegaconf import OmegaConf

from nemo.utils.exp_manager import exp_manager
from nemo.collections import nlp as nemo_nlp
from nemo.collections import asr as nemo_asr

def download_data_script(work_dir, branch='main'):
    """Download the LibriTTS data preparation script"""
    os.makedirs(work_dir, exist_ok=True)
    script_path = os.path.join(work_dir, 'get_libritts_data.py')
    
    if not os.path.exists(script_path):
        print('Downloading get_libritts_data.py...')
        url = f'https://raw.githubusercontent.com/NVIDIA/NeMo/{branch}/examples/nlp/token_classification/data/get_libritts_data.py'
        wget.download(url, work_dir)
        print(f'\nDownloaded to {script_path}')
    else:
        print(f'get_libritts_data.py already exists at {script_path}')
    
    return script_path

def download_tarred_dataset_script(work_dir, branch='main'):
    """Download the tarred dataset creation script"""
    script_path = os.path.join(work_dir, 'create_punctuation_capitalization_tarred_dataset.py')
    
    if not os.path.exists(script_path):
        print('Downloading create_punctuation_capitalization_tarred_dataset.py...')
        url = f'https://raw.githubusercontent.com/NVIDIA/NeMo/{branch}/examples/nlp/token_classification/data/create_punctuation_capitalization_tarred_dataset.py'
        wget.download(url, work_dir)
        print(f'\nDownloaded to {script_path}')
    else:
        print(f'create_punctuation_capitalization_tarred_dataset.py already exists')
    
    return script_path

def download_config(work_dir, model_config, branch='main'):
    """Download the model configuration file"""
    config_dir = os.path.join(work_dir, 'configs')
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, model_config)
    
    if not os.path.exists(config_path):
        print('Downloading config file...')
        url = f'https://raw.githubusercontent.com/NVIDIA/NeMo/{branch}/examples/nlp/token_classification/conf/{model_config}'
        wget.download(url, config_dir)
        print(f'\nDownloaded to {config_path}')
    else:
        print(f'Config file already exists at {config_path}')
    
    return config_path

def prepare_libritts_data(data_dir, work_dir, datasets=['dev_clean', 'dev_other'], clean=True):
    """Download and preprocess LibriTTS data"""
    script_path = os.path.join(work_dir, 'get_libritts_data.py')
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Data script not found at {script_path}. Run download_data_script first.")
    
    datasets_str = ','.join(datasets)
    cmd = f"python {script_path} --data_dir {data_dir} --data_set {datasets_str}"
    if clean:
        cmd += " --clean"
    
    print(f"Running: {cmd}")
    os.system(cmd)
    
    # Verify files were created
    required_files = ['text_dev.txt', 'labels_dev.txt', 'audio_dev.txt']
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Expected file {file_path} was not created")
    
    print("Data preparation completed successfully")

def create_tarred_dataset(data_dir, work_dir, num_batches_per_tarfile=20, tokens_in_batch=1024):
    """Create tarred dataset for large-scale training"""
    script_path = os.path.join(work_dir, 'create_punctuation_capitalization_tarred_dataset.py')
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Tarred dataset script not found at {script_path}")
    
    output_dir = os.path.join(data_dir, 'train_tarred')
    
    cmd = f"""python {script_path} \
        --text {data_dir}/text_dev.txt \
        --labels {data_dir}/labels_dev.txt \
        --output_dir {output_dir} \
        --num_batches_per_tarfile {num_batches_per_tarfile} \
        --tokens_in_batch {tokens_in_batch} \
        --lines_per_dataset_fragment 4000 \
        --tokenizer_name bert-base-uncased \
        --n_jobs 2 \
        --use_audio \
        --sample_rate 16000 \
        --audio_file {data_dir}/audio_dev.txt"""
    
    print(f"Creating tarred dataset...")
    os.system(cmd)
    
    print(f"Tarred dataset created in {output_dir}")
    return output_dir

def setup_config(config_path, data_dir, args):
    """Setup and modify the model configuration"""
    config = OmegaConf.load(config_path)
    
    # Data paths
    config.model.train_ds.ds_item = data_dir
    config.model.validation_ds.ds_item = data_dir
    
    # Remove test_ds if it exists (we only have train/dev)
    if 'test_ds' in config.model:
        del config.model.test_ds
    
    # Model parameters
    config.model.language_model.pretrained_model_name = args.bert_model
    config.model.audio_encoder.pretrained_model = args.asr_model
    config.model.train_ds.tokens_in_batch = args.tokens_in_batch
    config.model.validation_ds.tokens_in_batch = args.tokens_in_batch
    config.model.optim.lr = args.learning_rate
    
    # Data files
    config.model.train_ds.text_file = 'text_dev.txt'
    config.model.train_ds.labels_file = 'labels_dev.txt'
    config.model.train_ds.audio_file = 'audio_dev.txt'
    config.model.validation_ds.text_file = 'text_dev.txt'
    config.model.validation_ds.labels_file = 'labels_dev.txt'
    config.model.validation_ds.audio_file = 'audio_dev.txt'
    
    # Audio preloading
    config.model.train_ds.preload_audios = args.preload_audios
    config.model.validation_ds.preload_audios = args.preload_audios
    
    # Trainer configuration
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    config.trainer.devices = args.devices
    config.trainer.accelerator = accelerator
    config.trainer.precision = 16 if torch.cuda.is_available() and args.use_mixed_precision else 32
    config.trainer.max_epochs = args.max_epochs
    config.trainer.strategy = 'auto'
    
    # Experiment manager
    config.exp_manager.use_datetime_version = False
    config.exp_manager.explicit_log_dir = args.exp_dir
    
    # Tarred dataset configuration
    if args.use_tarred_dataset:
        tarred_dir = os.path.join(data_dir, 'train_tarred')
        config.model.train_ds.ds_item = tarred_dir
        config.model.train_ds.use_tarred_dataset = True
        config.model.train_ds.tar_metadata_file = 'metadata.punctuation_capitalization.tokens1024.max_seq_length512.bert-base-uncased.json'
    
    return config

def train_model(config, resume_from_checkpoint=None):
    """Train the punctuation and capitalization model"""
    # Create trainer
    trainer = pl.Trainer(**config.trainer)
    
    # Setup experiment manager
    exp_dir = exp_manager(trainer, config.get("exp_manager", None))
    
    # Initialize model
    print("Initializing model...")
    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        model = nemo_nlp.models.PunctuationCapitalizationLexicalAudioModel.restore_from(resume_from_checkpoint)
        model.set_trainer(trainer)
    else:
        model = nemo_nlp.models.PunctuationCapitalizationLexicalAudioModel(cfg=config.model, trainer=trainer)
    
    # Start training
    print("Starting training...")
    trainer.fit(model, ckpt_path=resume_from_checkpoint)
    
    print(f"Training completed. Experiment directory: {exp_dir}")
    return model, exp_dir

def finetune_pretrained_model(checkpoint_path, data_dir, config):
    """Finetune a pretrained model with new data"""
    print(f"Loading pretrained model from: {checkpoint_path}")
    model = nemo_nlp.models.PunctuationCapitalizationLexicalAudioModel.restore_from(checkpoint_path)
    
    # Update configuration for new data
    model.update_config_after_restoring_from_checkpoint(
        train_ds={
            'ds_item': data_dir,
            'text_file': 'text_dev.txt',
            'labels_file': 'labels_dev.txt',
            'audio_file': 'audio_dev.txt',
            'tokens_in_batch': config.model.train_ds.tokens_in_batch,
        },
        validation_ds={
            'ds_item': data_dir,
            'text_file': 'text_dev.txt',
            'labels_file': 'labels_dev.txt',
            'audio_file': 'audio_dev.txt',
            'tokens_in_batch': config.model.validation_ds.tokens_in_batch,
        },
    )
    
    # Create trainer and setup data
    trainer = pl.Trainer(**config.trainer)
    model.set_trainer(trainer)
    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)
    
    # Start finetuning
    print("Starting finetuning...")
    trainer.fit(model)
    
    return model

def main():
    parser = argparse.ArgumentParser(
        description="Train Punctuation and Capitalization Model with Lexical and Audio Features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with data preparation
  python punctuation_capitalization_lexical_audio_train.py --data_dir ./data --work_dir ./work

  # Training with tarred dataset
  python punctuation_capitalization_lexical_audio_train.py --data_dir ./data --use_tarred_dataset

  # Finetuning from checkpoint
  python punctuation_capitalization_lexical_audio_train.py --data_dir ./data --finetune_from ./model.nemo

  # Training with custom models
  python punctuation_capitalization_lexical_audio_train.py --data_dir ./data --bert_model distilbert-base-uncased --asr_model stt_en_conformer_ctc_medium
        """
    )
    
    # Required arguments
    parser.add_argument("--data_dir", required=True, help="Directory to store/load training data")
    parser.add_argument("--work_dir", default="./work", help="Working directory for scripts and configs")
    
    # Data preparation
    parser.add_argument("--prepare_data", action="store_true", help="Download and prepare LibriTTS data")
    parser.add_argument("--datasets", nargs="+", default=["dev_clean", "dev_other"], 
                        help="LibriTTS datasets to download")
    parser.add_argument("--clean_raw_data", action="store_true", 
                        help="Remove raw LibriTTS data after processing")
    
    # Tarred dataset
    parser.add_argument("--create_tarred_dataset", action="store_true", 
                        help="Create tarred dataset for large-scale training")
    parser.add_argument("--use_tarred_dataset", action="store_true", 
                        help="Use tarred dataset for training")
    parser.add_argument("--num_batches_per_tarfile", type=int, default=20,
                        help="Number of batches per tar file")
    
    # Model configuration
    parser.add_argument("--model_config", default="punctuation_capitalization_lexical_audio_config.yaml",
                        help="Model configuration file name")
    parser.add_argument("--bert_model", default="bert-base-uncased",
                        help="Pretrained BERT model name")
    parser.add_argument("--asr_model", default="stt_en_conformer_ctc_small",
                        help="Pretrained ASR model name")
    
    # Training parameters
    parser.add_argument("--tokens_in_batch", type=int, default=1024, help="Number of tokens per batch")
    parser.add_argument("--max_seq_length", type=int, default=64, help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=0.00002, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=1, help="Maximum number of training epochs")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use")
    parser.add_argument("--use_mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--preload_audios", action="store_true", help="Preload audio files into memory")
    
    # Experiment
    parser.add_argument("--exp_dir", default="Punctuation_And_Capitalization_Lexical_Audio",
                        help="Experiment directory name")
    parser.add_argument("--resume_from_checkpoint", help="Path to checkpoint to resume training from")
    parser.add_argument("--finetune_from", help="Path to pretrained model checkpoint for finetuning")
    
    # Utility
    parser.add_argument("--list_models", action="store_true", help="List available models and exit")
    parser.add_argument("--branch", default="main", help="NeMo branch to download scripts from")
    
    args = parser.parse_args()
    
    # List available models and exit
    if args.list_models:
        print("Available BERT-like models:")
        print(nemo_nlp.modules.get_pretrained_lm_models_list())
        print("\nAvailable ASR models:")
        print(nemo_asr.models.ASRModel.list_available_models())
        return
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Download required scripts
    print("Downloading required scripts...")
    download_data_script(args.work_dir, args.branch)
    download_tarred_dataset_script(args.work_dir, args.branch)
    config_path = download_config(args.work_dir, args.model_config, args.branch)
    
    # Prepare data if requested
    if args.prepare_data:
        print("Preparing LibriTTS data...")
        prepare_libritts_data(args.data_dir, args.work_dir, args.datasets, args.clean_raw_data)
    
    # Create tarred dataset if requested
    if args.create_tarred_dataset:
        print("Creating tarred dataset...")
        create_tarred_dataset(args.data_dir, args.work_dir, args.num_batches_per_tarfile, args.tokens_in_batch)
    
    # Verify data exists
    required_files = ['text_dev.txt', 'labels_dev.txt', 'audio_dev.txt']
    for file in required_files:
        file_path = os.path.join(args.data_dir, file)
        if not os.path.exists(file_path):
            print(f"Error: Required file {file_path} not found.")
            print("Please run with --prepare_data to download and prepare the data.")
            return
    
    # Setup configuration
    print("Setting up model configuration...")
    config = setup_config(config_path, args.data_dir, args)
    
    # Print configuration
    print("\nModel Configuration:")
    print(OmegaConf.to_yaml(config))
    
    try:
        if args.finetune_from:
            # Finetune from pretrained checkpoint
            model = finetune_pretrained_model(args.finetune_from, args.data_dir, config)
        else:
            # Train from scratch or resume
            model, exp_dir = train_model(config, args.resume_from_checkpoint)
            
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
