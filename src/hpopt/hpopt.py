"""Hyperparameter optimization with Ray Tune."""

import argparse
import os

# Fix Ray FutureWarning about accelerator env vars on 0 GPUs.
# This must be set before 'import ray' or before 'ray.init' depending on when it's read, but environment variables are best set early.
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
# Silence Ray V2 migration warnings
os.environ["RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS"] = "0"

import ray
from ray import tune
import jax


vision_datasets = ['mnist', 'electron-photon', 'quark-gluon']  # TODO: add medmnist
text_datasets = ['imdb']

def train(config) -> None:
    # Perform imports here to avoid warning messages when running only --help

    from src.quantum_transformers import datasets
    from src.quantum_transformers.transformers import Transformer, VisionTransformer
    from src.quantum_transformers.training import train_and_evaluate
    from src.quantum_transformers.swintransformer import SwinTransformer

    c = config  # Shorter alias for config

    num_classes = {'imdb': 2, 'mnist': 10, 'electron-photon': 2, 'quark-gluon': 2}  # TODO: add medmnist
    model: Transformer | VisionTransformer | SwinTransformer  # Update model type

    from src.quantum_transformers.quantum_layer import get_circuit

    if c['dataset'] in text_datasets:  # Text datasets
        if c['dataset'] == 'imdb':
            (train_dataloader, val_dataloader, test_dataloader), vocab, _ = datasets.get_imdb_dataloaders(
                data_dir=c['data_dir'], batch_size=c['batch_size'],
                max_seq_len=c['max_seq_len'], max_vocab_size=c['vocab_size'])
        else:
            raise ValueError(f"Unknown dataset {c['dataset']}")

        model = Transformer(num_tokens=len(vocab), max_seq_len=c['max_seq_len'], num_classes=num_classes[c['dataset']],
                            hidden_size=c['hidden_size'], num_heads=c['num_heads'],
                            num_transformer_blocks=c['num_transformer_blocks'], mlp_hidden_size=c['mlp_hidden_size'],
                            dropout=c['dropout'],
                            quantum_attn_circuit=get_circuit() if c['quantum'] else None,
                            quantum_mlp_circuit=get_circuit() if c['quantum'] else None)
    elif c.get('swin', False):
        train_dataloader, val_dataloader, test_dataloader = datasets.get_mnist_dataloaders(data_dir=c['data_dir'],
                                                                                               batch_size=c[
                                                                                                   'batch_size'])
        # Initialize SwinTransformer
        model = SwinTransformer(
            hidden_dim=c['hidden_size'],
            layers=(2, 2),
            heads=(2, 4),
            channels=1,
            num_classes=num_classes[c['dataset']],
            head_dim=4,
            window_size=7,
            downscaling_factors=(2, 1),
            relative_pos_embedding=True,
            quantum_attn_circuit=get_circuit() if c['quantum'] else None,
            quantum_mlp_circuit=get_circuit() if c['quantum'] else None
        )
    else:  # Vision datasets
        if c['dataset'] == 'mnist':
            train_dataloader, val_dataloader, test_dataloader = datasets.get_mnist_dataloaders(data_dir=c['data_dir'],
                                                                                               batch_size=c[
                                                                                                   'batch_size'])
        elif c['dataset'] == 'electron-photon':
            train_dataloader, val_dataloader, test_dataloader = datasets.get_electron_photon_dataloaders(
                data_dir=c['data_dir'], batch_size=c['batch_size'])
        elif c['dataset'] == 'quark-gluon':
            train_dataloader, val_dataloader, test_dataloader = datasets.get_quark_gluon_dataloaders(
                data_dir=c['data_dir'], batch_size=c['batch_size'])
        elif c['dataset'].startswith('medmnist-'):
            raise NotImplementedError("MedMNIST is not yet supported")  # TODO: add medmnist
            train_dataloader, val_dataloader, test_dataloader = datasets.get_medmnist_dataloaders(
                dataset=c['dataset'].split('-')[1], data_dir=c['data_dir'], batch_size=c['batch_size'])
        else:
            raise ValueError(f"Unknown dataset {c['dataset']}")

        # Initialize VisionTransformer
        model = VisionTransformer(num_classes=num_classes[c['dataset']], patch_size=c['patch_size'],
                                  hidden_size=c['hidden_size'], num_heads=c['num_heads'],
                                  num_transformer_blocks=c['num_transformer_blocks'],
                                  mlp_hidden_size=c['mlp_hidden_size'],
                                  pos_embedding=c['pos_embedding'], dropout=c['dropout'],
                                  quantum_attn_circuit=get_circuit() if c['quantum'] else None,
                                  quantum_mlp_circuit=get_circuit() if c['quantum'] else None)

    # Train and evaluate
    train_and_evaluate(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                       test_dataloader=test_dataloader, num_classes=num_classes[c['dataset']],
                       num_epochs=c['num_epochs'], lrs_peak_value=c['lrs_peak_value'],
                       lrs_warmup_steps=c['lrs_warmup_steps'], lrs_decay_steps=c['lrs_decay_steps'],
                       seed=c['seed'], use_ray=True)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='DO NOT RUN THIS DIRECTLY! Execute submit-ray-cluster.sh instead (see README.md for details).')

    argparser.add_argument('dataset', type=str, help='name of dataset to train on',
                           choices=vision_datasets + text_datasets)
    argparser.add_argument('--quantum', action='store_true', help='whether to use quantum transformers')
    argparser.add_argument('--swin', action='store_true', help='whether to use Swin Transformer for training')
    argparser.add_argument('--trials', type=int, default=1, help='number of trials to run')
    argparser.add_argument('--force-cpu', action='store_true', help='Force training on CPU even if GPU is available (or if no GPU found)')
    args, unknown = argparser.parse_known_args()
    print(f"args = {args}, unknown = {unknown}")

    common_params = {
        'seed': 42,
        'data_dir': '~/data',
        'dataset': args.dataset,
        'quantum': args.quantum,
        'swin': args.swin,
        'num_epochs': 1,
    }

    # 2. モードに応じたパラメータ空間の定義
    if args.quantum:
        # --- Quantum用 (シミュレーションコストを抑える設定) ---
        specific_params = {
            # シミュレーションが重いためバッチサイズは小さめに
            'batch_size': tune.choice([32, 64]),

            # QMLでは量子ビット数=Hidden Sizeになることが多いため、大きくしすぎない
            # (通常 4, 8, 16 程度が限界)
            'hidden_size': tune.choice([4, 8]),

            # 量子回路のパラメータ数が増えすぎないように抑制
            'mlp_hidden_size': tune.choice([4, 8, 16]),

            # 回路が深くなると勾配消失や計算コスト爆増するため浅くする
            'num_heads': tune.choice([1, 2]),
            'num_transformer_blocks': tune.choice([1, 2]),

            # 学習率などはClassicalと少し変える場合があるが、まずは広めに
            'lrs_peak_value': tune.loguniform(1e-3, 1e-1),
            'lrs_warmup_steps': tune.choice([0, 100]),
            'lrs_decay_steps': tune.choice([1000, 5000]),
            'dropout': tune.uniform(0.0, 0.1),  # 量子回路自体にノイズがあるならDropout不要なことも
        }

    else:
        # --- Classical (Swin/Standard) 用 (GPU性能を活かす設定) ---
        specific_params = {
            # GPU活用のためバッチサイズ大きめ
            'batch_size': tune.choice([64, 128, 256]),

            # 表現力を高めるために十分なサイズを確保
            'hidden_size': tune.choice([32, 64, 96]),
            'mlp_hidden_size': tune.choice([128, 256, 384]),

            # 深く、多頭に
            'num_heads': tune.choice([2, 4]),
            'num_transformer_blocks': tune.choice([2, 4, 6]),

            'lrs_peak_value': tune.loguniform(1e-4, 5e-3),
            'lrs_warmup_steps': tune.choice([100, 500]),
            'lrs_decay_steps': tune.choice([2000, 5000, 10000]),
            'dropout': tune.uniform(0.0, 0.3),
        }

    # 3. 辞書の結合 (Python 3.9+なら | 演算子が使えます)
    param_space = {**common_params, **specific_params}

    if args.dataset in text_datasets:
        param_space['max_seq_len'] = tune.choice([32, 64, 128, 256, 512])
        param_space['vocab_size'] = tune.choice([1000, 2000, 5000, 10000, 20000, 50000])
    elif args.dataset in vision_datasets:
        param_space['pos_embedding'] = tune.choice(['learn', 'sincos'])
        if args.dataset == 'mnist':
            if args.quantum:
                param_space['patch_size'] = tune.choice([7, 14])
            else:
                param_space['patch_size'] = tune.choice([4, 7])
        elif args.dataset == 'electron-photon':
            param_space['patch_size'] = tune.choice([4, 8, 16, 32])
        elif args.dataset == 'quark-gluon':
            param_space['patch_size'] = tune.choice([5, 10, 25])
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    ray.init()

    # GPU Check and Resource Configuration
    if args.force_cpu:
        print("Forced CPU mode enabled.")
        num_gpus_per_trial = 0
    else:
        try:
            # Check if JAX can see any GPUs
            gpus = jax.devices('gpu')
            if not gpus:
                 # This might happen if jax[cpu] is installed or CUDA is not visible
                 raise RuntimeError("No GPU found by JAX.")
            print(f"JAX detected {len(gpus)} GPU(s): {gpus}")
            num_gpus_per_trial = 1
        except RuntimeError as e:
             # JAX raises RuntimeError if 'gpu' backend is not found
             raise RuntimeError(f"No GPU found! ({e}). Use --force-cpu to run on CPU.")

    resources_per_trial = {"cpu": 4, "gpu": num_gpus_per_trial}
    tuner = tune.Tuner(
        tune.with_resources(train, resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            scheduler=tune.schedulers.ASHAScheduler(metric="val_auc", mode="max", max_t=param_space['num_epochs']),
            num_samples=args.trials,
        ),
        run_config=tune.RunConfig(),
        param_space=param_space,
    )
    results = tuner.fit()
