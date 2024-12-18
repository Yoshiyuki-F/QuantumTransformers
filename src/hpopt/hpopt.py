"""Hyperparameter optimization with Ray Tune."""

import argparse

import ray
from ray import tune, air


vision_datasets = ['mnist', 'electron-photon', 'quark-gluon']  # TODO: add medmnist
text_datasets = ['imdb']

def train(config) -> None:
    # Perform imports here to avoid warning messages when running only --help
    import tensorflow as tf
    tf.config.set_visible_devices([], device_type='GPU')  # Ensure TF does not see GPU and grab all GPU memory
    from src.quantum_transformers import datasets
    from src.quantum_transformers.transformers import Transformer, VisionTransformer
    from src.quantum_transformers.training import train_and_evaluate
    from src.quantum_transformers.swintransformer import SwinTransformer

    c = config  # Shorter alias for config
    tf.random.set_seed(c['seed'])  # For reproducible data loading

    num_classes = {'imdb': 2, 'mnist': 10, 'electron-photon': 2, 'quark-gluon': 2}  # TODO: add medmnist
    model: Transformer | VisionTransformer | SwinTransformer  # Update model type

    if c['pl']:
        from src.quantum_transformers.quantum_layer_pennylane import get_circuit
    else:
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
                            quantum_attn_circuit=get_circuit(qum_qubits=c['hidden_size']) if c['quantum'] else None,
                            quantum_mlp_circuit=get_circuit(qum_qubits=c['mlp_hidden_size']) if c['quantum'] else None,
                            use_pennylane=c['pl'])
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
            quantum_attn_circuit=get_circuit(qum_qubits=c['hidden_size']) if c['quantum'] else None,
            quantum_mlp_circuit=get_circuit(qum_qubits=c['hidden_size'] * 4) if c['quantum'] else None,
            use_pennylane=c['pl']
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
                                  quantum_attn_circuit=get_circuit(num_qubits=c['num_heads']) if c['quantum'] else None,
                                  quantum_mlp_circuit=get_circuit(num_qubits=c['mlp_hidden_size']) if c['quantum'] else None,
                                  use_pennylane=c['pl'])

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
    argparser.add_argument('--trials', type=int, default=10, help='number of trials to run')
    argparser.add_argument('--pl', action='store_true', help='whether to use pennylane')
    args, unknown = argparser.parse_known_args()
    print(f"args = {args}, unknown = {unknown}")

    param_space = {
        'seed': 42,
        'data_dir': tune.choice(['~/.tensorflow_datasets']),
        'dataset': args.dataset,
        'quantum': args.quantum,
        'swin': args.swin,  # Add swin parameter to config
        'pl': args.pl,
        'num_epochs': 10,
        'batch_size': tune.choice([32]),
        'hidden_size': tune.choice([8]),
        'num_heads': tune.choice([1, 2]),
        'num_transformer_blocks': tune.choice([1, 2]),
        'mlp_hidden_size': tune.choice([4]),
        'dropout': tune.uniform(0.0, 0.5),
        'lrs_peak_value': tune.loguniform(1e-5, 1),
        'lrs_warmup_steps': tune.choice([0, 1000, 5000, 10000]),
        'lrs_decay_steps': tune.choice([50000, 100000, 500000, 1000000]),
    }

    if args.dataset in text_datasets:
        param_space['max_seq_len'] = tune.choice([32, 64, 128, 256, 512])
        param_space['vocab_size'] = tune.choice([1000, 2000, 5000, 10000, 20000, 50000])
    elif args.dataset in vision_datasets:
        param_space['pos_embedding'] = tune.choice(['learn', 'sincos'])
        if args.dataset == 'mnist':
            param_space['patch_size'] = tune.choice([4, 7, 14, 28])
        elif args.dataset == 'electron-photon':
            param_space['patch_size'] = tune.choice([4, 8, 16, 32])
        elif args.dataset == 'quark-gluon':
            param_space['patch_size'] = tune.choice([5, 10, 25])
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    ray.init()

    resources_per_trial = {"cpu": 12, "gpu": 1}
    tuner = tune.Tuner(
        tune.with_resources(train, resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            scheduler=tune.schedulers.ASHAScheduler(metric="val_auc", mode="max", max_t=param_space['num_epochs']),
            num_samples=args.trials,
        ),
        run_config=air.RunConfig(),
        param_space=param_space,
    )
    results = tuner.fit()
