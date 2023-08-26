import os
import tarfile

import numpy as np
import gdown
import tensorflow_datasets as tfds
import tensorflow as tf
# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type='GPU')


class NumPyFolderDataset(tfds.core.GeneratorBasedBuilder):
    """
    A dataset consisting of NumPy arrays stored in folders (one folder per class),
    downloaded from Google Drive in .tar.xz format.
    """
    VERSION = tfds.core.Version('1.0.0')  # to avoid ValueError

    def __init__(self, name, gdrive_id, img_shape, num_classes, **kwargs):
        """Creates a NumPyFolderDataset."""
        self.name = name
        self.gdrive_id = gdrive_id
        self.img_shape = img_shape
        self.num_classes = num_classes
        super().__init__(**kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Tensor(shape=self.img_shape, dtype=np.float32),
                'label': tfds.features.ClassLabel(num_classes=self.num_classes),
            }),
            supervised_keys=('image', 'label')
        )

    def _split_generators(self, _):
        """Returns SplitGenerators."""
        if os.path.exists(f'{self.data_dir}/{self.name}'):
            print(f'{self.data_dir}/{self.name} already exists, skipping download')
        else:
            os.makedirs(f'{self.data_dir}/{self.name}')
            gdown.download(id=self.gdrive_id, output=f'{self.data_dir}/{self.name}.tar.xz', quiet=False)
            with tarfile.open(f'{self.data_dir}/{self.name}.tar.xz', 'r:xz') as f:
                print(f'Extracting {self.name}.tar.xz to {self.data_dir}')
                f.extractall(self.data_dir)
            os.remove(f'{self.data_dir}/{self.name}.tar.xz')

        dataset_path = tfds.core.Path(f'{self.data_dir}/{self.name}')
        return {
            'train': self._generate_examples(dataset_path / 'train'),
            'test': self._generate_examples(dataset_path / 'test'),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        class_names = {c: i for i, c in enumerate(sorted([f.name for f in path.glob('*')]))}
        for class_folder in path.glob('*'):
            for f in class_folder.glob('*.npy'):
                try:
                    yield f"{class_folder.name}_{f.name}", {
                        'image': np.load(f),
                        'label': class_names[class_folder.name],
                    }
                except FileNotFoundError as e:
                    print(e)


def datasets_to_dataloaders(train_dataset, valid_dataset, batch_size, drop_remainder=True, transform=None):
    # Shuffle train dataset
    train_dataset = train_dataset.shuffle(train_dataset.cardinality(), reshuffle_each_iteration=True)

    # Batch
    train_dataset = train_dataset.batch(batch_size, drop_remainder=drop_remainder)
    valid_dataset = valid_dataset.batch(batch_size, drop_remainder=drop_remainder)

    # Transform
    if transform is not None:
        train_dataset = train_dataset.map(transform, num_parallel_calls=tf.data.AUTOTUNE)
        valid_dataset = valid_dataset.map(transform, num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    valid_dataset = valid_dataset.prefetch(tf.data.AUTOTUNE)

    # Convert to NumPy for JAX
    return tfds.as_numpy(train_dataset), tfds.as_numpy(valid_dataset)


def get_mnist_dataloaders(data_dir: str = '~/data', batch_size: int = 1, drop_remainder: bool = True):
    """Returns dataloaders for the MNIST dataset (computer vision, multi-class classification)"""
    data_dir = os.path.expanduser(data_dir)

    # Load datasets
    train_dataset = tfds.load(name='mnist', split='train', as_supervised=True, data_dir=data_dir)
    valid_dataset = tfds.load(name='mnist', split='test', as_supervised=True, data_dir=data_dir)

    def normalize_image(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return (image - 0.1307) / 0.3081, label

    return datasets_to_dataloaders(train_dataset, valid_dataset, batch_size, drop_remainder=drop_remainder, transform=normalize_image)


def get_electron_photon_dataloaders(data_dir: str = '~/data', batch_size: int = 1, drop_remainder: bool = True):
    """Returns dataloaders for the electron-photon dataset (computer vision - particle physics, binary classification)"""
    data_dir = os.path.expanduser(data_dir)

    # Load datasets
    electron_photon_builder = NumPyFolderDataset(data_dir=data_dir, name="electron-photon", gdrive_id="1VAqGQaMS5jSWV8gTXw39Opz-fNMsDZ8e",
                                                 img_shape=(32, 32, 2), num_classes=2)
    electron_photon_builder.download_and_prepare(download_dir=data_dir)
    train_dataset = electron_photon_builder.as_dataset(split='train', as_supervised=True)
    valid_dataset = electron_photon_builder.as_dataset(split='test', as_supervised=True)

    return datasets_to_dataloaders(train_dataset, valid_dataset, batch_size, drop_remainder=drop_remainder)


def get_quark_gluon_dataloaders(data_dir: str = '~/data', batch_size: int = 1, drop_remainder: bool = True):
    """Returns dataloaders for the quark-gluon dataset (computer vision - particle physics, binary classification)"""
    data_dir = os.path.expanduser(data_dir)

    # Load datasets
    quark_gluon_builder = NumPyFolderDataset(data_dir=data_dir, name="quark-gluon", gdrive_id="1G6HJKf3VtRSf7JLms2t1ofkYAldOKMls",
                                             img_shape=(125, 125, 3), num_classes=2)
    quark_gluon_builder.download_and_prepare(download_dir=data_dir)
    train_dataset = quark_gluon_builder.as_dataset(split='train', as_supervised=True)
    valid_dataset = quark_gluon_builder.as_dataset(split='test', as_supervised=True)

    return datasets_to_dataloaders(train_dataset, valid_dataset, batch_size, drop_remainder=drop_remainder)


def get_medmnist_dataloaders(dataset: str, data_dir: str = '~/data', batch_size: int = 1, drop_remainder: bool = True):
    """Returns dataloaders for a MedMNIST dataset"""
    raise NotImplementedError


def get_imdb_dataloaders(data_dir: str = '~/data', batch_size: int = 1, drop_remainder: bool = True,
                         max_vocab_size: int = 8192, max_seq_len: int = 512):
    """
    Returns dataloaders for the IMDB sentiment analysis dataset (natural language processing, binary classification),
    as well as the vocabulary and tokenizer.
    """
    import tensorflow_text as tf_text
    from tensorflow_text.tools.wordpiece_vocab.bert_vocab_from_dataset import bert_vocab_from_dataset

    data_dir = os.path.expanduser(data_dir)

    # Load datasets
    train_dataset = tfds.load(name='imdb_reviews', split='train', as_supervised=True, data_dir=data_dir)
    valid_dataset = tfds.load(name='imdb_reviews', split='test', as_supervised=True, data_dir=data_dir)

    # Build vocabulary and tokenizer
    bert_tokenizer_params = dict(lower_case=True)
    vocab = bert_vocab_from_dataset(
        train_dataset.batch(10_000).prefetch(tf.data.AUTOTUNE).map(lambda x, _: x),
        vocab_size=max_vocab_size,
        reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"],
        bert_tokenizer_params=bert_tokenizer_params
    )
    vocab_lookup_table = tf.lookup.StaticVocabularyTable(
        num_oov_buckets=1,
        initializer=tf.lookup.KeyValueTensorInitializer(keys=vocab,
                                                        values=tf.range(len(vocab), dtype=tf.int64))  # setting tf.int32 here causes an error
    )
    tokenizer = tf_text.BertTokenizer(vocab_lookup_table, **bert_tokenizer_params)

    def preprocess(text, label):
        # Tokenize
        tokens = tokenizer.tokenize(text).merge_dims(-2, -1)
        # Cast to int32 for compatibility with JAX (note that the vocabulary size is small)
        tokens = tf.cast(tokens, tf.int32)
        # Pad (all sequences to the same length so that JAX jit compiles the model only once)
        padded_inputs, _ = tf_text.pad_model_inputs(tokens, max_seq_length=max_seq_len)
        return padded_inputs, label

    return datasets_to_dataloaders(train_dataset, valid_dataset, batch_size, drop_remainder=drop_remainder, transform=preprocess), vocab, tokenizer
