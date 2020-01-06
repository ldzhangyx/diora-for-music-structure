## DIORA for music structure extraction

This program is based on the DIORA program, which is published on NAACL 2019.

Features:

- Change word embeddings to symbolic music representations
- Provide Nottingham dataset inferface
- Tree structure visualizasion module (ongoing)
- MIDI/numpy transfer toolkit (ongoing)

Extra resources:

- Comparsion with ON-LSTM model (ongoing)
- Demos for music structure transfer (ongoing)

License:

We follow the Apache Agreement license of the origin program. All code changes are indicated in the commit history.

## Quick Start

```
# Install Dependencies (using Conda as a virtual environment)
conda create -n diora python=3.6
source activate diora

## DIORA uses Pytorch (>= 1.0.0).
## Note: If you need GPU-support, make sure that you follow directions on pytorch.org
conda install pytorch torchvision -c pytorch

## AllenNLP is for context-insensitive ELMo embeddings.
pip install allennlp

## There are a few other libraries being used.
pip install tqdm

## Deactivate Conda when not being used.
source deactivate

# Clone Repo
cd ~/code && git clone git@github.com:iesl/diora.git

# Run Example
source activate diora
cd ~/code/diora/pytorch
export PYTHONPATH=$(pwd):$PYTHONPATH
python ...  # (Training/Parsing)
```

### Data

We do not directly provide Nottingham dataset in numpy format. 

You can visit this website and transfer MIDI file to numpy format use our toolkit.

https://github.com/jukedeck/nottingham-dataset


### Training

A simple setting to get started:

```
python diora/scripts/train.py \
    --batch_size 10 \
    --hidden_dim 50 \
    --log_every_batch 100 \
    --save_after 1000 \
    --train_filter_length 20 \
    --cuda
```

### Parsing

```
python diora/scripts/parse.py \
    --batch_size 10 \
    --data_type txt_id \
    --elmo_cache_dir ~/data/elmo \
    --embeddings_path ~/data/glove/glove.840B.300d.txt \
    --load_model_path ~/checkpoints/diora/experiment-001/model.step_300000.pt \
    --model_flags ~/checkpoints/diora/experiment-001/flags.json \
    --validation_path ./sample.txt \
    --validation_filter_length 10
```


## Easy Argument Assignment

Every experiment generates a `flags.json` file under its `experiment_path`. This file is useful when loading a checkpoint, as it specifies important properties for model configuration such as number-of-layers or model-size.

Note: Only arguments that are related to the model configuration will be used in this scenario.

Example Usage:

```
# First, train your model.
python diora/scripts/train.py \
    --experiment_path ~/log/experiment-01 \
    ... # other args

# Later, load the model checkpoint, and specify the flags file.
python diora/scripts/parse.py \
    --load_model_path ~/log/experiment-01/model_periodic.pt \
    --model_flags ~/log/experiment-01/flags.json \
    ... # other args
```

## Logging

Various logs, checkpoints, and useful files are saved to a "log" directory when running DIORA. By default, this directory will be at `/path/to/diora/pytorch/log/${data_type}-${date}-${timestamp}`. For example, this might be the log directory: `~/code/diora/pytorch/txt_id-20181117-1542478880`. You can specify your own directory using the `--experiment_path` flag.

Some files stored in the log directory are:

```
- experiment.log  # The output of the logger.
- flags.json  # All the arguments the experiment was run with as a JSON file.
- model_periodic.pt  # The latest model checkpoint, saved every N batches.
- model.step_X.pt  # Another checkpoint is saved every X batches.
```

## License

Copyright 2018, University of Massachusetts Amherst

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
