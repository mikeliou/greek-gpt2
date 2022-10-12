import jax
import optax
import flax
import jax.numpy as jnp
import math
import numpy as np

from datasets import load_dataset, concatenate_datasets, DatasetDict
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard
from tokenizers import trainers, Tokenizer, normalizers, ByteLevelBPETokenizer
from transformers import AutoConfig, AutoTokenizer, FlaxAutoModelForCausalLM
from tqdm import tqdm
from pathlib import Path
# from utilities import batch_iterator, tokenize_function, group_texts


def batch_iterator(batch_size=1000):
    for i in range(0, len(raw_dataset), batch_size):
        yield raw_dataset["train"][i: i + batch_size]["text"]


def tokenize_function(examples):
    return tokenizer(examples["text"])


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // max_seq_length) * max_seq_length
    result = {
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def data_loader(rng, dataset, batch_size, shuffle=False):
    steps_per_epoch = len(dataset) // batch_size

    if shuffle:
        batch_idx = jax.random.permutation(rng, len(dataset))
    else:
        batch_idx = jnp.arange(len(dataset))

    batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: jnp.array(v) for k, v in batch.items()}

        batch = shard(batch)

        yield batch


def train_step(state, batch, dropout_rng):
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_fn(params):
        print(batch)
        labels = batch.pop("labels")
        logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]

        loss = optax.softmax_cross_entropy(logits[..., :-1, :], onehot(labels[..., 1:], logits.shape[-1])).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    grad = jax.lax.pmean(grad, "batch")
    new_state = state.apply_gradients(grads=grad)

    metrics = jax.lax.pmean(
        {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}, axis_name="batch"
    )

    return new_state, metrics, new_dropout_rng


def eval_step(params, batch):
    labels = batch.pop("labels")

    logits = model(**batch, params=params, train=False)[0]

    loss = optax.softmax_cross_entropy(logits[..., :-1, :], onehot(labels[..., 1:], logits.shape[-1])).mean()

    # summarize metrics
    metrics = {"loss": loss, "perplexity": jnp.exp(loss)}
    metrics = jax.lax.pmean(metrics, axis_name="batch")
    return metrics


language = "el"
model_config = "distilgpt2"

max_seq_length = 512  # 512  # 1024

per_device_batch_size = 64  # 16  # 64
num_epochs = 10
training_seed = 0
learning_rate = 3e-4

root_dir = "//mnt//disks//persist"
model_dir = model_config + f"-pretrained-{language}-v2"
Path(model_dir).mkdir(parents=True, exist_ok=True)

config = AutoConfig.from_pretrained(model_config)
config.save_pretrained(f"{model_dir}")

# train tokenizer
oscar_dataset = load_dataset("oscar", f"unshuffled_deduplicated_{language}", cache_dir=root_dir)
grcorpus_dataset = load_dataset("text", data_files={'train': ['../grcorpus.txt']},
                                                cache_dir=root_dir)
wiki_dataset = load_dataset("text", data_files={'train': ['../wiki_el.txt']},
                                            cache_dir=root_dir)
raw_dataset = DatasetDict({'train': concatenate_datasets([oscar_dataset['train'], grcorpus_dataset['train'], wiki_dataset['train']])})

tokenizer = ByteLevelBPETokenizer()
tokenizer.train_from_iterator(batch_iterator(), vocab_size=config.vocab_size, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save(f"{model_dir}/tokenizer.json")

oscar_dataset["train"] = load_dataset("oscar", f"unshuffled_deduplicated_{language}", split="train[5%:]", cache_dir=root_dir)
oscar_dataset["validation"] = load_dataset("oscar", f"unshuffled_deduplicated_{language}", split="train[:5%]", cache_dir=root_dir)
print(f"oscar_dataset: train: {len(oscar_dataset['train'])}, validation: {len(oscar_dataset['validation'])}")

grcorpus_dataset["train"] = load_dataset("text", data_files={'train': ['../grcorpus.txt']}, split="train[5%:]", cache_dir=root_dir)
grcorpus_dataset["validation"] = load_dataset("text", data_files={'train': ['../grcorpus.txt']}, split="train[:5%]", cache_dir=root_dir)
print(f"grcorpus_dataset: train: {len(grcorpus_dataset['train'])}, validation: {len(grcorpus_dataset['validation'])}")

wiki_dataset["train"] = load_dataset("text", data_files={'train': ['../wiki_el.txt']}, split="train[5%:]", cache_dir=root_dir)
wiki_dataset["validation"] = load_dataset("text", data_files={'train': ['../wiki_el.txt']},  split="train[:5%]",cache_dir=root_dir)
print(f"wiki_dataset: train: {len(wiki_dataset['train'])}, validation: {len(wiki_dataset['validation'])}")

raw_dataset["train"] = concatenate_datasets([oscar_dataset['train'], grcorpus_dataset['train'], wiki_dataset['train']])
raw_dataset["validation"] = concatenate_datasets([oscar_dataset['validation'], grcorpus_dataset['validation'], wiki_dataset['validation']])
print(f"raw_dataset: train: {len(raw_dataset['train'])}, validation: {len(raw_dataset['validation'])}")

# these cells should be commented out to run on full dataset
# raw_dataset["train"] = raw_dataset["train"].select(range(20000))
# raw_dataset["validation"] = raw_dataset["validation"].select(range(2000))

tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}")
tokenized_datasets = raw_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=raw_dataset["train"].column_names)
# tokenized_datasets = raw_dataset.map(tokenize_function, batched=True, remove_columns=raw_dataset["train"].column_names)
tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=4)
# tokenized_datasets = tokenized_datasets.map(group_texts, batched=True)

total_batch_size = per_device_batch_size * jax.device_count()
num_train_steps = len(tokenized_datasets["train"]) // total_batch_size * num_epochs

model = FlaxAutoModelForCausalLM.from_config(config, seed=training_seed, dtype=jnp.dtype("bfloat16"))
linear_decay_lr_schedule_fn = optax.linear_schedule(init_value=learning_rate, end_value=0, transition_steps=num_train_steps)
adamw = optax.adamw(learning_rate=linear_decay_lr_schedule_fn, b1=0.9, b2=0.98, eps=1e-8, weight_decay=0.01)

state = train_state.TrainState.create(apply_fn=model.__call__, params=model.params, tx=adamw)
parallel_train_step = jax.pmap(train_step, "batch")
parallel_eval_step = jax.pmap(eval_step, "batch")
state = flax.jax_utils.replicate(state)
rng = jax.random.PRNGKey(training_seed)
dropout_rngs = jax.random.split(rng, jax.local_device_count())

for epoch in tqdm(range(1, num_epochs + 1), desc=f"Epoch ...", position=0, leave=True):
    rng, input_rng = jax.random.split(rng)

    # -- Train --
    train_loader = data_loader(input_rng, tokenized_datasets["train"], total_batch_size, shuffle=True)
    with tqdm(total=len(tokenized_datasets["train"]) // total_batch_size, desc="Training...",
              leave=False) as progress_bar_train:
        for model_inputs in train_loader:
            # Model forward
            state, train_metric, dropout_rngs = parallel_train_step(state, model_inputs, dropout_rngs)

            progress_bar_train.update(1)

        progress_bar_train.write(
            f"Train... ({epoch}/{num_epochs} | Loss: {round(train_metric['loss'].mean(), 3)}, Learning Rate: {round(train_metric['learning_rate'].mean(), 6)})"
        )

    # -- Eval --
    eval_loader = data_loader(input_rng, tokenized_datasets["validation"], total_batch_size)
    eval_metrics = []

    with tqdm(total=len(tokenized_datasets["validation"]) // total_batch_size, desc="Evaluation...",
              leave=False) as progress_bar_eval:
        for model_inputs in eval_loader:
            # Model forward
            eval_metric = parallel_eval_step(state.params, model_inputs)
            eval_metrics.append(eval_metric)

            progress_bar_eval.update(1)

        eval_metrics = get_metrics(eval_metrics)
        eval_metrics = jax.tree_map(jnp.mean, eval_metrics)
        progress_bar_eval.write(
            f"Eval... ({epoch}/{num_epochs} | Loss: {eval_metrics['loss']} | Perplexity: {eval_metrics['perplexity']})"
        )

model.push_to_hub("gpt-oscar_grcorpus_wiki-seq512-ep10-bs64", use_temp_dir=True)
tokenizer.push_to_hub("gpt-oscar_grcorpus_wiki-seq512-ep10-bs64", use_temp_dir=True)
