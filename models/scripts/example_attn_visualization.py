from typing import Optional

import fire

from llama_models.llama3.reference_impl.generation import Llama
from tqdm import tqdm
from termcolor import cprint
import random
import torch
from nltk.corpus import words

def generate_random_dict(num_pairs=1, seed=None):
    """
    Generate a dictionary with random words as keys and values using the NLTK words corpus.

    Args:
        num_pairs (int): Number of key-value pairs to generate.
        seed (int, optional): Seed for the random number generator.

    Returns:
        dict: A dictionary with random words as keys and values.
    """
    if seed is not None:
        random.seed(seed)  # Set the seed for deterministic results

    word_list = words.words()
    
    return {
        random.choice(word_list): random.choice(word_list)
        for _ in range(num_pairs)
    }

def keep_first_k_sorted_keys(input_dict, k):
    """
    Retain only the first k keys from the dictionary in sorted order.

    Args:
        input_dict (dict): The original dictionary.
        k (int): The number of keys to keep.

    Returns:
        dict: A new dictionary with the first k keys in sorted order.
    """
    sorted_keys = sorted(input_dict.keys())
    # Select the first k keys and construct a new dictionary
    return {key: input_dict[key] for key in sorted_keys[:k]}

def select_random_key(input_dict):
    """
    Select a random key from the dictionary.

    Args:
        input_dict (dict): The dictionary to select a key from.

    Returns:
        The randomly selected key.
    """
    if not input_dict:
        raise ValueError("The dictionary is empty.")
    return random.choice(list(input_dict.keys()))

def run_main(
    ckpt_dir: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: int = 512,
    model_parallel_size: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size,
    )
    random_dict = generate_random_dict(
      num_pairs=16384,
      seed=42
    )

    num_pairs_to_consider = [16, 32, 64, 128, 256]

    attns_per_sequence_length = {}
    for num_pairs in tqdm(num_pairs_to_consider):
      trunc_dict = keep_first_k_sorted_keys(random_dict, num_pairs)
      target_key = select_random_key(trunc_dict)

      prompt = f"""
Your task is to read a list of key value pairs, provided in the format "key->value", separated by commas, and then determine what value corresponds to the target key.

Key value pairs: {",".join([f'{k}->{v}' for (k, v) in trunc_dict.items()])}

Target key: {target_key}

Your answer needs to be a single word - the value corresponding to the target key! Please do not use any programming language or step-by-step-reasoning.
      """

      expected_token_length = len(
          generator.tokenizer.encode(target_key, bos=True, eos=True)
      )

      print(f"expected length: {expected_token_length}")
      
      result = generator.text_completion(
          prompt,
          temperature=temperature,
          top_p=top_p,
          max_gen_len=expected_token_length,
          logprobs=False,
          return_attn=True
      )

      attns_per_sequence_length[num_pairs] = result.attns[0] # first generated token

      # print(result.attns)
      cprint(f"{prompt}", end="")
      cprint(f"{result.generation}", color="yellow")
      print("\n==================================\n")
    
    torch.save(attns_per_sequence_length, "attns_per_sequence_length.pth")


def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()