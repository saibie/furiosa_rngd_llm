
import argparse
from transformers import AutoTokenizer
from typing import Union, List
import os

def target_token_extractor(
    tokenizer,
    start_unicodes: Union[int, List[int]],
    end_unicodes: Union[int, List[int]],
):
    """
    Extracts tokens from a tokenizer's vocabulary that correspond to characters
    within the given Unicode ranges.
    """
    if isinstance(start_unicodes, int):
        start_unicodes = [start_unicodes]
    if isinstance(end_unicodes, int):
        end_unicodes = [end_unicodes]

    detected_tokens = set()

    print("Scanning vocabulary by decoding tokens...")
    for token_id in range(tokenizer.vocab_size):
        try:
            decoded = tokenizer.decode([token_id])
            for char in decoded:
                for s_unicode, e_unicode in zip(start_unicodes, end_unicodes):
                    if s_unicode <= ord(char) <= e_unicode:
                        detected_tokens.add(token_id)
                        break  # Move to the next character
        except Exception as e:
            # Some token IDs might not be decodable on their own.
            # print(f"Warning: Could not decode token {token_id}: {e}")
            pass

    print("Scanning vocabulary by encoding characters...")
    for s_unicode, e_unicode in zip(start_unicodes, end_unicodes):
        for i in range(s_unicode, e_unicode + 1):
            char = chr(i)
            try:
                encoded_tokens = tokenizer.encode(char, add_special_tokens=False)
                for token_id in encoded_tokens:
                    detected_tokens.add(token_id)
            except Exception as e:
                # print(f"Warning: Could not encode character {char}: {e}")
                pass
    
    return sorted(list(detected_tokens))

def main():
    parser = argparse.ArgumentParser(
        description="Extract language-specific tokens from a tokenizer and create banned/amplified token lists."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Path to the Hugging Face model or tokenizer.",
    )
    parser.add_argument(
        "--banned_output_file",
        type=str,
        default="banned_tokens.txt",
        help="Output file for banned tokens (Chinese/Japanese minus Korean).",
    )
    parser.add_argument(
        "--amplified_output_file",
        type=str,
        default="amplified_tokens.txt",
        help="Output file for amplified tokens (Korean minus Chinese/Japanese).",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer from '{args.model_path}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Unicode ranges:
    # CJK Unified Ideographs (Chinese): 0x4E00–0x9FFF
    # Hiragana & Katakana (Japanese): 0x3040–0x30FF
    # Hangul Syllables (Korean): 0xAC00–0xD7A3
    
    print("Extracting Chinese and Japanese tokens...")
    cj_tokens = target_token_extractor(
        tokenizer,
        [0x4E00, 0x3040],  # Start unicodes for Chinese and Japanese
        [0x9FFF, 0x30FF],  # End unicodes for Chinese and Japanese
    )
    print(f"Found {len(cj_tokens)} Chinese/Japanese tokens.")

    print("Extracting Korean tokens...")
    k_tokens = target_token_extractor(
        tokenizer,
        0xAC00,  # Start unicode for Korean
        0xD7A3,  # End unicode for Korean
    )
    print(f"Found {len(k_tokens)} Korean tokens.")

    # Create sets for efficient operation
    cj_token_set = set(cj_tokens)
    k_token_set = set(k_tokens)

    # 1. Banned tokens: Chinese/Japanese tokens NOT in Korean set
    cj_minus_k_tokens = sorted(list(cj_token_set - k_token_set))
    print(f"Found {len(cj_minus_k_tokens)} tokens that are Chinese/Japanese but not Korean.")

    banned_output_path = os.path.join(os.getcwd(), args.banned_output_file)
    print(f"Saving banned token list to '{banned_output_path}'...")
    with open(banned_output_path, 'w', encoding='utf-8') as f:
        f.write(str(cj_minus_k_tokens))

    # 2. Amplified tokens: Korean tokens NOT in Chinese/Japanese set
    k_minus_cj_tokens = sorted(list(k_token_set - cj_token_set))
    print(f"Found {len(k_minus_cj_tokens)} tokens that are Korean but not Chinese/Japanese.")
    
    amplified_output_path = os.path.join(os.getcwd(), args.amplified_output_file)
    print(f"Saving amplified token list to '{amplified_output_path}'...")
    with open(amplified_output_path, 'w', encoding='utf-8') as f:
        f.write(str(k_minus_cj_tokens))

    print("Done.")

if __name__ == "__main__":
    main()
