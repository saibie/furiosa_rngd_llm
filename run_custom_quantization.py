import os
import torch
from typing import Optional, List
import shutil

# --- 1. SDK 및 커스텀 로직에 필요한 모든 클래스 임포트 ---
try:
    from furiosa_llm.optimum import QuantizerForCausalLM, QuantizationConfig
    from furiosa_llm.optimum.dataset_utils import create_data_loader
    from furiosa_llm_models.llama3.symbolic.mlperf_submission import LlamaForCausalLM as FuriosaLlamaForCausalLM
except ImportError:
    print("오류: furiosa-llm SDK가 설치되어 있지 않거나, 필요한 클래스를 찾을 수 없습니다.")
    exit()


# --- 2. 금지할 토큰 ID 로드 ---
banned_tokens_file = "banned_tokens.txt"
with open(banned_tokens_file, 'r', encoding='utf-8') as f:
    banned_tokens_str = f.read().replace('[', '').replace(']', '')
    banned_token_ids = [int(x.strip()) for x in banned_tokens_str.split(',')]

print(f"총 {len(banned_token_ids)}개의 토큰을 금지하도록 설정을 준비합니다.")


# --- 3. ⭐️⭐️⭐️ SDK 클래스를 실시간으로 수정 (몽키 패칭) ⭐️⭐️⭐️ ---
# Quantizer가 모델을 로드하기 전에, SDK의 원본 클래스 자체를 수정합니다.

# 3-1. SDK의 원본 forward 메서드를 백업해 둡니다.
original_forward = FuriosaLlamaForCausalLM.forward
print("SDK의 원본 LlamaForCausalLM.forward 메서드를 백업했습니다.")

# 3-2. 우리가 주입할 새로운 forward 함수를 정의합니다.
#      시그니처는 원본 파일과 완벽하게 동일해야 합니다.
def patched_forward(
    self,
    input_ids: torch.LongTensor = None, attention_mask: Optional[torch.Tensor] = None,
    causal_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None, inputs_embeds: Optional[torch.FloatTensor] = None,
    new_key_location: Optional[torch.IntTensor] = None, new_value_location: Optional[torch.IntTensor] = None,
    past_valid_key_indices: Optional[torch.IntTensor] = None, past_valid_value_indices: Optional[torch.IntTensor] = None,
    is_prefill: bool = True, bucket_size: int = 2048, labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None, **kwargs
):
    # 백업해 둔 원본 forward 메서드를 먼저 호출하여 SDK의 모든 로직을 수행합니다.
    outputs = original_forward(
        self, input_ids=input_ids, attention_mask=attention_mask, causal_mask=causal_mask,
        position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds,
        new_key_location=new_key_location, new_value_location=new_value_location,
        past_valid_key_indices=past_valid_key_indices, past_valid_value_indices=past_valid_value_indices,
        is_prefill=is_prefill, bucket_size=bucket_size, labels=labels, use_cache=use_cache,
        output_attentions=output_attentions, output_hidden_states=output_hidden_states,
        return_dict=return_dict, **kwargs
    )
    
    # 원본 메서드가 끝난 후, 우리의 커스텀 로직을 결과에 추가합니다.
    logits = outputs.logits
    if banned_token_ids:
        logits[:, :, banned_token_ids] = torch.finfo(logits.dtype).min
    
    # 수정된 결과를 담아 반환합니다.
    outputs.logits = logits
    return outputs

# 3-3. SDK의 Llama 클래스의 forward 메서드를 우리가 만든 함수로 교체합니다.
FuriosaLlamaForCausalLM.forward = patched_forward
print("SDK의 LlamaForCausalLM 클래스에 커스텀 로직을 성공적으로 주입(몽키 패칭)했습니다.")


# --- 4. 메인 양자화 실행 로직 ---
def main():
    # 이제 순정 모델 ID를 그대로 사용합니다.
    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    
    # 양자화된 모델을 저장할 경로
    quantized_model_output_dir = "./quantized_not_CJ_llama_for_npu"

    # ⭐️ 캐시를 지워 이전의 실패한 시도의 영향을 완전히 제거합니다.
    cache_dir = os.path.expanduser("~/.cache/furiosa/llm")
    if os.path.exists(cache_dir):
        print(f"'{cache_dir}' 캐시를 삭제합니다...")
        shutil.rmtree(cache_dir)

    print(f"\n'{model_id}' 모델을 로드하여 양자화를 시작합니다...")

    dataloader = create_data_loader(
        tokenizer=model_id,
        dataset_name_or_path="mit-han-lab/pile-val-backup",
        dataset_split="validation",
        num_samples=5,
        max_sample_length=2048, # SDK 기본 버킷 크기에 맞춤
    )

    quantizer = QuantizerForCausalLM.from_pretrained(model_id)
    
    quantizer.quantize(
        quantized_model_output_dir,
        dataloader,
        QuantizationConfig.w_f8_a_f8_kv_f8()
    )

    print(f"\n🎉 양자화 성공! 결과가 '{quantized_model_output_dir}'에 저장되었습니다.")
    print("이제 이 경로를 사용하여 ArtifactBuilder로 최종 아티팩트를 빌드할 수 있습니다.")


if __name__ == '__main__':
    main()
