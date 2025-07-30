from furiosa_llm.optimum import QuantizerForCausalLM, QuantizationConfig
from furiosa_llm.optimum.dataset_utils import create_data_loader

def main():
    # 순정 모델 ID를 사용합니다.
    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    # 양자화된 모델을 저장할 경로
    quantized_model_output_dir = "./quantized_original_llama"

    print(f"'{model_id}' 순정 모델의 양자화를 시작합니다...")

    dataloader = create_data_loader(
        tokenizer=model_id,
        dataset_name_or_path="mit-han-lab/pile-val-backup",
        dataset_split="validation",
        num_samples=5,
        max_sample_length=2048,
    )

    quantizer = QuantizerForCausalLM.from_pretrained(model_id)
    
    quantizer.quantize(
        quantized_model_output_dir,
        dataloader,
        QuantizationConfig.w_f8_a_f8_kv_f8()
    )

    print(f"\n🎉 순정 모델 양자화 성공! 결과가 '{quantized_model_output_dir}'에 저장되었습니다.")

if __name__ == '__main__':
    main()
