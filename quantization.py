from furiosa_llm.optimum.dataset_utils import create_data_loader
from furiosa_llm.optimum import QuantizerForCausalLM, QuantizationConfig, QDtype

model_id = "meta-llama/Llama-3.3-70B-Instruct"
quantized_model = "./quantized-Llama-3.3"

dataloader = create_data_loader(
    tokenizer=model_id,
    dataset_name_or_path="mit-han-lab/pile-val-backup",
    dataset_split="validation",
    num_samples=5, # Increase this number for better calibration
    max_sample_length=1024,
)

quantizer = QuantizerForCausalLM.from_pretrained(model_id, device_map='auto', trust_remote_code=True)

quantizer.quantize(quantized_model, dataloader, QuantizationConfig(weight=QDtype.INT4, activation=QDtype.INT4, kv_cache=QDtype.INT4))
