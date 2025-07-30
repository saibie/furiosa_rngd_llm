import os
from furiosa_llm.artifact import ArtifactBuilder

def main():
    # 양자화가 완료된 모델의 경로
    quantized_model_path = "./quantized_not_CJ_llama_for_npu"
    
    # 최종 아티팩트를 저장할 경로
    artifact_output_dir = "./npu_artifact_from_quantized"

    print(f"'{quantized_model_path}' 경로의 양자화된 모델을 기반으로 최종 아티팩트 빌드를 시작합니다.")

    # ArtifactBuilder를 사용하여 양자화된 모델로부터 아티팩트를 생성합니다.
    builder = ArtifactBuilder(
        model_id_or_path=quantized_model_path,
        name="final-llama3-model-from-quantized", # 아티팩트 이름을 더 명확하게 변경
        tensor_parallel_size=8,
        pipeline_parallel_size=2,
    )

    # 별도의 커스텀 로직 없이 빌드를 실행합니다.
    builder.build(save_dir=artifact_output_dir)

    print(f"\n🎉🎉🎉 최종 아티팩트 빌드 성공! 결과가 '{artifact_output_dir}'에 저장되었습니다. 🎉🎉🎉")

if __name__ == '__main__':
    main()