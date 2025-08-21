This project was built using the Gemini CLI—even this README.md file.

<details open>
<summary>한국어 설명 보기/숨기기</summary>

# LLM NPU 아티팩트 빌더

이 프로젝트는 특정 토큰을 금지하는 커스텀 로직이 적용된 대규모 언어 모델(LLM)을 양자화하고, NPU 아티팩트로 빌드하는 스크립트를 포함합니다.

## 사전 준비

빌드 과정을 실행하기 전에, 필요한 의존성 패키지들이 설치되어 있는지 확인해야 합니다. 주 요구사항은 `requirements.txt`에 명시되어 있으며, 환경에 따라 `furiosa_requirements.txt`의 패키지도 설치해야 할 수 있습니다.

이 프로젝트는 FuriosaAI의 Renegade 칩 2개를 사용하는 환경을 기준으로 합니다.

```bash
pip install -r requirements.txt
```

또한, 프로젝트의 루트 디렉터리에 `banned_tokens.txt` 파일이 필요합니다. 이 파일은 모델의 출력에서 제외할 토큰 ID들을 쉼표로 구분하여 포함해야 합니다.

## 빌드 과정

최종 NPU 아티팩트를 생성하는 과정은 두 가지 주요 단계로 이루어집니다.

### 1단계: 양자화된 모델 생성

이 스크립트는 베이스 모델(`meta-llama/Llama-3.3-70B-Instruct`)을 로드하고, SDK를 실시간으로 수정(몽키패칭)하여 커스텀 로직을 적용한 뒤, 모델을 양자화합니다.

#### 커스텀 로직 상세:
1.  **토큰 금지**: `banned_tokens.txt` 파일에 명시된 토큰 ID들의 로짓(logit)을 최소값으로 설정하여 해당 토큰이 생성되지 않도록 합니다.
2.  **토큰 증폭**: `amplified_tokens.txt` 파일에 명시된 토큰 ID들의 로짓에 특정 계수(factor)를 곱하여 해당 토큰의 생성 확률을 높입니다.

#### 스크립트 실행:
```bash
# 기본값(증폭 계수 5.0)으로 실행
python run_custom_quantization.py

# 증폭 계수를 10.0으로 지정하여 실행
python run_custom_quantization.py --amplification_factor 10.0
```

-   **입력**: 
    -   Hugging Face의 `meta-llama/Llama-3.3-70B-Instruct` 모델.
    -   `banned_tokens.txt`: 금지할 토큰 ID 목록 (쉼표로 구분).
    -   (선택 사항) `amplified_tokens.txt`: 증폭할 토큰 ID 목록 (쉼표로 구분).
-   **출력**: `./quantized_not_CJ_llama_for_npu` 디렉터리에 저장되는 양자화된 모델.

### 2단계: 최종 NPU 아티팩트 빌드

이 스크립트는 이전 단계에서 생성된 양자화 모델을 입력으로 받아, 배포 가능한 최종 NPU 아티팩트를 빌드합니다.

스크립트 실행:
```bash
python build_from_quantized.py
```

-   **입력**: `./quantized_not_CJ_llama_for_npu` 디렉터리의 내용물.
-   **출력**: `./npu_artifact_from_quantized` 디렉터리에 저장되는 서빙 가능한 최종 아티팩트.

이 두 단계를 완료하면, `./npu_artifact_from_quantized` 디렉터리에 배포에 필요한 모든 것이 준비됩니다.

</details>

<br>

<details>
<summary>Show/Hide English Description</summary>

# LLM NPU Artifact Builder

This project contains scripts to quantize a large language model (LLM) with custom logic and build it into an NPU artifact.

## Prerequisites

Before running the build process, ensure you have the necessary dependencies installed. The primary requirements are listed in `requirements.txt`. Depending on your setup, you may also need to install packages from `furiosa_requirements.txt`.

This project is based on an environment using two Renegade chips from FuriosaAI.

```bash
pip install -r requirements.txt
```

You also need to have a `banned_tokens.txt` file in the root directory. This file should contain a comma-separated list of token IDs to be excluded from the model's output.

## Build Process

The process to create the final NPU artifact involves two main steps.

### Step 1: Create the Quantized Model

This script loads the base model (`meta-llama/Llama-3.3-70B-Instruct`), applies custom logic by monkey-patching the SDK, and then quantizes the model.

#### Custom Logic Details:
1.  **Token Banning**: Sets the logits of token IDs specified in `banned_tokens.txt` to the minimum value to prevent them from being generated.
2.  **Token Amplification**: Multiplies the logits of token IDs specified in `amplified_tokens.txt` by a specific factor to increase their generation probability.

#### Execute the script:
```bash
# Run with default amplification factor (5.0)
python run_custom_quantization.py

# Run with a specific amplification factor of 10.0
python run_custom_quantization.py --amplification_factor 10.0
```

-   **Input**: 
    -   `meta-llama/Llama-3.3-70B-Instruct` model from Hugging Face.
    -   `banned_tokens.txt`: A comma-separated list of token IDs to ban.
    -   (Optional) `amplified_tokens.txt`: A comma-separated list of token IDs to amplify.
-   **Output**: A quantized model saved in the `./quantized_not_CJ_llama_for_npu` directory.

### Step 2: Build the Final NPU Artifact

This script takes the quantized model generated in the previous step and builds the final, deployable NPU artifact.

Execute the script:
```bash
python build_from_quantized.py
```

-   **Input**: The contents of the `./quantized_not_CJ_llama_for_npu` directory.
-   **Output**: The final artifact ready for serving, located in the `./npu_artifact_from_quantized` directory.

After completing these two steps, the `./npu_artifact_from_quantized` directory will contain everything needed for deployment.

</details>