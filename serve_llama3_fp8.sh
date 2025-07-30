#!/bin/bash
furiosa-llm serve ./Llama-3.3-70B-Instruct-FP8 --devices "npu:0, npu:1" -tp 8 -pp 2 -dp 1 --port 61786 --enable-auto-tool-choice --tool-call-parser llama3_json