# KVSort: Drastically Improving LLM Inference Performacne via KV Cache COmpression

## Prerequisite
### FZ-GPU
```
https://github.com/szcompressor/FZ-GPU
```

### SZ3
```
https://github.com/szcompressor/SZ3
```

### lm-eval harness
```
https://github.com/EleutherAI/lm-evaluation-harness
```


### Obtain Key and Value tensor from transformer package and dump as FP32 numpy array
#### Replace https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py with code/huggingface.py. Change the directory to save data, search "Baixi"to locate in my huggingface.py

## Running lm_eval to obtain key and value cache
```
cd code && bash evaluate_llama3_gsm8k.sh
```

### Note that only generate_until tasks will use KV cache.
For example
```
https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml#L6C14-L6C28
```

## Running demo
```
cd code && bash run.sh
```

## Shared index
```
python3 key_value_shared_idx.py
```

## Explore the CR to ratio of sorted data
```
python3 CR_to_sort_ratio.py
```

# Important:
## The code may require path modification before you can run them, each path is bracked in '[' and ']'.

## Paper link
```
https://sc24.supercomputing.org/proceedings/poster/poster_files/post189s2-file3.pdf
```
## Citation
```
@article{sunkvsort,
  title={KVSort: Drastically Improving LLM Inference Performance via KV Cache Compression},
  author={Sun, Baixi and Yu, Xiaodong and Tao, Dingwen}
}
```

# Acknowledgement
## The material was supported by IU Bigred200. This work was also supported by the National Science Foundation (Grant Nos. 2312673, 2247080, 2303064, 2326494, and 2326495). Dingwen Tao contributed to this work while he was at Indiana University.