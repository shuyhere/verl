start eva api endpoint:
`cd SWE-bench/swebench/harness`
`uvicorn eval_api:app --host 0.0.0.0 --port 8000`

upload your perdiction:
```
curl -X POST "http://10.68.171.9:8000/evaluate" \
  -F "predictions=@/ibex/project/c2328/verl_singularity/ExpeRepair/ExpeRepair-v1.0/results/preds_example/final_0.jsonl" \
  -F "run_id=test_run" \
  -F "dataset_name=SWE-bench/SWE-bench_Lite" \
  -F "split=test" \
  -F "max_workers=4" \
  -F "timeout=1800"
```

```
request use python:
import requests

url = "http:/<api>:8000/evaluate"
files = {"predictions": open("predictions.json", "rb")}
data = {
    "run_id": "test_run",
    "dataset_name": "SWE-bench/SWE-bench_Lite",
    "split": "test",
    "max_workers": 4,
    "timeout": 1800
}
response = requests.post(url, files=files, data=data)
print(response.json())


```