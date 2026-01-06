from typing import List, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import tqdm

def batch_inference(program, args_list, use_process=False, max_workers=32) -> List[Any]:
    futures = {}
    results = [None] * len(args_list)

    executor_class = ProcessPoolExecutor if use_process else ThreadPoolExecutor
    
    with executor_class(max_workers=max_workers) as executor:
        for i, args in enumerate(args_list):
            future = executor.submit(
                program,
                **args
            )
            futures[future] = i

        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            index = futures[future]
            results[index] = result
    return results