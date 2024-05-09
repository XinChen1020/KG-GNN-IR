import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode
import pandas as pd
import json
import argparse
from functools import partial
import nest_asyncio


from llama_index.core.evaluation import RetrieverEvaluator


metrics = ["mrr", "hit_rate"]
nest_asyncio.apply()

def display_results(name, eval_results):
    """Display results from evaluate."""

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result['metric_vals_dict']
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()
    columns = {"retrievers": [name], "hit_rate": [hit_rate], "mrr": [mrr]}

    metric_df = pd.DataFrame(columns)

    return metric_df

def update_hit_rates(results):
    """
    Recalculates and updates the hit rate for each result entry based on the expected and retrieved IDs.

    Args:
    - results (list of dicts): A list containing result dictionaries. Each dictionary must have:
        'expected_ids' (list of str): The list of expected IDs.
        'retrieved_ids' (list of str): The list of IDs that were retrieved.
        'metric_vals_dict' (dict): Dictionary where hit rate will be updated.

    Returns:
    - None: The function modifies the list in place.
    """

    for r in results:
        # Ensure expected_ids and retrieved_ids are provided
        if 'expected_ids' in r and 'retrieved_ids' in r:
            expected_ids = set(r['expected_ids'])
            retrieved_ids = set(r['retrieved_ids'])

            # Calculate hit rate as the number of expected items retrieved divided by total expected items
            hit_count = len(expected_ids & retrieved_ids)
            total_expected = len(expected_ids)
            hit_rate = hit_count / total_expected if total_expected > 0 else 0

            # Update the metric_vals_dict for the hit rate
            r['metric_vals_dict']['hit_rate'] = hit_rate
        else:
            print("Error: Missing expected_ids or retrieved_ids in some entries.")
    return results

def base_line_evaluation(output, top_k, retriever_mode, embed_model):
    
    base_nodes = []

    expected_id = [fact[0] + "_" + str(fact[1]) for fact in output['supporting_facts']]
    for paragraph in output['context']:
        for sentence in paragraph[1]:
            id = 0
            if len(sentence.strip()) != 0:
                base_nodes.append(TextNode(text = sentence.strip(), id_= paragraph[0] + "_" + str(id)))
            else:
                base_nodes.append(TextNode(text = " ", id_= paragraph[0] + "_" + str(id)))
            id += 1

    if retriever_mode == 'embedding':
        base_index = VectorStoreIndex(base_nodes, embed_model = embed_model)
        base_retriever = base_index.as_retriever(similarity_top_k=top_k)
    else:
        base_retriever = BM25Retriever.from_defaults(nodes=base_nodes, similarity_top_k=top_k)
    
    
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        metrics, retriever=base_retriever
    )

    eval_result = retriever_evaluator.evaluate(output['question'], expected_id)
    
    return eval_result



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('--model_name', type=str, default="BAAI/bge-small-en-v1.5", help='Model name')
    parser.add_argument('--file_name', type=str, default="hotpot_dev_distractor_v1.json", help='File name')
    parser.add_argument('--top_k', type=int, default= 5, help='Top K')
    parser.add_argument('--retriever_mode', type=str, default="bm25", help='Retriever mode')
    args = parser.parse_args()

    results = []

    if args.retriever_mode == 'embedding':
        embed_model = HuggingFaceEmbedding(model_name=args.model_name)
    else:
        embed_model = None

    with open(args.file_name, encoding = "utf-8") as f:
        outputs = json.load(f)[:100]

    with tqdm.tqdm(total=len(outputs)) as pbar:
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(partial(base_line_evaluation, top_k=args.top_k, retriever_mode=args.retriever_mode, embed_model = embed_model), output) for output in outputs]
            for future in as_completed(futures):
                results.append(future.result())
                pbar.update(1)
    data = []
    for r in results:
      data.append({
              "query": r.query,
              "expected_ids": r.expected_ids,
              "retrieved_ids": r.retrieved_ids,
              "metric_vals_dict": r.metric_vals_dict
          })
    data = update_hit_rates(data)
    print(display_results(args.retriever_mode, data))