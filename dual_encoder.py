import argparse
import json
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

import pathlib
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from utils import create_batch_dict, logger, move_to_cuda, pool

parser = argparse.ArgumentParser(description='evaluation for BEIR benchmark')
parser.add_argument('--model-name-or-path', default='intfloat/e5-small-v2',
                    type=str, metavar='N', help='which model to use')
parser.add_argument('--output-dir', default='tmp-outputs/',
                    type=str, metavar='N', help='output directory')
parser.add_argument('--pool-type', default='avg', help='pool type')
parser.add_argument('--prefix-type', default='query_or_passage', help='prefix type')
parser.add_argument('--dry-run', action='store_true', help='whether to run the script in dry run mode')

args = parser.parse_args()
args.pool_type='weightedavg'
args.prefix_type='instruction'
data_path='./dataset_curation'
# model_path="/data1/chh/models/yutaozhu94/INTERS-Falcon-1b"
# model_path="/data1/chh/models/yutaozhu94/INTERS-LLaMA-7b-chat"
# model_path="/data1/chh/models/yutaozhu94/INTERS-Mistral-7b"
model_path="/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct"
# model_path="/data1/chh/models/mistralai/Mistral-7B-Instruct-v0.1"


args.model_name_or_path=model_path

logger.info('Args: {}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
assert args.pool_type in ['cls', 'avg', 'last', 'weightedavg'], 'pool_type should be cls / avg / last'
assert args.prefix_type in ['query_or_passage', 'instruction'], 'prefix_type should be query_or_passage / instruction'
# os.makedirs(args.output_dir, exist_ok=True)


class RetrievalModel():
    def __init__(self, **kwargs):
        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)
        self.base_name: str = args.model_name_or_path.split('/')[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.encoder.resize_token_embeddings(len(self.tokenizer))
        self.prompt = None
        self.query_instruction=None
        self.document_instruction=None
        if args.prefix_type == 'instruction':
            self.set_instruction()
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)
        self.encoder.cuda()
        self.encoder.eval()

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        if args.prefix_type == 'query_or_passage':
            input_texts = [f'query: {q} ' for q in queries]
        else:
            # input_texts = [f'{q}' for q in queries]
            input_texts = [self.query_instruction.format(content=q) for q in queries]

        # print(input_texts)
        return self._do_encode(input_texts)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        # no need to add prefix for instruct models
        if args.prefix_type == 'query_or_passage':
            input_texts = ['passage: {}'.format(t) for t in input_texts]
        else:
            input_texts = [self.document_instruction.format(content=d) for d in input_texts]
            # input_texts = ['{}'.format(t) for t in input_texts]


        return self._do_encode(input_texts)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str]) -> np.ndarray:
        encoded_embeds = []
        batch_size = 8 * self.gpu_count
        for start_idx in tqdm.tqdm(range(0, len(input_texts), batch_size), desc='encoding', mininterval=10):
            batch_input_texts: List[str] = input_texts[start_idx: start_idx + batch_size]

            batch_dict = create_batch_dict(self.tokenizer, batch_input_texts)
            batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], args.pool_type)
                embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)

    def set_prompt(self, prompt: str):
        self.prompt = prompt
    def set_instruction(self):

        if 'INTERS' in self.base_name:
            self.query_instruction="Represent this sentence for searching relevant passages: {content}"
            self.document_instruction="Represent this passage for retrieval: {content}"
        elif 'Meta-Llama-3-8B-Instruct' in self.base_name:
            # self.query_instruction="Represent this sentence for searching relevant passages: {content}"
            # self.document_instruction="Represent this passage for retrieval: {content}"
            self.query_instruction="<|begin_of_text|> <|start_header_id|> user <|end_header_id|> \n\n Represent this sentence for searching relevant passages: {content} <|eot_id|> <|start_header_id|> assistant <|end_header_id|> \n\n"
            self.document_instruction="<|begin_of_text|> <|start_header_id|> user <|end_header_id|> \n\n Represent this passage for retrieval: {content} <|eot_id|> <|start_header_id|> assistant <|end_header_id|> \n\n"
        elif 'Mistral-7B-Instruct' in self.base_name:
            # self.query_instruction="Represent this sentence for searching relevant passages: {content}"
            # self.document_instruction="Represent this passage for retrieval: {content}"
            self.query_instruction="<s>[INST] Represent this sentence for searching relevant passages: {content} [/INST]"
            self.document_instruction="<s>[INST] Represent this passage for retrieval: {content} [/INST]"

def main():

    model = RetrievalModel()
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    model = DRES(model)
    retriever = EvaluateRetrieval(model, score_function="dot") # or "cos_sim" for cosine similarity
    results = retriever.retrieve(corpus, queries)

    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

   
    logger.info("ndcg: {}, map: {}, recall: {}, precision: {}".format(ndcg,_map,recall,precision))

    # top_k = 5
    # for query_id, ranking_scores in list(results.items()):
    #     scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    #     logger.info("Query : %s\n" % queries[query_id])

    #     for rank in range(top_k):
    #         doc_id = scores_sorted[rank][0]
    #         # Format: Rank x: ID [Title] Body
    #         logger.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))

        

if __name__ == '__main__':
    main()
