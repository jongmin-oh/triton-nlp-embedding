from typing import Final
from tritonclient.http import InferenceServerClient
from tritonclient.http import InferInput

import numpy as np
from transformers import AutoTokenizer

from utils import mean_pooling

MAX_LENGTH: Final[int] = 128


class SentenceEmbedding:
    def __init__(self):
        self.client = None
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
        self.url = "localhost:8000"

    def __enter__(self):
        self.client = InferenceServerClient(url=self.url)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    def _request(self, sentence: str) -> np.ndarray:
        bert_inputs = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="np",
        )
        # 입력 데이터 생성
        input_data = [
            InferInput("input_ids", bert_inputs["input_ids"].shape, "INT64"),
            InferInput("attention_mask", bert_inputs["attention_mask"].shape, "INT64"),
            InferInput("token_type_ids", bert_inputs["token_type_ids"].shape, "INT64"),
        ]

        # 입력 데이터에 값을 설정
        for i, name in enumerate(["input_ids", "attention_mask", "token_type_ids"]):
            input_data[i].set_data_from_numpy(bert_inputs[name])

        # 추론 요청 보내기
        output = self.client.infer(
            model_name="embedding",
            inputs=input_data,
            headers={"Content-Type": "application/json"},
        )
        question_embedding = mean_pooling(
            [output.as_numpy("output_0")], bert_inputs["attention_mask"]
        )[0][0]
        return question_embedding

    def encode(self, sentence: str):
        logit = self._request(sentence)
        return logit


if __name__ == "__main__":
    with SentenceEmbedding() as embedder:
        print(embedder.encode("안녕하세요")[:5])
