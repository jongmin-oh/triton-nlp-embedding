from transformers import AutoTokenizer
import numpy as np
import onnxruntime as rt

from utils import mean_pooling

session = rt.InferenceSession(
    "model_repository/embedding/1/model.onnx",
    providers=["CPUExecutionProvider"],
)

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")


def sentence_embedding(sentence: str, normalize_embeddings=False):
    # user turn sequence to query embedding
    model_inputs = tokenizer(sentence, return_tensors="pt")
    inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}
    sequence = session.run(None, inputs_onnx)
    print(sequence.shape)
    question_embedding = mean_pooling(sequence, inputs_onnx["attention_mask"])[0][0]

    if normalize_embeddings:
        question_embedding = question_embedding / np.linalg.norm(question_embedding)

    return question_embedding.numpy()


if __name__ == "__main__":
    embedding = sentence_embedding("안녕하세요")
    print(embedding.shape)
