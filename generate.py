import json
import logging
import os
from itertools import chain
from dataclasses import dataclass
from datasets import load_dataset
from lmqg import TransformersQG
from lmqg.spacy_module import SpacyPipeline
from lmqg.automatic_evaluation import LANG_NEED_TOKENIZATION

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

model = './checkpoint/epoch_2'
path = 'shnl/test'


@dataclass
class eval_QAG:
    model: str = model
    model_ae: str = None
    max_length: int = 512
    max_length_output: int = 256
    dataset_path: str = path
    dataset_name: str = 'default'
    test_split: str = 'test'
    validation_split: str = 'validation'
    n_beams: int = 8
    batch_size: int = 4
    language: str = 'vi'
    export_dir: str = './output/eval_QAG'
    hyp_test: str = None
    hyp_dev: str = None
    overwrite_prediction: bool = True
    overwrite_metric: bool = True
    is_qg: bool = None
    is_ae: bool = None
    is_qag: bool = True
    use_reference_answer: bool = False


def main():
    opt = eval_QAG()
    os.makedirs(opt.export_dir, exist_ok=True)

    def load_model():
        if opt.model is not None:
            _model = TransformersQG(opt.model,
                                    is_ae=None if opt.is_ae else True,
                                    is_qg=None if opt.is_qg else True,
                                    is_qag=None if opt.is_qag else True,
                                    model_ae=opt.model_ae,
                                    skip_overflow_error=True,
                                    drop_answer_error_text=True,
                                    language=opt.language,
                                    max_length=opt.max_length,
                                    max_length_output=opt.max_length_output)
            _model.eval()
            return _model
        raise ValueError(f"require `-m` or `--model`")

    if opt.model_ae is not None:
        metric_file = f"{opt.export_dir}/metric.first.answer.paragraph.questions_answers." \
                      f"{opt.dataset_path.replace('/', '_')}.{opt.dataset_name}." \
                      f"{opt.model_ae.replace('/', '_')}.json"
    else:
        metric_file = f"{opt.export_dir}/metric.first.answer.paragraph.questions_answers." \
                      f"{opt.dataset_path.replace('/', '_')}.{opt.dataset_name}.json"
    if os.path.exists(metric_file):
        with open(metric_file) as f:
            output = json.load(f)
    else:
        output = {}
    spacy_model = SpacyPipeline(language=opt.language) if opt.language in LANG_NEED_TOKENIZATION else None
    for _split, _file in zip([opt.test_split, opt.validation_split], [opt.hyp_test, opt.hyp_dev]):
        if _file is None:
            if opt.model_ae is not None:
                _file = f"{opt.export_dir}/samples.{_split}.hyp.paragraph.questions_answers." \
                        f"{opt.dataset_path.replace('/', '_')}.{opt.dataset_name}." \
                        f"{opt.model_ae.replace('/', '_')}.txt"
            else:
                _file = f"{opt.export_dir}/samples.{_split}.hyp.paragraph.questions_answers." \
                        f"{opt.dataset_path.replace('/', '_')}.{opt.dataset_name}.txt"

        logging.info(f'generate qa for split {_split}')
        if _split not in output:
            output[_split] = {}

        dataset = load_dataset(opt.dataset_path, None if opt.dataset_name == 'default' else opt.dataset_name,
                               split=_split, use_auth_token=True)
        df = dataset.to_pandas()
        # formatting data into qag format
        model_input = []
        gold_reference = []
        model_highlight = []
        for paragraph, g in df.groupby("paragraph"):
            model_input.append(paragraph)
            model_highlight.append(g['answer'].tolist())
            gold_reference.append(' | '.join([
                f"question: {i['question']}, answer: {i['answer']}" for _, i in g.iterrows()
            ]))
        prediction = None
        if not opt.overwrite_prediction and os.path.exists(_file):
            with open(_file) as f:
                _prediction = f.read().split('\n')
            if len(_prediction) != len(gold_reference):
                logging.warning(f"found prediction file at {_file} but length not match "
                                f"({len(_prediction)} != {len(gold_reference)})")
            else:
                prediction = _prediction
        if prediction is None:
            model = load_model()
            # model prediction
            if not opt.use_reference_answer:
                logging.info("model prediction: (qag model)")
                prediction = model.generate_qa(
                    list_context=model_input,
                    num_beams=opt.n_beams,
                    batch_size=opt.batch_size)
            else:
                logging.info("model prediction: (qg model, answer fixed by reference)")
                model_input_flat = list(chain(*[[i] * len(h) for i, h in zip(model_input, model_highlight)]))
                model_highlight_flat = list(chain(*model_highlight))
                prediction_flat = model.generate_q(
                    list_context=model_input_flat,
                    list_answer=model_highlight_flat,
                    num_beams=opt.n_beams,
                    batch_size=opt.batch_size)
                _index = 0
                prediction = []
                for h in model_highlight:
                    questions = prediction_flat[_index:_index + len(h)]
                    answers = model_highlight_flat[_index:_index + len(h)]
                    prediction.append(list(zip(questions, answers)))
                    _index += len(h)

            # formatting prediction
            prediction = [' | '.join([f"question: {q}, answer: {a}" for q, a in p]) if p is not None else "" for p in
                          prediction]
            assert len(prediction) == len(model_input), f"{len(prediction)} != {len(model_input)}"
            with open(_file, 'w') as f:
                f.write('\n'.join(prediction))


main()