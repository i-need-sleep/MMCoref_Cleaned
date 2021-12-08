## UNITER-Based Situated Coreference Resolution with Rich Multimodal Input: [arXiv](https://arxiv.org/abs/2112.03521)
# MMCoref_cleaned
Code for the MMCoref task of the [SIMMC 2.0](https://github.com/facebookresearch/simmc2) dataset.  
Pretrained vision-language models adapted from [Transformers-VQA](https://github.com/YIKUAN8/Transformers-VQA).  
Zero-shot visual feature extraction using [CLIP](https://github.com/openai/CLIP) and [BUTD](https://github.com/airsplay/py-bottom-up-attention).  
Zero-shot non-visual prefab feature (flattened into strings) extraction using [BERT](https://huggingface.co/bert-large-uncased) and [SBERT](https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1).

## Dependencies
    requirements.txt
    
## Download the data and pretrained/trained model checkpoints
* Data: Put the data in ./data. Unpack all image in ./data/all_images and all scene.jsons (including teststd split) in ./data/simmc2_scene_jsons_dstc10_public/public.
* Pretrained models: Checkpoints in ./pretrained and ./model/Transformers-VQA-master/models/pretrained. Download links in placeholder.txt in these folders.
* Trained models: Checkpints in ./trained. Download from ./trained/placeholder.txt

## Preprocess
* Convert json files ~~using ./scripts/converter.py~~ *Currently not working. (Someone managed to lose the latest converter.py.) Download the processed data instead.
* Get BERT/SBERT embeddings of non-visual prefab features using ./scripts/{get_KB_embedding, get_KB_embedding_SBERT, get_KB_embedding_no_duplicate}.py
* Get CLIP/BUTD embeddigns for images using scripts ./scripts/get-visual-features-{CLIP, RCNN}.ipynb
* Or just download everything from ./processed/placeholder.txt

## Train
* Under ./sh/train. See the arguments for used input.

## Inference and evaluate
* Under ./sh/infer_eval (devtest split) and ./sh/infer_eval_dev (dev split)
* Outputs at ./output (same format as the original dialogue json).
* Logits at ./output/logit {dialogue_idx: {round_idx: \[\[logit, label\], ...\]}}
* run ./scripts/output_filter_error.py to select and reformat error cases.

## Ensemble
`cd script`
`python ensemble --method optuna`
* output saved to `output/logit/blended_devtest.json`

