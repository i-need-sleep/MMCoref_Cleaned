import os
import json
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup

from utils.focalloss import FocalLoss
from utils.dataset import make_loader
from utils.evaluate_dst import evaluate_from_json

from model.modified_uniter import Modified_Uniter
from model.modified_lxmert import Modified_Lxmert
from model.model_utils import make_KBid_emb_init

def infer(args):
    # Constant setup
    NAME = args.NAME
    MODEL = args.MODEL
    BATCH_SIZE = 1
    SPLIT = args.SPLIT
    CHECKPOINT = args.CHECKPOINT
    print(f'{NAME}, split: {SPLIT}\n')

    arg_dict = vars(args)
    for model_flag in ['obj_id', 'vis_feats_clip', 'vis_feats_rcnn', 'pos', 'scene_seg', 'obj_embs_bert', 'obj_embs_sbert', 'kb_id_bert', 'kb_id_sbert', 'attn_bias','graph_attn','pred_men']:
        print(model_flag)
        if arg_dict[model_flag] == 'True':
            arg_dict[model_flag] = True
            print('True')
        else:
            arg_dict[model_flag] = False
            print('False')

    torch.manual_seed(21)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Make loaders
    test_loader = make_loader(SPLIT, BATCH_SIZE, more_roi=arg_dict['more_roi'])

    # Load models
    if MODEL == 'UNITER':
        model = Modified_Uniter(arg_dict['obj_id'], arg_dict['vis_feats_clip'], arg_dict['vis_feats_rcnn'], arg_dict['pos'], arg_dict['scene_seg'], arg_dict['obj_embs_bert'], arg_dict['obj_embs_sbert'], arg_dict['kb_id_bert'], arg_dict['kb_id_sbert'], arg_dict['attn_bias'], arg_dict['graph_attn'], arg_dict['obj_men'], arg_dict['pred_men']).to(device)
    elif MODEL == 'LXMERT':
        model = Modified_Lxmert(arg_dict['obj_id'], arg_dict['vis_feats_clip'], arg_dict['vis_feats_rcnn'], arg_dict['pos'], arg_dict['scene_seg'], arg_dict['obj_embs_bert'], arg_dict['obj_embs_sbert'], arg_dict['kb_id_bert'], arg_dict['kb_id_sbert'], arg_dict['attn_bias'], arg_dict['graph_attn'], arg_dict['obj_men']).to(device)
    model.load_state_dict(torch.load(f'./trained/{CHECKPOINT}.bin', map_location=device)['model_state_dict'])

    # Infer
    out = {}
    logit_out = {}
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            txt_seg_ids = batch['txt_seg_ids'].to(device)
            vis_feats_clip = batch['vis_feats'].to(device)
            vis_feats_rcnn = batch['vis_feats_rcnn'].to(device)
            kb_embs_bert = batch['obj_embs'].to(device)
            kb_embs_sbert = batch['obj_embs_SBERT'].to(device)
            obj_ids = batch['obj_ids'].to(device)
            pos_x = batch['pos_x'].to(device)
            pos_y = batch['pos_y'].to(device)
            pos_z = batch['pos_z'].to(device)
            bboxes = batch['bboxes'].to(device)
            vis_seg = batch['vis_seg'].to(device)
            extended_attention_mask = batch['extended_attention_mask'].to(device)
            output_mask = batch['output_mask'].to(device)
            reference = batch['reference'].to(device)
            scene_seg = batch['scene_segs'].to(device)
            kb_id = batch['KB_ids'].to(device)
            rel_mask_left = batch['rel_mask_left'].to(device).long()
            rel_mask_right = batch['rel_mask_right'].to(device).long()
            rel_mask_up = batch['rel_mask_up'].to(device).long()
            rel_mask_down = batch['rel_mask_down'].to(device).long()
            input_ids_uncased = batch['input_ids_uncased'].to(device)
            input_mask_uncased = batch['input_mask_uncased'].to(device).long()
            segment_ids_uncased = batch['segment_ids_uncased'].to(device)
            bboxes_lxmert = batch['bboxes_lxmert'].to(device).float()
            vis_mask = batch['vis_mask'].to(device).long()
            dial_idx = batch['dial_idx']
            round_idx = batch['round_idx']

            obj_men = batch['obj_men'].to(device).long()

            truth = reference.float().reshape(-1,1)

            if MODEL == "UNITER":
                if arg_dict['pred_men']:
                    pred, pred_men = model(input_ids, txt_seg_ids, vis_seg, bboxes, extended_attention_mask, obj_ids, vis_feats_clip, vis_feats_rcnn, pos_x, pos_y, pos_z, scene_seg, kb_embs_bert, kb_embs_sbert, kb_id, rel_mask_left, rel_mask_right, rel_mask_up, rel_mask_down)
                else:
                    pred = model(input_ids, txt_seg_ids, vis_seg, bboxes, extended_attention_mask, obj_ids, vis_feats_clip, vis_feats_rcnn, pos_x, pos_y, pos_z, scene_seg, kb_embs_bert, kb_embs_sbert, kb_id, rel_mask_left, rel_mask_right, rel_mask_up, rel_mask_down, obj_men=obj_men)
            elif MODEL == 'LXMERT':
                pred = model(input_ids_uncased, input_mask_uncased, segment_ids_uncased, bboxes_lxmert, obj_ids, vis_feats_clip, vis_feats_rcnn, pos_x, pos_y, pos_z, scene_seg, kb_embs_bert, kb_embs_sbert, kb_id, vis_mask, obj_men=obj_men)

            pred = pred.reshape(1,-1)
            pred = pred[output_mask==1].reshape(-1,1)

            obj_ids -= 1 # -1 for padding. 0 is a valid index
            line_out = []
            logit_line_out = []
            dial_idx = dial_idx[0]
            round_idx = round_idx[0]

            for idx, prediction in enumerate(pred):
                if prediction > 0:
                    line_out.append(obj_ids[0][idx].item())
                logit_line_out.append([prediction.item(), truth[idx].item()])
            
            try:
                out[dial_idx][round_idx] = line_out
                logit_out[dial_idx][round_idx] = logit_line_out
            except:
                out[dial_idx] = {round_idx: line_out}
                logit_out[dial_idx] = {round_idx: logit_line_out}
        
    with open(f'./data/simmc2_dials_dstc10_{SPLIT}.json', 'r') as data_file:
        data = json.load(data_file)

    for dial in data['dialogue_data']:
        dial_mentions = []
        dial_idx = dial['dialogue_idx']
        for round_idx, round in enumerate(dial['dialogue']):
            try:
                if round['disambiguation_label'] == 1:
                    continue
            except:
                pass
            round['transcript_annotated']['act_attributes']['objects_real'] = round['transcript_annotated']['act_attributes']['objects']
            round['transcript_annotated']['act_attributes']['objects'] = out[dial_idx][round_idx]
            for obj_idx in out[dial_idx][round_idx]:
                if obj_idx not in dial_mentions:
                    dial_mentions.append(obj_idx)
        dial['mentioned_object_ids'] = dial_mentions

    with open(f'./output/{NAME}_{SPLIT}.json', 'w', encoding='utf-8') as out_file:
        json.dump(data, out_file)

    with open(f'./output/logit/{NAME}_{SPLIT}.json', 'w', encoding='utf-8') as out_file:
        json.dump(logit_out, out_file)

    # Evaluate
    json_target = json.load(open(f'./data/simmc2_dials_dstc10_{SPLIT}.json', "r"))
    json_predicted = data

    # Evaluate
    report = evaluate_from_json(
        json_target["dialogue_data"], json_predicted["dialogue_data"]
    )
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--NAME', default='unnamed')
    parser.add_argument('--MODEL', default='UNITER')
    parser.add_argument('--SPLIT', default='devtest')
    parser.add_argument('--CHECKPOINT', default='')
    
    parser.add_argument('--obj_id', default=False)
    parser.add_argument('--vis_feats_clip', default=False)
    parser.add_argument('--vis_feats_rcnn', default=False)
    parser.add_argument('--pos', default=False)
    parser.add_argument('--scene_seg', default=False)
    parser.add_argument('--obj_embs_bert', default=False)
    parser.add_argument('--obj_embs_sbert', default=False)
    parser.add_argument('--kb_id_bert', default=False)
    parser.add_argument('--kb_id_sbert', default=False) # Not implemented. This line should always be False
    parser.add_argument('--obj_men', default=False)

    parser.add_argument('--pred_men', default=False) # Auxilliary prediction 
    
    parser.add_argument('--attn_bias', default=False)
    parser.add_argument('--graph_attn', default=False)

    parser.add_argument('--more_roi', default=False)

    args = parser.parse_args()
    
    infer(args)