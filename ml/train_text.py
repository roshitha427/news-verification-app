"""Fine-tune multilingual BERT from three explicit, disjoint text,label CSVs.

Labels must be exactly fake or real. Do not provide the image archive here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments


def main(args):
    torch.set_num_threads(args.threads)
    splits={}
    seen=set()
    for split,path in [('test',args.test),('validation',args.validation),('train',args.train)]:
        frame=pd.read_csv(path)
        if not {'text','label'}.issubset(frame.columns):
            raise ValueError('Every CSV must contain text and label columns.')
        frame=frame[['text','label']].dropna()
        if not set(frame['label']).issubset({'fake','real'}):
            raise ValueError('Labels must be exactly fake or real.')
        if frame['label'].nunique()!=2:
            raise ValueError('Each split needs both classes.')
        normalized=frame['text'].astype(str).str.lower().str.replace(r'\s+',' ',regex=True).str.strip()
        hashes=normalized.map(lambda text:hashlib.sha256(text.encode()).hexdigest())
        if seen.intersection(hashes):
            raise ValueError('Duplicate text across splits. Deduplicate before training.')
        seen.update(hashes)
        frame=frame.loc[~hashes.duplicated()].copy()
        frame['label']=frame['label'].map({'real':0,'fake':1})
        splits[split]=Dataset.from_pandas(frame,preserve_index=False)
    tokenizer=AutoTokenizer.from_pretrained(args.base)
    dataset=DatasetDict(splits).map(lambda batch:tokenizer(batch['text'],truncation=True,max_length=args.max_length),batched=True)
    model=AutoModelForSequenceClassification.from_pretrained(args.base,num_labels=2,id2label={0:'real',1:'fake'},label2id={'real':0,'fake':1})
    def metrics(pred):
        logits,y=pred
        guess=np.argmax(logits,axis=1)
        return {'accuracy':float(accuracy_score(y,guess)),'f1':float(f1_score(y,guess))}
    trainer=Trainer(model=model,args=TrainingArguments(output_dir=args.output,learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,per_device_eval_batch_size=args.batch_size,num_train_epochs=args.epochs,
        weight_decay=.01,eval_strategy='epoch',save_strategy='epoch',load_best_model_at_end=True,
        metric_for_best_model='f1',save_total_limit=2,report_to='none',seed=42,logging_steps=100,dataloader_pin_memory=False),
        train_dataset=dataset['train'],eval_dataset=dataset['validation'],processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),compute_metrics=metrics)
    trainer.train()
    # Temperature calibration uses validation, never the test set.
    from scipy.optimize import minimize_scalar
    from sklearn.metrics import log_loss, brier_score_loss
    val_predictions=trainer.predict(dataset['validation'])
    val_logits=val_predictions.predictions[:,1]-val_predictions.predictions[:,0]
    sigmoid=lambda value:1/(1+np.exp(-np.clip(value,-30,30)))
    temperature=float(minimize_scalar(lambda temp:log_loss(val_predictions.label_ids,sigmoid(val_logits/temp),labels=[0,1]),bounds=(.1,10),method='bounded').x)
    predictions=trainer.predict(dataset['test'])
    p=sigmoid((predictions.predictions[:,1]-predictions.predictions[:,0])/temperature)
    evaluation={'accuracy':predictions.metrics['test_accuracy'],'f1':predictions.metrics['test_f1'],
        'test_count':len(dataset['test']),'brier_score':float(brier_score_loss(predictions.label_ids,p)),
        'confusion_matrix':confusion_matrix(predictions.label_ids,np.argmax(predictions.predictions,axis=1)).tolist()}
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    card={'name':'Fine-tuned BERT news classifier','version':'1.0','fake_label':'fake','evaluation':evaluation,
        'base_model':args.base,'target_met':evaluation['accuracy']>=.85,'temperature':temperature,'max_length':args.max_length,
        'languages':['en'],'calibration':'Validation-set temperature scaling.',
        'limitations':['Accuracy only describes the historical English ISOT corpus, not current truth or unseen publishers/events.',
                       'Hindi and Telugu classifier performance is unvalidated; abstain for these languages.',
                       'Random article split may retain topic, publisher-style, and near-duplicate leakage.']}
    manifest=Path(args.train).with_name('manifest.json')
    if manifest.is_file():
        card['dataset']=json.loads(manifest.read_text(encoding='utf8'))
    Path(args.output,'model-card.json').write_text(json.dumps(card,indent=2),encoding='utf8')
    print(json.dumps(card,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for split in ('train','validation','test'):
        parser.add_argument('--'+split,required=True)
    parser.add_argument('--base',default='bert-base-multilingual-cased')
    parser.add_argument('--output',default='models/text')
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--max-length',type=int,default=192)
    parser.add_argument('--threads',type=int,default=2)
    main(parser.parse_args())
