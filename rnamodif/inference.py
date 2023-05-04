import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pickle
from collections import defaultdict
import numpy as np

from rnamodif.data_utils.dataloading_5eu import CompleteReadsInferenceDataset
from rnamodif.rodan_seq_5eu import RodanPretrained
from rnamodif.data_utils.workers import worker_init_fn_inference


def predictions_to_read_predictions(predictions, pooling):
    id_to_preds = defaultdict(list)
    for pr, ids in predictions:
        readid_probs = zip(ids['readid'], pr.numpy())
        for readid, probab in readid_probs:
            id_to_preds[readid].append(probab)
                
    if(pooling == 'max'):
        for k,v in id_to_preds.items():
            id_to_preds[k] = np.array(v).max()
        return id_to_preds
    if(pooling == 'mean'):
        for k,v in id_to_preds.items():
            id_to_preds[k] = np.array(v).mean()
        return id_to_preds
    if(pooling == 'none'):
        id_to_preds_nopool = {}
        for k,v in id_to_preds.items():
            for i,prob in enumerate(v):
                id_to_preds_nopool[f'{k}_{i}'] = prob
        return id_to_preds_nopool
    
    else:
        raise Exception(f'{pooling} pooling not implemented')
        
        
def main(args):
    files = list(Path(args.path).rglob('*.fast5'))
    print('Number of fast5 files found:', len(files))

    stride = args.window - args.overlap
    dset = CompleteReadsInferenceDataset(files=files, window=args.window, stride=stride)

    model = RodanPretrained().load_from_checkpoint(args.checkpoint)

    workers = min([args.max_workers, len(dset.files)])
    dataloader = DataLoader(dset, batch_size=args.batch_size, num_workers=workers, pin_memory=True, worker_init_fn=worker_init_fn_inference)

    trainer = pl.Trainer(accelerator='gpu', precision=16)
    window_preds = trainer.predict(model, dataloader)

    with open(args.window_output, 'wb') as handle:
        pickle.dump(window_preds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    read_preds = predictions_to_read_predictions(window_preds, pooling=args.pooling)
    with open(args.read_output, 'wb') as handle:
        pickle.dump(read_preds, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
    parser.add_argument('--path', type=str, required=True, help='Path to the folder containing FAST5 files.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint file.')
    parser.add_argument('--window_output', type=str, required=True, help='Path to the output pickle file for window predictions.')
    parser.add_argument('--read_output', type=str, required=True, help='Path to the output pickle file for read predictions.')
    parser.add_argument('--max_workers', type=int, default=16, help='Maximum number of workers for data loading (default: 16).')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for data loading (default: 256).')
    parser.add_argument('--window', type=int, default=4096, help='Window size for data processing (default: 4096).')
    parser.add_argument('--overlap', type=int, default=1024, help='Overlap of neighbouring windows (default: 1024).')
    parser.add_argument('--pooling', type=str, default='mean', help='Type of pooling to use to combine window predictions to read predictions (default: mean).')
    
    
    args = parser.parse_args()
    main(args)