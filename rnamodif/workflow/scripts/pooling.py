import pickle
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd

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
    with open(args.window_predictions, 'rb') as file:
        preds = pickle.load(file)
        
    read_preds = predictions_to_read_predictions(preds, pooling=args.pooling)
    with open(args.out_pickle, 'wb') as handle:
        pickle.dump(read_preds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(args.out_csv, 'wb') as handle:
        df = pd.DataFrame.from_dict(read_preds, orient='index').reset_index()
        df.columns = ['read_id', '5eu_mod_score']
        df['5eu_modified_prediction'] = df['5eu_mod_score'] > args.threshold
        df.to_csv(handle, index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run pooling on predictions and save results. Valid mostly for models that slice a read into chunks for prediction.')
    parser.add_argument('--window_predictions', type=str, required=True, help='Path to the file containing read predictions.')
    parser.add_argument('--out_pickle', type=str, required=True, help='Path to the output pickle file for pooled predictions.')
    parser.add_argument('--out_csv', type=str, required=True, help='Path to the output csv file for pooled predictions.')
    parser.add_argument('--threshold', type=float, required=True, help='Threshold for the predictions to be considered positives')
    parser.add_argument('--pooling', type=str, default='mean', help='Type of pooling to use to combine predictions to read predictions (default: mean).')
    
    
    args = parser.parse_args()
    main(args)