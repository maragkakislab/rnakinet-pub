import pysam
import numpy as np
import argparse

def main(bam_path, output_path):
    all_read_ids = []
    with pysam.AlignmentFile(bam_path, "rb") as bam_file:
        for read in bam_file:
            all_read_ids.append(read.query_name)
            
    rng = np.random.RandomState(seed=42)
    rng.shuffle(all_read_ids)
            
    num_reads = len(all_read_ids)
    train_end = int(num_reads * 0.70)
    test_end = train_end + int(num_reads * 0.20)

    train_readids = all_read_ids[:train_end]
    test_readids = all_read_ids[train_end:test_end]
    validation_readids = all_read_ids[test_end:]
    
    #Txt files
    with open(output_path+'/randtrain_readids.txt','w') as txt_file:
        for readid in train_readids:
            txt_file.write(readid + '\n')
    with open(output_path+'/randtest_readids.txt','w') as txt_file:
         for readid in test_readids:
            txt_file.write(readid + '\n')    
    with open(output_path+'/randvalidation_readids.txt','w') as txt_file:
        for readid in validation_readids:
            txt_file.write(readid + '\n')  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a map from a bam file.')
    parser.add_argument('--bam_path', required=True, help='The path to the bam file.')
    parser.add_argument('--output_path', required=True, help='The output path for the final txt files.')

    args = parser.parse_args()

    main(
        bam_path=args.bam_path, 
        output_path=args.output_path, 
    )
