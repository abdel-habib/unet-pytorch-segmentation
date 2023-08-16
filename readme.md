Tested on the following datasets
1. <a href="https://datasets.simula.no/kvasir-seg/" target="_blank">Kvasir SEG - Segmented Polyp Dataset for Computer Aided Gastrointestinal Disease Detection.</a>
2. <a href="https://www.kaggle.com/datasets/zionfuo/drive2004">DRIVE 2004 - Digital Retinal Images for Vessel Extraction</a>

Run the segmentation training using the following command line \
`python train.py --train_path "datasets/<dataset-folder-name>/augmented/train" --val_path "datasets/<dataset-folder-name>/augmented/val" --output "results" --dataset "<dataset-name>" --max_epochs 50 --batch_size 2 --base_lr 0.0001 --patience 7 --img_size 512 --seed 42 --ckpt "checkpoints/" 
`


Run segmentation test using the following command line \
`python test.py --test_path "datasets/<dataset-folder-name>/augmented/test" --output "results" --seed 42 --ckpt "results/<dataset-results-folder-name>/checkpoints/<checkpoint-file-name>.pth" --img_size 512
`
