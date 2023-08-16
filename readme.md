Tested on the following datasets
1. <a href="https://datasets.simula.no/kvasir-seg/" target="_blank">Kvasir SEG - Segmented Polyp Dataset for Computer Aided Gastrointestinal Disease Detection.</a>

Run the segmentation training using the following command line
```
python train.py --train_path "datasets/kvasir-seg/augmented/train" --val_path "datasets/kvasir-seg/augmented/val" --output "results" --dataset "kvasir" --dataset "kvasir" --max_epochs 50 --batch_size 2 --base_lr 0.0001 --patience 7 --img_size 512 --seed 42 --ckpt "checkpoints/" 
```

Run segmentation test using the following command line
```
python test.py --test_path "datasets/kvasir-seg/augmented/test" --output "results" --seed 42 --ckpt "results/kvasir_512/checkpoints/kvasir_512_epo50_bs2_lr0.0001_s42.pth" --img_size 512
```
