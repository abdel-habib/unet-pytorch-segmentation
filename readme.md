Tested on the following datasets
1. <a href="https://datasets.simula.no/kvasir-seg/" target="_blank">Kvasir SEG - Segmented Polyp Dataset for Computer Aided Gastrointestinal Disease Detection.</a>


Run segmentation test using the following command
```
python test.py --test_path "datasets/kvasir-seg/augmented/test" --output "results" --seed 42 --ckpt "checkpoints/checkpoint.pth"
```
