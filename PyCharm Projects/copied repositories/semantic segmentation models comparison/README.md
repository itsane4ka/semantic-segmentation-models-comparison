# Semantic Segmentation

## Performance:

| Model | mean IOU |  
| :---: | :---: | 
| FCN32s | 45.07% | 
| FCN16s | 44.92% | 
| FCN8s  | 45.28% |  
| PSPNet | 49.25% |
| PSPNet + focal Loss| **50.11%** |

## Train the model
```
python main.py --test False --model pspnet --batch_size 24 --epoch 50 --gpu_mode True
```

## Test the model
```
python main.py --test True --model pspnet --gpu_mode True --pretrain [location of your pretrained model]
```
