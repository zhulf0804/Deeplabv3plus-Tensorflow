## About

A Tensorflow implementation of [Deeplabv3plus](http://openaccess.thecvf.com/content_ECCV_2018/papers/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.pdf), trained on [VOC2012](http://host.robots.ox.ac.uk:8080/) data set.


## Train
> python train.py

## Predict

+ randomly select images

    > python predict.py
   
+ select one image
    
    > python predict.py --prediction_on_which val --filename 2009_003804
    

## Evaluation

+ for val data, generate prediction results and get mIoU.
+ for test data,  generate prediction results.

> python evaluate.py