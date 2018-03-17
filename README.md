# Tensorflow implementation of handwritten sequense of small letter recognition.

* The handwritten dataset used is [IAM](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
* The wrapper for code was taken from [youtube-8m](https://github.com/google/youtube-8m) contest.
* The lstm2d.py and lstm1d.py was taken from Tensorflow contrib.
* The input from training and testing are stored in tfrecods files.

### In order to make tfrecords files
* open make tfRecods.ipynb notebook
* change the path '../xml/' and '../forms/' to the path to the xml and forms folders from IAM dataset
* run all the cells on the notebook
* in the 'test-batch1' folder will be created tfrecords files
* note: sorted(glob.glob(pathXML+"*.xml"))[:200] will process just 200 images, change 200 to more

### In order to run the training with slice-image lstm: 
```
python train.py --slices 55 --width 12 --stride 1  --Bwidth 350 --vocabulary_size 29 \
--height 25 --train_data_pattern test-batch1/handwritten-test-{}.tfrecords --train_dir models-feds \
--test_data_pattern test-batch1/handwritten-test-{}.tfrecords  --max_steps 20 --batch_size 20 --beam_size 1 \
--input_chanels 1 --start_new_model --rnn_cell LSTM --model LSTMCTCModel --num_epochs 6000
```

### In order to run the inference with slice-image lstm: 
```
python inference.py --slices 55 --width 12 --Bwidth 350 --stride 1 \
    --input_chanels 1 --height 25 --input_data_pattern  test-batch1/handwritten-test-1.tfrecords \
    --train_dir models-feds  --batch_size 20 --beam_size 1 
```

### In order to see statistics in tensorboard:
```
tensorboard --logdir=separable_lstm --port=8080
```

<center>
label rate error for test images  
<img src="./loss.png">
</center>

<center>
ctc loss for test images 
<img src="./labelrateerror.png">
</center>


* In order to see some prediction example checkout the [jupyter notebook](https://github.com/johnsmithm/handwritten-tf-1.0/blob/master/inference%20example.ipynb)
