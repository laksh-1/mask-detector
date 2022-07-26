# MaskUp - Mask Detector for COVID-19 

The motivation of building this kind of project, came from the recent intense situation of COVID-19. The tedious task of constantly keeping a check for whether a person is following the safety protocols of wearing a mask or not, can be automated.

Keeping the above thought in mind, this project was started. 

The tech stack mainly used here includes: Tensorflow and Keras. Theis application does the job of  highlighting faces on a live video stream and categorizing them on basis of presence and absence of face mask.

## Usage

```python

# for training the model, and generating the data loss stats.
> python ./train_mask_model.py

# runs the application, video source can be altered accordingly.
> python ./detect_mask_video.py

```

## Example
![Example Video](example.mp4)

## License
[MIT](https://choosealicense.com/licenses/mit/)
