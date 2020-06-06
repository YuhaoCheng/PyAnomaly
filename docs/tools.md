## Content
- [Create Logger](#create-logger)
- [Data Reader](#data-reader)

## Create Logger

The code is in `lib/utils/utils.py`


## Data Reader

### Image Reader

The code is in `lib/datatools/abstract/tools.py`. The aim of this reader is to read the image based on the `opencv` or `pillow` and use data augmentation. The final output shape is **[C,H,W]**

#### Introduction

The reader is a class, so you need to create the instance firstly and then use the `read` function of the instance. We think the advantages are that the instance can keep some useful parameters of format, data augmentation and so on. 

The parameters of build the reader are:
- **read_format**: *string.* The reading format of the picture. Users can choose 'opencv' or 'pillow'. *Default: 'opencv'*
- **channel_num**: *int.* The number of images' channel. *Default: 3*
- **channel_name**: *string.* The name of each channel. *Default: 'rgb'*
- **params**: the params of the configuration file. *Default: None*
- **transforms**: the data augmentation functions of `imgaug`. *Default: None*
- **normalize**: *bool.* whether use the normalization. *Default: False*
- **mean**: *list.* the mean value of each channel. *Default: None*
- **std**: *list.* the std value of each channel. *Default: None*
- **deterministic**: bool. the same function of the keyword in `imgaug`. *Default: False*

After create the instance, users can use the `read` function to read the image, the parameters are:

- **name:** the absolute path of the image 
- **flag:** use the torchvision transforms ----> 'torchvision'; use other opensource transforms, like imgaug -----> 'other'. Default: 'other'
- **array_type:** the return type of the image. 'tensor' | 'ndarray'. Default: 'tensor'

#### Example

```python
read_format = 'opencv'
channel_num = 3
channel_name = 'rgb'
name='01.jpg'
image_reader = ImageReader(read_fromat=read_format, channel_num=channel_num)
image = image_reader.read(name)
```

### Video Reader

The code is in `lib/datatools/abstract/tools.py`.

#### Introduction

The reader is based on the [Image Reader](#image-reader). Like the image reader, the video reader also is a class. Users should create the instance and use the `read` function. The final output shape is **[C,D,H,W]**

The parameters of build the reader are:

- **image_loader**: the image reader instance
- **params**: the params of the configuration file. *Default: None*
- **transforms**: the data augmentation functions of `imgaug`. *Default: None*
- **normalize**: *bool.* whether use the normalization. *Default: False*
- **mean**: *list.* the mean value of each channel. *Default: None*
- **std**:  *list.* the std value of each channel. *Default: None*

After create the instance, users can use the `read` function to read the image, the parameters are:

- **frames_list**: *list.* the absolute path of each image.
- **start**: the start number base on the frames_list
- **end**: the end number base on the frames_list
- **clip_length**: the length of each clip. *Default: 2*
- **step**: the step between each frame. *Default: 1*
- **array_type**: the output format. *Default: 'tensor'*

#### Example

```python
read_format = 'opencv'
channel_num = 3
channel_name = 'rgb'
frames_list = ['01.jpg', '02.jpg', '03.jpg', '04.jpg']
image_reader = ImageReader(read_fromat=read_format, channel_num=channel_num)
video_reader = VideoReader(image_loader=image_reader)
clip = video_reader.read(frames_list=frames_list, 0, 3)
```

