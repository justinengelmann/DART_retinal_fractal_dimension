## Deep approximation of retinal traits (DART)

**Preprint**: https://arxiv.org/abs/2207.05757
(to appear in the Proceedings of the 9th MICCAI Workshop on Ophthalmic Medical Image Analysis at MICCAI 2022)

Currently, we support retinal Fractal Dimension FD as calculated by VAMPIRE (an awesome tool,
see: https://vampire.computing.dundee.ac.uk/index.html) with the multi-fractal method. FD is a measure of how
complex/branching the retinal vasculature is.

### Quick start

#### Tutorial/Demo

The [tutorial notebook](DART_tutorial_and_demo.ipynb) demonstrates how to use DART and also showcases its speed and
robustness to image quality issues. For more in-depth results see our [preprint](https://arxiv.org/abs/2207.05757).

#### Inference pipeline (most convenient way to use DART)

```python
# simply create an inference pipeline
from dart import get_inference_pipeline
inference_pipeline = get_inference_pipeline(model_name='resnet18')
# your retinal fundus color image, PIL image or numpy array
# ideally square and no large black borders, and similar in appearance to UK Biobank / DRIVE
your_image = ...
FD_of_your_image = inference_pipeline(your_image)[0]
print('Fractal dimension of your image is:', FD_of_your_image)
```

### Other ways to use DART

```python
# 1. batched inference with the inference pipeline (faster than single images)
from dart import get_inference_pipeline
inference_pipeline = get_inference_pipeline(model_name='resnet18')
# iterate over your torch style dataloader
for batch_of_images in your_torch_data_loader:
    batch_of_images.cuda()  # if using GPU
    FD_of_batch = inference_pipeline(batch_of_images)[0]  # returns np array on cpu

# 2. access the components of the inference pipeline themselves
from dart import get_model_and_processing
# this returns a dict containing the model, preprocessing and postprocessing pipelines, and config
model_and_processing = get_model_and_processing(model_name='resnet18')
your_image = ...
your_image_preprocessed = model_and_processing['preprocessing'](your_image)
FD_unscaled = model_and_processing['model'](your_image_preprocessed)
FD_of_your_image = model_and_processing['postprocessing'](FD_unscaled)
    
# 3. access just the model
from dart import load_model
# pytorch compatible model, e.g. for writing your own highly efficient inference loop 
# (with your own preprocessing and postprocessing, see the cfg for details)
model = load_model(model_name='resnet18')
```

If you use our work, please cite our preprint:

```
@article{engelmann2022robust,
  title={Robust and efficient computation of retinal fractal dimension through deep approximation},
  author={Engelmann, Justin and Villaplana-Velasco, Ana and Storkey, Amos and Bernabeu, Miguel O},
  journal={arXiv preprint arXiv:2207.05757},
  year={2022}
}
```

as well as any and all relevant publications by the VAMPIRE authors, for example:

```
@inproceedings{trucco2013novel,
  title={Novel VAMPIRE algorithms for quantitative analysis of the retinal vasculature},
  author={Trucco, E and Ballerini, L and Relan, D and Giachetti, A and MacGillivray, T and Zutis, K and Lupascu, C and Tegolo, D and Pellegrini, E and Robertson, G and others},
  booktitle={2013 ISSNIP Biosignals and Biorobotics Conference: Biosignals and Robotics for Better and Safer Living (BRC)},
  pages={1--4},
  year={2013},
  organization={IEEE}
}
```