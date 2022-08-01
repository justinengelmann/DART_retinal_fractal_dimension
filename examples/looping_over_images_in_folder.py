# temporarily append the parent directory to the path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dart import get_inference_pipeline
from PIL import Image
import glob
from pathlib import Path

CROP_THRESHOLD = 70

def prompt_for_bool(prompt):
    while True:
        try:
            return {"yes": True, "no": False}[input(prompt).lower()]
        except KeyError:
            print("Invalid input. Please type 'yes' or 'no'!")


# check if dart_inference_results.csv exists
assert not os.path.exists('dart_inference_results.csv'), \
    'ERROR: dart_inference_results.csv already exists. Please delete \ rename it before running this script.'

# img_folder = 'PATH_TO_YOUR_IMAGES_FOLDER'
# img_ext = 'jpg'
# get user input
img_folder = input('Enter the path to the folder containing your images: ')
img_ext = input('Enter the extension of your images (e.g. jpg, png, etc.): ')

crop_black_borders = prompt_for_bool('Do you want to crop black borders from your images? (yes/no): ')
print(f'Okay, we {"will" if crop_black_borders else "will not"} crop black borders.')

print(f'Looking for images in {img_folder} with extension {img_ext}')
images = list(glob.glob(img_folder + f'/*.{img_ext}'))

assert len(images) > 0, 'No images found'
print(f'Found {len(images)} images, e.g. {Path(images[0]).name}, ..., {Path(images[-1]).name}')

print('Loading pipeline...')
dart_pipeline = get_inference_pipeline(model_name='resnet18', device='cpu',
                                       resize_images=True, preprocessing_backend='albumentations_if_available',
                                       crop_black_borders=crop_black_borders,
                                       crop_threshold=CROP_THRESHOLD,
                                       loading_verbose=True, loading_pbar=True)

print('\nStarting inference...')
results = []
for img_path in images:
    print(f'Inferring {img_path}', end='\r')
    img = Image.open(img_path)
    print(f'Inferring {img_path} - Image loaded, running model', end='\r')
    FD = dart_pipeline(img)[0]
    print(f'{img_path}: {FD}')
    results.append((img_path, Path(img_path).name, FD))

print('Inference complete!')

print('Writing results to file...')
with open('dart_inference_results.csv', 'w') as f:
    f.write('image_path,image_name,FD\n')
    for img_path, img_name, FD in results:
        f.write(f'{img_path}, {img_name}, {FD}\n')

print('Created dart_inference_results.csv')

# remove the temporarily added parent directory
sys.path.pop()

print('Done!')
