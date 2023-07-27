import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from align_detect_face import Align_Detect_Face

### Inference
img_list = []
with open("label_test.txt", "r") as file:
   lines = file.readlines()
   for line in lines:
      img_list.append('VN-celeb_align_frontal_full/' + line.split()[0])

img_dict = dict(enumerate(img_list))

def show_imgs(query, f_ids):
  """
  It takes in a query image and a list of filepaths to images, and displays the query image and the
  top 6 images from the list of filepaths

  :param query: the image we want to find similar images to
  :param f_ids: the list of file ids of the images that are most similar to the query image
  """
  plt.imshow(query)
  fig = plt.figure(figsize=(12, 12))
  columns = 3
  rows = 2
  for i in range(1, columns*rows +1):
    img = mpimg.imread(img_list[f_ids[i - 1]])
    ax = fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    plt.axis("off")
  plt.show()

def inference(query_path, model, index):
  """
  Given a query image, we use the retrieval function to extract the feature vector of the query image.
  Then, we use the index to search for the top-k nearest neighbors of the query image

  :param query_path: the path to the query image
  :param retrieval_func: The function that will be used to retrieve the embedding of the query image
  :param index: the index object that we created earlier
  """
  align_detect_face = Align_Detect_Face()
  align_img = align_detect_face (query_path)
  if align_img == None:
    return -1, -1
  else:
    align_img = align_img.unsqueeze(0)
    query = model(align_img).detach().numpy()
    scores, idx_image = index.search(query, k=7)
    return scores[0], idx_image[0]

query_path = 'VN-celeb_align_frontal_full/10/10_0009.png'

normal_scores, normal_ids = inference(query_path, model, normal_idx)
if type(normal_scores) is int and type(normal_ids) is int:
  raise "Error Reading Image"
else:
  print(f"scores: {normal_scores}")
  print(f"idx: {normal_ids}")
  query = Image.open(query_path)
  show_imgs(query, normal_ids)