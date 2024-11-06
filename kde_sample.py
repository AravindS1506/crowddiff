import numpy as np
import cv2
from sklearn.neighbors import KernelDensity
import random
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
def kde_real(img_path):
  # Load the image (grayscale mode)

  image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
# Step 1: Denoise the image using Non-Local Means Denoising
  denoisedImg = cv2.fastNlMeansDenoising(image)

# Step 2: Thresholding the denoised image using Otsu's method
  threshold = 0  # Otsu's method determines the optimal threshold automatically
  th, threshedImg = cv2.threshold(denoisedImg, threshold, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Step 3: Extract non-zero points from the thresholded image (representing crowd points)
  points = np.column_stack(np.where(threshedImg == 0))

  bandwidth = 22  # Adjust bandwidth as needed
  kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
  kde.fit(points)


  x_min, x_max = points[:, 1].min() - 10, points[:, 1].max() + 10
  y_min, y_max = points[:, 0].min() - 10, points[:, 0].max() + 10
  x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
  grid_points = np.c_[y_grid.ravel(), x_grid.ravel()]  # KDE expects y, x order

  # Calculate KDE density values for each point on the grid
  log_densities = kde.score_samples(grid_points)
  densities = np.exp(log_densities).reshape(x_grid.shape)

  # Plot the KDE-fitted distribution and save as a file
  plt.figure(figsize=(10, 6))

  # Show original thresholded image for reference
  plt.imshow(threshedImg, cmap='gray', extent=(x_min, x_max, y_max, y_min))

  # Overlay the KDE density as a contour plot
  plt.contourf(x_grid, y_grid, densities, cmap='hot', alpha=0.5)
  plt.colorbar(label="Density")
  plt.title("KDE-Fitted Density Distribution")
  plt.xlabel("X-axis")
  plt.ylabel("Y-axis")

  # Save the figure without displaying it
  plt.savefig("density_map_kde.png")
  plt.close()


  log_densities = kde.score_samples(points)
  densities = np.exp(log_densities)
  density_threshold = np.percentile(densities, 2)
  real_points = points[densities>density_threshold]
  k=4
  beta=0.85
  neigh = NearestNeighbors(n_neighbors=k+1)
  neigh.fit(real_points)
  distances, _ = neigh.kneighbors(real_points)
  r=(beta/2)*np.mean(distances[:, 1:], axis=1)

  new_realization = kde.sample(4*real_points.shape[0])
  densities = np.exp(kde.score_samples(new_realization))
  density_threshold_up = np.percentile(densities, 60)
  density_threshold_down = np.percentile(densities, 20)

# Use element-wise conditions with `&`
  high_density_points = new_realization[(densities < density_threshold_up) & (densities > density_threshold_down)]

  high_density_points = high_density_points.astype(int)
  added_points_count = 0
  k_increment=0
  beta_increment=0.0125
  for i in range(high_density_points.shape[0]):
    distances, _ = neigh.kneighbors(high_density_points[i].reshape(1,-1))
    dists = np.linalg.norm(real_points - high_density_points[i].reshape(1,-1), axis=1)
    if np.all(dists > r):
      real_points = np.append(real_points, [high_density_points[i]], axis=0)
      neigh = NearestNeighbors(n_neighbors=k+1)
      neigh.fit(real_points)
      distances, _ = neigh.kneighbors(real_points)

      r=(beta/2)*np.mean(distances[:, 1:], axis=1)
      added_points_count += 1

          # Increment k every 5 points added
      if added_points_count % 5 == 0:
        beta += beta_increment  # Slightly increase beta
        k+=k_increment

  fig = plt.figure(figsize=(2.56, 2.56))  # 2.56 inches * 100 DPI = 256 pixels
  plt.scatter(real_points[:, 1], real_points[:, 0], c='red', s=0.25)
  plt.xlim(0, 256)
  plt.ylim(0, 256)
  plt.title("KDE Realization")
  plt.gca().invert_yaxis()

  # Save the figure with a DPI of 100 to ensure it's 256x256 pixels
  plt.savefig("fused_kde_realization.png", dpi=100, bbox_inches='tight')
  plt.close()
  return real_points.shape[0]

def fuse_mul(img1,img2,img3):

  image = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
  denoisedImg = cv2.fastNlMeansDenoising(image)

  threshold = 0 
  th, threshedImg = cv2.threshold(denoisedImg, threshold, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
  points1 = np.column_stack(np.where(threshedImg == 0))


  img_path = img2
  image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  denoisedImg = cv2.fastNlMeansDenoising(image)

  threshold = 0
  th, threshedImg = cv2.threshold(denoisedImg, threshold, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

  points2 = np.column_stack(np.where(threshedImg == 0))

  img_path = img3 
  image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  denoisedImg = cv2.fastNlMeansDenoising(image)


  threshold = 0 
  th, threshedImg = cv2.threshold(denoisedImg, threshold, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

  points3 = np.column_stack(np.where(threshedImg == 0))
  new=[]
  new.append(points2)
  new.append(points3)
  new=np.vstack(new)
  k=4
  beta=0.85
  neigh = NearestNeighbors(n_neighbors=k+1)
  neigh.fit(points1)
  real_points=points1
  distances, _ = neigh.kneighbors(real_points)
  r=(beta/2)*np.mean(distances[:, 1:], axis=1)
  for i in range(new.shape[0]):
    distances, _ = neigh.kneighbors(new[i].reshape(1,-1))
    dists = np.linalg.norm(real_points - new[i].reshape(1,-1), axis=1)
    if np.all(dists > r):
      real_points = np.append(real_points, [new[i]], axis=0)
      neigh = NearestNeighbors(n_neighbors=k+1)
      neigh.fit(real_points)
      distances, _ = neigh.kneighbors(real_points)

      r=(beta/2)*np.mean(distances[:, 1:], axis=1)
  return real_points.shape[0]
img1="sample_imgs/real_1.png"
img2="sample_imgs/real_2.png"
img3="sample_imgs/real_3.png"
kde_res=kde_real("sample_imgs/real_1.png")
diff_res=fuse_mul(img1,img2,img3)
print(f"Samples count obtained from KDE {kde_res}")
print(f"Sample count obtained from multiple realization {diff_res}")
