import numpy as np
def kde_real(img_path,pred_count):
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
  if(pred_count<150):
    beta_increment = 0.05
    k_increment= 1  # Increment for beta
  elif(pred_count<300):
    beta_increment=0.025
  else:
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
  return real_points.shape[0]
print(kde_real("pred_density.png"))
