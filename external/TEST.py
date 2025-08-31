import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Générer une image à partir d'une liste d'étoiles
def render_galaxy(stars, size=(1000, 1000), blur_sigma=1.5, mass_scale=2.0):
    image = np.zeros(size)

    for x, y, mass in stars:
        ix = int(x * (size[1] - 1))
        iy = int(y * (size[0] - 1))
        if 0 <= ix < size[1] and 0 <= iy < size[0]:
            image[iy, ix] += mass ** mass_scale

    # Appliquer un flou gaussien pour un effet de halo/luminosité
    blurred = gaussian_filter(image, sigma=blur_sigma)

    # Normaliser l'image pour une visualisation plus dynamique
    norm_blurred = np.clip(blurred / np.percentile(blurred, 99), 0, 1)

    # Créer une image RGB pour coloration artistique (optionnel)
    rgb_image = np.zeros((*size, 3))
    rgb_image[..., 0] = norm_blurred ** 1.5  # R
    rgb_image[..., 1] = norm_blurred ** 1.2  # G
    rgb_image[..., 2] = norm_blurred        # B

    return rgb_image

# Exemple de génération de données aléatoires
def generate_fake_stars(n=10000):
    # Distribution en spirale ou sphérique
    theta = np.random.rand(n) * 2 * np.pi
    r = np.random.normal(0.3, 0.1, n)
    x = 0.5 + r * np.cos(theta)
    y = 0.5 + r * np.sin(theta)
    masses = np.random.exponential(scale=1.0, size=n)
    return list(zip(x, y, masses))

# Générer et afficher l'image
stars = generate_fake_stars(10000)
img = render_galaxy(stars)

plt.figure(figsize=(8, 8))
plt.imshow(img, origin='lower')
plt.axis('off')
plt.title("Galaxy Simulation")
plt.show()
