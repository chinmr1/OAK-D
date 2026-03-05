import numpy as np
import matplotlib.pyplot as plt

def generate_circular_gradient(size, center_x=None, center_y=None):
    """
    Generates a circular gradient in a 2D NumPy array.
    Values range from 1.0 (white center) to 0.0 (black edges).
    """
    if center_x is None: center_x = size / 2
    if center_y is None: center_y = size / 2

    # 1. Create a coordinate grid
    y, x = np.ogrid[:size, :size]

    # 2. Calculate the squared distance from the center
    # Using squared distance first is more computationally efficient
    dist_sq = (x - center_x)**2 + (y - center_y)**2

    # 3. Calculate actual distance
    dist = np.sqrt(dist_sq)

    # 4. Normalize the distance
    # We divide by the maximum possible distance to the furthest corner
    # so that the black reaches the edges.
    max_dist = np.sqrt(max(center_x, size-center_x)**2 + max(center_y, size-center_y)**2)
    
    # 5. Invert the distance (1.0 at center, 0.0 at edges)
    gradient = 1.0 - (dist / max_dist)
    
    # 6. Clip values to ensure they stay within [0, 1]
    gradient = np.clip(gradient, 0, 1)

    return gradient

# --- Usage & Plotting ---
size = 640
gradient_data = generate_circular_gradient(size)

plt.figure(figsize=(6, 6))
plt.imshow(gradient_data, cmap='gray', vmin=0, vmax=1)
plt.title(f"Circular Gradient ({size}x{size})")
plt.axis('off')
plt.colorbar
plt.show()