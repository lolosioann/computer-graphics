class MatPhong:
    def __init__(self, ka: float, kd: float, ks: float, n: float) -> None:
        """
        Represents a material with Phong lighting properties.

        Parameters:
        ka (float): Ambient reflection coefficient - controls how much ambient light the material reflects.
        kd (float): Diffuse reflection coefficient - controls how much diffuse light the material reflects.
        ks (float): Specular reflection coefficient - controls the strength of the specular highlight.
        n (float): Phong exponent - controls the shininess or glossiness of the material surface.
        """
        self.ka = ka  # Ambient reflection coefficient
        self.kd = kd  # Diffuse reflection coefficient
        self.ks = ks  # Specular reflection coefficient
        self.n = n    # Phong exponent


