class MaterialUtils:
    @staticmethod
    def Ms_T_dependence(T, beta, Tc, M0):
        """
        Saturation magnetization as a function of temperature.
        
        Parameters:
        T (float): Temperature in K
        beta (float): Critical exponent
        Tc (float): Curie temperature
        M0 (float): Saturation magnetization at 0 K
        
        Returns:
        float: Saturation magnetization at temperature T
        """
        return M0 * (1 - T / Tc) ** beta