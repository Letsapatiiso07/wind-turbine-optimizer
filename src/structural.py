import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

# Handle SciPy version compatibility for cumulative trapezoidal integration
try:
    # SciPy >= 1.6.0
    from scipy.integrate import cumulative_trapezoid
    def cumtrapz(y, x=None, initial=0):
        """Wrapper for scipy.integrate.cumulative_trapezoid."""
        return cumulative_trapezoid(y, x=x, initial=initial)
except ImportError:
    try:
        # SciPy < 1.6.0
        from scipy.integrate import cumtrapz
    except ImportError:
        # Fallback: manual implementation
        def cumtrapz(y, x=None, initial=0):
            """Manual implementation of cumulative trapezoidal integration."""
            if x is None:
                x = np.arange(len(y))
            result = np.zeros(len(y))
            result[0] = initial
            for i in range(1, len(y)):
                result[i] = result[i-1] + (y[i] + y[i-1]) * (x[i] - x[i-1]) / 2
            return result


@dataclass
class MaterialProperties:
    """Composite material properties for blade structure."""
    
    E_flapwise: float = 40e9    # [Pa] Modulus in flap direction
    E_edgewise: float = 30e9    # [Pa] Modulus in edge direction
    G: float = 5e9              # [Pa] Shear modulus
    rho: float = 1800.0         # [kg/m³] Density
    sigma_ult: float = 500e6    # [Pa] Ultimate tensile strength
    safety_factor: float = 2.5  # Design safety factor
    
    @property
    def sigma_allow(self) -> float:
        """Allowable stress."""
        return self.sigma_ult / self.safety_factor


@dataclass
class StructuralResult:
    """Results from structural analysis."""
    
    # Deflections
    tip_deflection_flap: float    # [m]
    tip_deflection_edge: float    # [m]
    deflection_flap: np.ndarray   # [m] along span
    deflection_edge: np.ndarray   # [m] along span
    
    # Stresses
    max_stress_flap: float        # [Pa]
    max_stress_edge: float        # [Pa]
    stress_flap: np.ndarray       # [Pa] along span
    stress_edge: np.ndarray       # [Pa] along span
    
    # Mass
    blade_mass: float             # [kg]
    
    # Structural constraint violations
    deflection_ratio: float       # tip_deflection / span
    stress_ratio: float           # max_stress / sigma_allow
    
    def is_feasible(self, max_deflection_ratio: float = 0.05) -> bool:
        """Check if design meets structural constraints."""
        return (self.deflection_ratio < max_deflection_ratio and 
                self.stress_ratio < 1.0)


class StructuralModel:
    """
    1D Euler-Bernoulli beam model for blade structural analysis.
    
    Computes deflections, stresses, and blade mass.
    """
    
    def __init__(self, material: Optional[MaterialProperties] = None):
        """
        Initialize structural model.
        
        Args:
            material: Material properties (uses default if None)
        """
        self.material = material or MaterialProperties()
    
    def analyze(
        self,
        blade_geometry,  # BladeGeometry instance
        loads_flap: np.ndarray,  # [N/m] distributed load in flap
        loads_edge: np.ndarray   # [N/m] distributed load in edge
    ) -> StructuralResult:
        """
        Analyze blade structural response.
        
        Args:
            blade_geometry: BladeGeometry object
            loads_flap: Flapwise distributed loads [N/m]
            loads_edge: Edgewise distributed loads [N/m]
            
        Returns:
            StructuralResult object
        """
        span = blade_geometry.span_positions
        chord = blade_geometry.chord
        n_sections = len(span)
        
        # Estimate blade cross-sectional properties
        # Simple model: assume airfoil-like cross-section
        thickness_ratio = 0.18 - 0.12 * (span / span[-1])  # Root thick, tip thin
        thickness = chord * thickness_ratio
        
        # Second moment of area (simplified box beam approximation)
        # I ≈ (chord * thickness³) / 12
        I_flap = (chord * thickness**3) / 12
        I_edge = (thickness * chord**3) / 12
        
        # Cross-sectional area
        A = chord * thickness * 0.7  # 0.7 accounts for hollow airfoil
        
        # Blade mass
        blade_mass = np.trapz(A * self.material.rho, span)
        
        # Solve beam equation for deflections (double integration of curvature)
        # M(x) = ∫∫ q(x) dx dx  (bending moment)
        # w(x) = ∫∫ M(x)/(EI) dx dx (deflection)
        
        # Flapwise analysis
        moment_flap = self._compute_moment_distribution(span, loads_flap)
        curvature_flap = moment_flap / (self.material.E_flapwise * I_flap)
        slope_flap = cumtrapz(curvature_flap, span, initial=0)
        deflection_flap = cumtrapz(slope_flap, span, initial=0)
        
        # Edgewise analysis
        moment_edge = self._compute_moment_distribution(span, loads_edge)
        curvature_edge = moment_edge / (self.material.E_edgewise * I_edge)
        slope_edge = cumtrapz(curvature_edge, span, initial=0)
        deflection_edge = cumtrapz(slope_edge, span, initial=0)
        
        # Tip deflections
        tip_deflection_flap = abs(deflection_flap[-1])
        tip_deflection_edge = abs(deflection_edge[-1])
        
        # Stresses (bending stress σ = M * c / I, where c is distance from neutral axis)
        c_flap = thickness / 2
        c_edge = chord / 2
        
        stress_flap = np.abs(moment_flap * c_flap / I_flap)
        stress_edge = np.abs(moment_edge * c_edge / I_edge)
        
        max_stress_flap = np.max(stress_flap)
        max_stress_edge = np.max(stress_edge)
        
        # Constraint ratios
        deflection_ratio = max(tip_deflection_flap, tip_deflection_edge) / span[-1]
        stress_ratio = max(max_stress_flap, max_stress_edge) / self.material.sigma_allow
        
        return StructuralResult(
            tip_deflection_flap=tip_deflection_flap,
            tip_deflection_edge=tip_deflection_edge,
            deflection_flap=deflection_flap,
            deflection_edge=deflection_edge,
            max_stress_flap=max_stress_flap,
            max_stress_edge=max_stress_edge,
            stress_flap=stress_flap,
            stress_edge=stress_edge,
            blade_mass=blade_mass,
            deflection_ratio=deflection_ratio,
            stress_ratio=stress_ratio
        )
    
    def _compute_moment_distribution(
        self,
        span: np.ndarray,
        loads: np.ndarray
    ) -> np.ndarray:
        """
        Compute bending moment distribution from distributed loads.
        
        M(x) = ∫(x to L) ∫(ξ to L) q(η) dη dξ
        
        Args:
            span: Spanwise positions
            loads: Distributed loads [N/m]
            
        Returns:
            Bending moment distribution [Nm]
        """
        n = len(span)
        moment = np.zeros(n)
        
        # Integrate from tip (free end) to root
        for i in range(n):
            # Moment at position i is integral of loads from i to tip
            moment[i] = np.trapz(
                loads[i:] * (span[i:] - span[i]),
                span[i:]
            )
        
        return moment


@dataclass
class SNCurve:
    """S-N curve parameters for fatigue analysis."""
    
    m: float = 10.0        # Wöhler exponent (typical for composites)
    C: float = 1e13        # Fatigue strength coefficient
    
    def cycles_to_failure(self, stress_range: float) -> float:
        """
        Compute cycles to failure for a given stress range.
        
        Args:
            stress_range: Stress range [Pa]
            
        Returns:
            Number of cycles to failure
        """
        if stress_range <= 0:
            return np.inf
        
        # N = C / S^m
        return self.C / (stress_range ** self.m)


class FatigueAnalyzer:
    """
    Fatigue damage analysis using rainflow counting and Miner's rule.
    
    Simplified implementation for rapid analysis.
    """
    
    def __init__(self, sn_curve: Optional[SNCurve] = None):
        """
        Initialize fatigue analyzer.
        
        Args:
            sn_curve: S-N curve for material
        """
        self.sn_curve = sn_curve or SNCurve()
    
    def compute_damage(
        self,
        load_time_series: np.ndarray,
        n_cycles_lifetime: int = int(1e8)  # 20 years at typical frequency
    ) -> float:
        """
        Compute fatigue damage using Miner's rule.
        
        D = Σ(n_i / N_i) where n_i is number of cycles at stress range S_i
        and N_i is cycles to failure at S_i.
        
        Args:
            load_time_series: Time series of load (e.g., bending moment)
            n_cycles_lifetime: Total cycles in lifetime
            
        Returns:
            Damage ratio D (D < 1 is safe)
        """
        # Simplified rainflow counting (for speed, use binned approach)
        ranges, counts = self._simple_rainflow(load_time_series)
        
        # Compute damage contribution from each bin
        damage = 0.0
        for stress_range, n_cycles in zip(ranges, counts):
            N_f = self.sn_curve.cycles_to_failure(stress_range)
            damage += n_cycles / N_f
        
        # Scale to lifetime
        damage_lifetime = damage * (n_cycles_lifetime / len(load_time_series))
        
        return damage_lifetime
    
    def _simple_rainflow(
        self,
        signal: np.ndarray,
        n_bins: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simplified rainflow counting using range-pair method.
        
        Args:
            signal: Load time series
            n_bins: Number of bins for histogram
            
        Returns:
            (stress_ranges, cycle_counts)
        """
        # Find peaks and valleys
        extrema = self._find_extrema(signal)
        
        if len(extrema) < 2:
            return np.array([0]), np.array([0])
        
        # Compute ranges between consecutive extrema
        ranges = np.abs(np.diff(extrema))
        
        # Histogram of ranges
        hist, bin_edges = np.histogram(ranges, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return bin_centers, hist
    
    def _find_extrema(self, signal: np.ndarray) -> np.ndarray:
        """Find local maxima and minima in signal."""
        # Simple approach: use where derivative changes sign
        diff = np.diff(signal)
        
        extrema_indices = []
        for i in range(1, len(diff)):
            if diff[i-1] * diff[i] < 0:  # Sign change
                extrema_indices.append(i)
        
        if len(extrema_indices) == 0:
            return signal[[0, -1]]
        
        return signal[extrema_indices]
    
    def estimate_damage_from_power_curve(
        self,
        blade_geometry,
        power_curve_results: Dict,
        wind_distribution: Dict,
        lifetime_years: float = 20.0
    ) -> float:
        """
        Estimate fatigue damage from power curve and wind distribution.
        
        Args:
            blade_geometry: BladeGeometry object
            power_curve_results: Results from BEMSolver.compute_power_curve
            wind_distribution: {'wind_speed': array, 'probability': array}
            lifetime_years: Design lifetime
            
        Returns:
            Fatigue damage ratio D
        """
        # Assume bending moment proportional to thrust
        thrust = power_curve_results['thrust']  # [kN]
        wind_speeds = power_curve_results['wind_speed']
        
        # Convert to bending moment (simplified: M = T * L/2)
        span = blade_geometry.params.blade_span
        moment = thrust * 1000 * span / 2  # [Nm]
        
        # Build load spectrum weighted by wind probability
        wind_prob = wind_distribution['probability']
        
        # Total damage
        damage = 0.0
        
        # Hours per year at each wind speed
        hours_per_year = 8760
        cycles_per_hour = 0.5 * 60  # Assume ~0.5 Hz blade rotation
        
        for i, (m, p) in enumerate(zip(moment, wind_prob)):
            if p == 0 or m == 0:
                continue
            
            # Number of cycles at this stress level over lifetime
            n_cycles = p * hours_per_year * lifetime_years * cycles_per_hour
            
            # Cycles to failure at this stress
            N_f = self.sn_curve.cycles_to_failure(m)
            
            # Damage contribution
            damage += n_cycles / N_f
        
        return damage


if __name__ == "__main__":
    # Test structural analysis
    from parameterization import create_baseline_blade
    from simulation import BEMSolver
    
    print("Testing structural analysis...")
    
    blade = create_baseline_blade()
    bem_solver = BEMSolver()
    struct_model = StructuralModel()
    
    # Get loads from BEM
    result = bem_solver.solve(blade, wind_speed=8.0, rotor_speed=12.1)
    
    # Structural analysis
    struct_result = struct_model.analyze(
        blade,
        loads_flap=result.normal_force,
        loads_edge=result.tangent_force
    )
    
    print(f"\nStructural Results:")
    print(f"  Tip deflection (flap): {struct_result.tip_deflection_flap:.3f} m")
    print(f"  Tip deflection (edge): {struct_result.tip_deflection_edge:.3f} m")
    print(f"  Deflection ratio: {struct_result.deflection_ratio:.4f} (limit: 0.05)")
    print(f"  Max stress (flap): {struct_result.max_stress_flap/1e6:.1f} MPa")
    print(f"  Max stress (edge): {struct_result.max_stress_edge/1e6:.1f} MPa")
    print(f"  Stress ratio: {struct_result.stress_ratio:.3f} (limit: 1.0)")
    print(f"  Blade mass: {struct_result.blade_mass:.0f} kg")
    print(f"  Feasible: {struct_result.is_feasible()}")
    
    # Test fatigue
    print("\nTesting fatigue analysis...")
    fatigue = FatigueAnalyzer()
    
    # Create simple load time series
    time = np.linspace(0, 100, 1000)
    loads = 1000 * np.sin(2 * np.pi * 0.5 * time)  # 0.5 Hz oscillation
    
    damage = fatigue.compute_damage(loads)
    print(f"  Fatigue damage: {damage:.4f} (limit: 1.0)")