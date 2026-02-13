import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BladeParameters:
    """Container for blade design parameters."""
    
    # Chord distribution control points (normalized span positions)
    chord_values: np.ndarray  # [m] at 5-7 spanwise locations
    
    # Twist distribution control points (normalized span positions)
    twist_values: np.ndarray  # [degrees] at 5-7 spanwise locations
    
    # Airfoil family indices (0=thick root, 1=mid, 2=thin tip)
    airfoil_indices: np.ndarray  # at 3-4 spanwise locations
    
    # Fixed parameters
    blade_span: float = 63.0  # [m] 5MW class
    hub_radius: float = 1.5   # [m]
    n_blades: int = 3
    
    # Control point locations (normalized 0-1)
    chord_stations: np.ndarray = None
    twist_stations: np.ndarray = None
    airfoil_stations: np.ndarray = None
    
    def __post_init__(self):
        """Set default station locations if not provided."""
        if self.chord_stations is None:
            n_chord = len(self.chord_values)
            self.chord_stations = np.linspace(0, 1, n_chord)
        
        if self.twist_stations is None:
            n_twist = len(self.twist_values)
            self.twist_stations = np.linspace(0, 1, n_twist)
        
        if self.airfoil_stations is None:
            n_airfoil = len(self.airfoil_indices)
            self.airfoil_stations = np.linspace(0, 1, n_airfoil)


class BladeGeometry:
    """
    Wind turbine blade geometry with smooth distributions.
    
    Attributes:
        params: BladeParameters object
        n_sections: Number of blade sections for analysis
        span_positions: Spanwise positions [m]
        radial_positions: Radial positions from hub center [m]
    """
    
    # Constraints
    MIN_CHORD = 0.1  # [m] minimum chord
    MAX_CHORD = 5.0  # [m] maximum chord
    MIN_TWIST = -5.0  # [deg] minimum twist
    MAX_TWIST = 25.0  # [deg] maximum twist
    
    def __init__(self, params: BladeParameters, n_sections: int = 30):
        """
        Initialize blade geometry.
        
        Args:
            params: Blade design parameters
            n_sections: Number of sections along span for analysis
        """
        self.params = params
        self.n_sections = n_sections
        
        # Create analysis stations
        self.span_positions = np.linspace(0, params.blade_span, n_sections)
        self.radial_positions = self.span_positions + params.hub_radius
        
        # Normalized span (0-1)
        self.span_normalized = self.span_positions / params.blade_span
        
        # Interpolate distributions
        self._interpolate_distributions()
    
    def _interpolate_distributions(self):
        """Interpolate chord, twist, and airfoil distributions using splines."""
        
        # Chord distribution (cubic spline for smoothness)
        chord_spline = CubicSpline(
            self.params.chord_stations,
            self.params.chord_values,
            bc_type='natural'
        )
        self.chord = chord_spline(self.span_normalized)
        
        # Twist distribution (cubic spline)
        twist_spline = CubicSpline(
            self.params.twist_stations,
            self.params.twist_values,
            bc_type='natural'
        )
        self.twist = twist_spline(self.span_normalized)
        
        # Airfoil distribution (nearest neighbor for discrete families)
        airfoil_interp = interp1d(
            self.params.airfoil_stations,
            self.params.airfoil_indices,
            kind='nearest',
            fill_value='extrapolate'
        )
        self.airfoil_family = airfoil_interp(self.span_normalized).astype(int)
    
    @classmethod
    def from_array(cls, param_array: np.ndarray, n_sections: int = 30) -> 'BladeGeometry':
        """
        Create blade geometry from flat parameter array.
        
        Args:
            param_array: Flattened parameters [chord_vals, twist_vals, airfoil_indices]
            n_sections: Number of analysis sections
            
        Returns:
            BladeGeometry instance
        """
        # Default: 6 chord points, 6 twist points, 3 airfoil transitions
        n_chord = 6
        n_twist = 6
        n_airfoil = 3
        
        chord_values = param_array[:n_chord]
        twist_values = param_array[n_chord:n_chord+n_twist]
        airfoil_indices = param_array[n_chord+n_twist:n_chord+n_twist+n_airfoil]
        
        # Round airfoil indices to integers
        airfoil_indices = np.round(airfoil_indices).astype(int)
        airfoil_indices = np.clip(airfoil_indices, 0, 2)  # 3 airfoil families
        
        params = BladeParameters(
            chord_values=chord_values,
            twist_values=twist_values,
            airfoil_indices=airfoil_indices
        )
        
        return cls(params, n_sections)
    
    def to_array(self) -> np.ndarray:
        """
        Convert blade parameters to flat array for optimization.
        
        Returns:
            Flattened parameter array
        """
        return np.concatenate([
            self.params.chord_values,
            self.params.twist_values,
            self.params.airfoil_indices.astype(float)
        ])
    
    def validate_constraints(self) -> Tuple[bool, List[str]]:
        """
        Validate geometric constraints.
        
        Returns:
            (is_valid, list of violations)
        """
        violations = []
        
        # Check chord bounds
        if np.any(self.chord < self.MIN_CHORD):
            violations.append(f"Chord below minimum ({self.MIN_CHORD}m)")
        if np.any(self.chord > self.MAX_CHORD):
            violations.append(f"Chord above maximum ({self.MAX_CHORD}m)")
        
        # Check twist bounds
        if np.any(self.twist < self.MIN_TWIST):
            violations.append(f"Twist below minimum ({self.MIN_TWIST}°)")
        if np.any(self.twist > self.MAX_TWIST):
            violations.append(f"Twist above maximum ({self.MAX_TWIST}°)")
        
        # Check monotonic chord decrease (manufacturing constraint)
        # Allow small violations due to spline smoothing
        chord_diff = np.diff(self.chord)
        if np.sum(chord_diff > 0.01) > len(chord_diff) * 0.2:  # >20% increasing
            violations.append("Chord not monotonically decreasing")
        
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def repair(self) -> 'BladeGeometry':
        """
        Repair constraint violations by clipping and smoothing.
        
        Returns:
            Repaired BladeGeometry instance
        """
        # Clip chord
        chord_repaired = np.clip(self.params.chord_values, self.MIN_CHORD, self.MAX_CHORD)
        
        # Enforce monotonic decrease in chord control points
        for i in range(1, len(chord_repaired)):
            if chord_repaired[i] > chord_repaired[i-1]:
                chord_repaired[i] = chord_repaired[i-1] * 0.95
        
        # Clip twist
        twist_repaired = np.clip(self.params.twist_values, self.MIN_TWIST, self.MAX_TWIST)
        
        # Create repaired parameters
        params_repaired = BladeParameters(
            chord_values=chord_repaired,
            twist_values=twist_repaired,
            airfoil_indices=self.params.airfoil_indices,
            blade_span=self.params.blade_span,
            hub_radius=self.params.hub_radius,
            n_blades=self.params.n_blades,
            chord_stations=self.params.chord_stations,
            twist_stations=self.params.twist_stations,
            airfoil_stations=self.params.airfoil_stations
        )
        
        return BladeGeometry(params_repaired, self.n_sections)
    
    def compute_properties(self) -> Dict[str, float]:
        """
        Compute derived blade properties.
        
        Returns:
            Dictionary of properties (area, aspect_ratio, etc.)
        """
        # Planform area (approximate)
        planform_area = np.trapz(self.chord, self.span_positions)
        
        # Aspect ratio
        aspect_ratio = self.params.blade_span / np.mean(self.chord)
        
        # Taper ratio
        taper_ratio = self.chord[-1] / self.chord[0]
        
        return {
            'planform_area': planform_area,
            'aspect_ratio': aspect_ratio,
            'taper_ratio': taper_ratio,
            'max_chord': np.max(self.chord),
            'min_chord': np.min(self.chord),
            'root_twist': self.twist[0],
            'tip_twist': self.twist[-1],
            'twist_range': self.twist[0] - self.twist[-1]
        }
    
    def get_section_data(self, station_idx: int) -> Dict:
        """
        Get geometric data for a specific blade section.
        
        Args:
            station_idx: Index of the station
            
        Returns:
            Dictionary with section geometry
        """
        return {
            'span': self.span_positions[station_idx],
            'radius': self.radial_positions[station_idx],
            'chord': self.chord[station_idx],
            'twist': self.twist[station_idx],
            'airfoil_family': self.airfoil_family[station_idx]
        }


def create_baseline_blade() -> BladeGeometry:
    """
    Create NREL 5MW baseline blade geometry.
    
    Returns:
        BladeGeometry for baseline design
    """
    # Approximate NREL 5MW blade (simplified)
    params = BladeParameters(
        chord_values=np.array([3.542, 4.167, 4.557, 4.652, 4.458, 3.891]),
        twist_values=np.array([13.308, 11.480, 10.162, 9.011, 7.795, 2.536]),
        airfoil_indices=np.array([0, 1, 2]),  # Thick to thin
        blade_span=63.0,
        hub_radius=1.5,
        n_blades=3
    )
    
    return BladeGeometry(params, n_sections=30)


def create_random_blade(
    seed: Optional[int] = None,
    bounds: Optional[Dict] = None
) -> BladeGeometry:
    """
    Create a random blade geometry for database generation.
    
    Args:
        seed: Random seed for reproducibility
        bounds: Optional parameter bounds
        
    Returns:
        Random BladeGeometry instance
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Default bounds
    if bounds is None:
        bounds = {
            'chord': (0.5, 5.0),
            'twist': (-2.0, 20.0)
        }
    
    # Generate random parameters
    n_chord = 6
    n_twist = 6
    
    # Chord: decreasing trend with noise
    chord_base = np.linspace(bounds['chord'][1], bounds['chord'][0], n_chord)
    chord_noise = np.random.uniform(-0.3, 0.3, n_chord)
    chord_values = chord_base + chord_noise
    chord_values = np.clip(chord_values, bounds['chord'][0], bounds['chord'][1])
    
    # Enforce monotonic decrease
    for i in range(1, n_chord):
        if chord_values[i] > chord_values[i-1]:
            chord_values[i] = chord_values[i-1] * 0.95
    
    # Twist: decreasing from root to tip
    twist_base = np.linspace(bounds['twist'][1], bounds['twist'][0], n_twist)
    twist_noise = np.random.uniform(-2.0, 2.0, n_twist)
    twist_values = twist_base + twist_noise
    twist_values = np.clip(twist_values, bounds['twist'][0], bounds['twist'][1])
    
    # Random airfoil distribution
    airfoil_indices = np.array([0, 1, 2])  # Fixed for now
    
    params = BladeParameters(
        chord_values=chord_values,
        twist_values=twist_values,
        airfoil_indices=airfoil_indices
    )
    
    return BladeGeometry(params)


if __name__ == "__main__":
    # Test the module
    print("Creating baseline blade...")
    baseline = create_baseline_blade()
    
    print("\nBaseline properties:")
    props = baseline.compute_properties()
    for key, value in props.items():
        print(f"  {key}: {value:.3f}")
    
    print("\nValidating constraints...")
    is_valid, violations = baseline.validate_constraints()
    print(f"  Valid: {is_valid}")
    if violations:
        for v in violations:
            print(f"  - {v}")
    
    print("\nCreating random blade...")
    random_blade = create_random_blade(seed=42)
    print("Random blade properties:")
    props_random = random_blade.compute_properties()
    for key, value in props_random.items():
        print(f"  {key}: {value:.3f}")