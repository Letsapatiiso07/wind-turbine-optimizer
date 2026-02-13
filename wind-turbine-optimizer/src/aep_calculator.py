"""
Annual Energy Production (AEP) Calculator

Computes AEP from power curve and wind distribution.
Includes wind rose data handling and Weibull distribution utilities.
"""

import numpy as np
import pandas as pd
from scipy.stats import weibull_min
from scipy.special import gamma
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class WindDistribution:
    """Wind speed distribution data."""
    
    wind_speeds: np.ndarray     # [m/s] bin centers
    probabilities: np.ndarray   # Probability density
    mean_speed: float           # [m/s] mean wind speed
    distribution_type: str = "weibull"  # or "measured"
    
    # Weibull parameters (if applicable)
    k: Optional[float] = None   # Shape parameter
    A: Optional[float] = None   # Scale parameter [m/s]
    
    def __post_init__(self):
        """Normalize probabilities."""
        # Ensure probabilities sum to 1
        self.probabilities = self.probabilities / np.sum(self.probabilities)


def create_weibull_distribution(
    mean_speed: float,
    k: float = 2.0,
    wind_speeds: Optional[np.ndarray] = None
) -> WindDistribution:
    """
    Create Weibull wind speed distribution.
    
    The Weibull distribution is commonly used to model wind speeds.
    f(v) = (k/A) * (v/A)^(k-1) * exp(-(v/A)^k)
    
    Args:
        mean_speed: Mean wind speed [m/s]
        k: Shape parameter (k=2 is Rayleigh distribution)
        wind_speeds: Wind speed bins (default: 0-30 m/s in 0.5 m/s steps)
        
    Returns:
        WindDistribution object
    """
    if wind_speeds is None:
        wind_speeds = np.arange(0.5, 30.5, 0.5)
    
    # Calculate scale parameter A from mean speed
    # Mean of Weibull: μ = A * Γ(1 + 1/k)
    A = mean_speed / gamma(1 + 1/k)
    
    # Compute probability density
    probabilities = weibull_min.pdf(wind_speeds, k, scale=A)
    
    # Convert to discrete probabilities (multiply by bin width)
    bin_width = wind_speeds[1] - wind_speeds[0]
    probabilities = probabilities * bin_width
    
    return WindDistribution(
        wind_speeds=wind_speeds,
        probabilities=probabilities,
        mean_speed=mean_speed,
        distribution_type="weibull",
        k=k,
        A=A
    )


def load_wind_rose_from_csv(filepath: str) -> WindDistribution:
    """
    Load wind rose data from CSV file.
    
    Expected format:
    - Column 'wind_speed' or 'speed': Wind speed [m/s]
    - Column 'frequency' or 'probability': Occurrence frequency
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        WindDistribution object
    """
    df = pd.read_csv(filepath)
    
    # Try to find wind speed column
    speed_col = None
    for col in ['wind_speed', 'speed', 'WindSpeed', 'v']:
        if col in df.columns:
            speed_col = col
            break
    
    if speed_col is None:
        raise ValueError("Could not find wind speed column in CSV")
    
    # Try to find probability column
    prob_col = None
    for col in ['frequency', 'probability', 'prob', 'occurrence']:
        if col in df.columns:
            prob_col = col
            break
    
    if prob_col is None:
        raise ValueError("Could not find probability column in CSV")
    
    wind_speeds = df[speed_col].values
    probabilities = df[prob_col].values
    
    # Compute mean speed
    mean_speed = np.sum(wind_speeds * probabilities) / np.sum(probabilities)
    
    return WindDistribution(
        wind_speeds=wind_speeds,
        probabilities=probabilities,
        mean_speed=mean_speed,
        distribution_type="measured"
    )


def create_sample_wind_rose(
    site_class: str = "IEC_II",
    return_csv: bool = False
) -> WindDistribution:
    """
    Create sample wind rose for testing.
    
    Args:
        site_class: IEC wind class ('IEC_I', 'IEC_II', 'IEC_III')
        return_csv: If True, return path to saved CSV
        
    Returns:
        WindDistribution object (or path if return_csv=True)
    """
    # IEC wind class average speeds
    mean_speeds = {
        'IEC_I': 10.0,    # High wind
        'IEC_II': 8.5,    # Medium wind
        'IEC_III': 7.5    # Low wind
    }
    
    mean_speed = mean_speeds.get(site_class, 8.5)
    
    # Create Weibull distribution
    distribution = create_weibull_distribution(mean_speed, k=2.0)
    
    if return_csv:
        # Save to CSV for testing
        import os
        csv_path = f"data/wind_roses/sample_{site_class}.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        df = pd.DataFrame({
            'wind_speed': distribution.wind_speeds,
            'frequency': distribution.probabilities
        })
        df.to_csv(csv_path, index=False)
        return csv_path
    
    return distribution


class AEPCalculator:
    """
    Annual Energy Production calculator.
    
    Computes AEP from power curve and wind distribution using:
    AEP = 8760 * Σ[P(v) * f(v) * Δv]
    """
    
    HOURS_PER_YEAR = 8760
    
    def __init__(
        self,
        wind_distribution: WindDistribution,
        availability: float = 0.95,  # 95% availability
        array_losses: float = 0.05   # 5% wake losses
    ):
        """
        Initialize AEP calculator.
        
        Args:
            wind_distribution: Wind speed distribution
            availability: Turbine availability (0-1)
            array_losses: Array/wake losses (0-1)
        """
        self.wind_dist = wind_distribution
        self.availability = availability
        self.array_losses = array_losses
        self.loss_factor = availability * (1 - array_losses)
    
    def compute_aep(
        self,
        power_curve: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute Annual Energy Production.
        
        Args:
            power_curve: Dict with 'wind_speed' and 'power' arrays
            
        Returns:
            Dictionary with AEP metrics
        """
        # Interpolate power curve to wind distribution bins
        power_interp = np.interp(
            self.wind_dist.wind_speeds,
            power_curve['wind_speed'],
            power_curve['power']
        )
        
        # Compute gross AEP
        # AEP = Σ[P(v) * f(v) * hours_per_year]
        aep_gross = self.HOURS_PER_YEAR * np.sum(
            power_interp * self.wind_dist.probabilities
        )
        
        # Net AEP with losses
        aep_net = aep_gross * self.loss_factor
        
        # Capacity factor
        rated_power = np.max(power_curve['power'])
        capacity_factor = aep_net / (rated_power * self.HOURS_PER_YEAR)
        
        # Average power
        avg_power = aep_net / self.HOURS_PER_YEAR
        
        return {
            'aep_gross_mwh': aep_gross / 1000,  # Convert to MWh
            'aep_net_mwh': aep_net / 1000,
            'capacity_factor': capacity_factor,
            'avg_power_kw': avg_power,
            'rated_power_kw': rated_power,
            'mean_wind_speed': self.wind_dist.mean_speed
        }
    
    def compute_aep_sensitivity(
        self,
        power_curve: Dict[str, np.ndarray],
        wind_speed_variations: np.ndarray = np.array([-10, -5, 0, 5, 10])
    ) -> pd.DataFrame:
        """
        Compute AEP sensitivity to wind speed variations.
        
        Args:
            power_curve: Power curve dictionary
            wind_speed_variations: Percent variations in mean wind speed
            
        Returns:
            DataFrame with sensitivity results
        """
        results = []
        
        for variation in wind_speed_variations:
            # Create modified wind distribution
            new_mean = self.wind_dist.mean_speed * (1 + variation / 100)
            
            if self.wind_dist.distribution_type == "weibull":
                modified_dist = create_weibull_distribution(
                    new_mean,
                    k=self.wind_dist.k
                )
            else:
                # Scale existing distribution
                modified_dist = WindDistribution(
                    wind_speeds=self.wind_dist.wind_speeds * (1 + variation / 100),
                    probabilities=self.wind_dist.probabilities,
                    mean_speed=new_mean,
                    distribution_type=self.wind_dist.distribution_type
                )
            
            # Compute AEP
            calc = AEPCalculator(modified_dist, self.availability, self.array_losses)
            aep_results = calc.compute_aep(power_curve)
            
            results.append({
                'variation_percent': variation,
                'mean_wind_speed': new_mean,
                'aep_net_mwh': aep_results['aep_net_mwh'],
                'capacity_factor': aep_results['capacity_factor']
            })
        
        return pd.DataFrame(results)


def compare_designs(
    baseline_power_curve: Dict,
    optimized_power_curve: Dict,
    wind_distribution: WindDistribution
) -> Dict:
    """
    Compare baseline and optimized designs.
    
    Args:
        baseline_power_curve: Baseline power curve
        optimized_power_curve: Optimized power curve
        wind_distribution: Wind distribution
        
    Returns:
        Dictionary with comparison metrics
    """
    calc = AEPCalculator(wind_distribution)
    
    baseline_aep = calc.compute_aep(baseline_power_curve)
    optimized_aep = calc.compute_aep(optimized_power_curve)
    
    aep_improvement = (
        (optimized_aep['aep_net_mwh'] - baseline_aep['aep_net_mwh']) /
        baseline_aep['aep_net_mwh'] * 100
    )
    
    cf_improvement = (
        (optimized_aep['capacity_factor'] - baseline_aep['capacity_factor']) /
        baseline_aep['capacity_factor'] * 100
    )
    
    return {
        'baseline_aep_mwh': baseline_aep['aep_net_mwh'],
        'optimized_aep_mwh': optimized_aep['aep_net_mwh'],
        'aep_improvement_percent': aep_improvement,
        'baseline_cf': baseline_aep['capacity_factor'],
        'optimized_cf': optimized_aep['capacity_factor'],
        'cf_improvement_percent': cf_improvement,
        'baseline_avg_power': baseline_aep['avg_power_kw'],
        'optimized_avg_power': optimized_aep['avg_power_kw']
    }


if __name__ == "__main__":
    # Test AEP calculator
    print("Testing AEP calculator...")
    
    # Create sample wind distribution
    wind_dist = create_weibull_distribution(mean_speed=8.5, k=2.0)
    print(f"\nWind Distribution:")
    print(f"  Mean speed: {wind_dist.mean_speed:.2f} m/s")
    print(f"  Weibull k: {wind_dist.k:.2f}")
    print(f"  Weibull A: {wind_dist.A:.2f}")
    
    # Create sample power curve
    wind_speeds = np.linspace(3, 25, 23)
    # Simple power curve model
    power = np.zeros_like(wind_speeds)
    rated_speed = 11.4
    rated_power = 5000.0
    
    for i, v in enumerate(wind_speeds):
        if v < 3:
            power[i] = 0
        elif v < rated_speed:
            power[i] = rated_power * ((v - 3) / (rated_speed - 3))**3
        elif v < 25:
            power[i] = rated_power
        else:
            power[i] = 0
    
    power_curve = {'wind_speed': wind_speeds, 'power': power}
    
    # Calculate AEP
    calc = AEPCalculator(wind_dist)
    results = calc.compute_aep(power_curve)
    
    print(f"\nAEP Results:")
    print(f"  Gross AEP: {results['aep_gross_mwh']:.0f} MWh/year")
    print(f"  Net AEP: {results['aep_net_mwh']:.0f} MWh/year")
    print(f"  Capacity factor: {results['capacity_factor']:.1%}")
    print(f"  Average power: {results['avg_power_kw']:.0f} kW")
    
    # Sensitivity analysis
    print("\nSensitivity Analysis:")
    sensitivity = calc.compute_aep_sensitivity(power_curve)
    print(sensitivity.to_string(index=False))
