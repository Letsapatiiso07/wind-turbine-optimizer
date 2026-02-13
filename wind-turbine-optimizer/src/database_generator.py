"""
Database Generator

Generates training database for ML surrogate models using Latin Hypercube Sampling.
"""

import numpy as np
from scipy.stats import qmc
import h5py
from tqdm import tqdm
import multiprocessing as mp
from typing import Dict, Tuple, Optional
import time

from parameterization import BladeGeometry, BladeParameters
from simulation import BEMSolver, AirfoilDatabase
from structural import StructuralModel, FatigueAnalyzer
from aep_calculator import AEPCalculator, create_weibull_distribution


class DatabaseGenerator:
    """
    Generate database of blade designs and performance metrics.
    
    Uses Latin Hypercube Sampling for efficient parameter space exploration.
    """
    
    def __init__(
        self,
        n_samples: int = 10000,
        n_processes: int = None,
        wind_distribution: Optional = None
    ):
        """
        Initialize database generator.
        
        Args:
            n_samples: Number of samples to generate
            n_processes: Number of parallel processes (default: CPU count)
            wind_distribution: Wind distribution for AEP calculation
        """
        self.n_samples = n_samples
        self.n_processes = n_processes or mp.cpu_count()
        
        # Default wind distribution if not provided
        if wind_distribution is None:
            wind_distribution = create_weibull_distribution(mean_speed=8.5, k=2.0)
        self.wind_dist = wind_distribution
        
        # Initialize solvers
        self.bem_solver = BEMSolver()
        self.struct_model = StructuralModel()
        self.aep_calc = AEPCalculator(wind_distribution)
    
    def generate_parameter_samples(
        self,
        param_bounds: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Generate parameter samples using Latin Hypercube Sampling.
        
        Args:
            param_bounds: Optional custom parameter bounds
            
        Returns:
            (n_samples, n_params) array of parameter samples
        """
        # Default parameter bounds
        if param_bounds is None:
            param_bounds = {
                'chord': (1.0, 5.0),      # 6 control points
                'twist': (0.0, 18.0),     # 6 control points
                'airfoil': (0, 2)         # 3 airfoil families
            }
        
        # Total parameters: 6 chord + 6 twist + 3 airfoil = 15
        n_chord = 6
        n_twist = 6
        n_airfoil = 3
        n_params = n_chord + n_twist + n_airfoil
        
        # Create Latin Hypercube Sampler
        sampler = qmc.LatinHypercube(d=n_params, seed=42)
        samples_unit = sampler.random(n=self.n_samples)
        
        # Scale to actual bounds
        samples = np.zeros_like(samples_unit)
        
        # Chord parameters (ensure monotonic decrease)
        for i in range(n_chord):
            # Linearly decrease expected value
            expected_chord = param_bounds['chord'][1] - i * (
                param_bounds['chord'][1] - param_bounds['chord'][0]
            ) / (n_chord - 1)
            
            # Add variation around expected value
            chord_range = (param_bounds['chord'][1] - param_bounds['chord'][0]) / n_chord
            samples[:, i] = expected_chord + (samples_unit[:, i] - 0.5) * chord_range
            
            # Clip to bounds
            samples[:, i] = np.clip(
                samples[:, i],
                param_bounds['chord'][0],
                param_bounds['chord'][1]
            )
        
        # Enforce monotonic decrease in chord
        for j in range(self.n_samples):
            for i in range(1, n_chord):
                if samples[j, i] > samples[j, i-1]:
                    samples[j, i] = samples[j, i-1] * 0.95
        
        # Twist parameters (decreasing from root to tip)
        for i in range(n_twist):
            expected_twist = param_bounds['twist'][1] - i * (
                param_bounds['twist'][1] - param_bounds['twist'][0]
            ) / (n_twist - 1)
            
            twist_range = (param_bounds['twist'][1] - param_bounds['twist'][0]) / n_twist
            samples[:, n_chord + i] = expected_twist + (samples_unit[:, n_chord + i] - 0.5) * twist_range
            
            samples[:, n_chord + i] = np.clip(
                samples[:, n_chord + i],
                param_bounds['twist'][0],
                param_bounds['twist'][1]
            )
        
        # Airfoil parameters (discrete 0, 1, 2)
        airfoil_start = n_chord + n_twist
        for i in range(n_airfoil):
            samples[:, airfoil_start + i] = np.round(
                samples_unit[:, airfoil_start + i] * 2
            ).astype(int)
        
        return samples
    
    def evaluate_design(
        self,
        params: np.ndarray
    ) -> Dict:
        """
        Evaluate a single blade design.
        
        Args:
            params: Design parameter vector
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            # Create blade geometry
            blade = BladeGeometry.from_array(params)
            
            # Validate constraints
            is_valid, violations = blade.validate_constraints()
            if not is_valid:
                # Repair if possible
                blade = blade.repair()
            
            # Compute power curve
            wind_speeds = np.linspace(3, 25, 23)
            power_curve = self.bem_solver.compute_power_curve(blade, wind_speeds)
            
            # Get operating point results (8 m/s, typical)
            bem_result = self.bem_solver.solve(blade, wind_speed=8.0, rotor_speed=12.1)
            
            # Structural analysis
            struct_result = self.struct_model.analyze(
                blade,
                loads_flap=bem_result.normal_force,
                loads_edge=bem_result.tangent_force
            )
            
            # AEP calculation
            aep_results = self.aep_calc.compute_aep(power_curve)
            
            # Compile outputs
            outputs = {
                # Global performance
                'aep_mwh': aep_results['aep_net_mwh'],
                'capacity_factor': aep_results['capacity_factor'],
                'max_power_kw': np.max(power_curve['power']),
                'max_cp': np.max(power_curve['cp']),
                
                # Operating point (8 m/s)
                'power_8ms': bem_result.power,
                'thrust_8ms': bem_result.thrust,
                'torque_8ms': bem_result.torque,
                'cp_8ms': bem_result.cp,
                
                # Structural
                'blade_mass_kg': struct_result.blade_mass,
                'tip_deflection_m': struct_result.tip_deflection_flap,
                'max_stress_pa': struct_result.max_stress_flap,
                'deflection_ratio': struct_result.deflection_ratio,
                'stress_ratio': struct_result.stress_ratio,
                
                # Constraints
                'is_feasible': int(struct_result.is_feasible()),
                
                # Root moments
                'flapwise_moment': bem_result.flapwise_moment,
                'edgewise_moment': bem_result.edgewise_moment
            }
            
            return outputs
        
        except Exception as e:
            print(f"Error evaluating design: {e}")
            # Return dummy/failed values
            return {
                'aep_mwh': 0,
                'capacity_factor': 0,
                'max_power_kw': 0,
                'max_cp': 0,
                'power_8ms': 0,
                'thrust_8ms': 0,
                'torque_8ms': 0,
                'cp_8ms': 0,
                'blade_mass_kg': 0,
                'tip_deflection_m': 0,
                'max_stress_pa': 0,
                'deflection_ratio': 1.0,
                'stress_ratio': 1.0,
                'is_feasible': 0,
                'flapwise_moment': 0,
                'edgewise_moment': 0
            }
    
    def generate_database(
        self,
        output_file: str,
        use_parallel: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complete database.
        
        Args:
            output_file: HDF5 file path to save database
            use_parallel: Use multiprocessing for parallel evaluation
            
        Returns:
            (input_array, output_array) with all samples
        """
        print(f"Generating database with {self.n_samples} samples...")
        print(f"Wind distribution: mean {self.wind_dist.mean_speed:.1f} m/s")
        
        # Generate parameter samples
        print("Generating parameter samples using LHS...")
        param_samples = self.generate_parameter_samples()
        
        # Evaluate all designs
        print(f"Evaluating designs...")
        start_time = time.time()
        
        if use_parallel and self.n_processes > 1:
            print(f"Using {self.n_processes} parallel processes")
            with mp.Pool(processes=self.n_processes) as pool:
                results = list(tqdm(
                    pool.imap(self.evaluate_design, param_samples),
                    total=self.n_samples
                ))
        else:
            print("Using sequential evaluation")
            results = [
                self.evaluate_design(params)
                for params in tqdm(param_samples)
            ]
        
        elapsed = time.time() - start_time
        print(f"Evaluation complete in {elapsed:.1f}s ({elapsed/self.n_samples:.3f}s per design)")
        
        # Convert results to arrays
        output_keys = list(results[0].keys())
        n_outputs = len(output_keys)
        
        output_array = np.zeros((self.n_samples, n_outputs))
        for i, result in enumerate(results):
            output_array[i] = [result[key] for key in output_keys]
        
        # Filter out failed evaluations
        valid_mask = output_array[:, output_keys.index('is_feasible')] > 0.5
        n_valid = np.sum(valid_mask)
        
        print(f"Valid designs: {n_valid}/{self.n_samples} ({n_valid/self.n_samples*100:.1f}%)")
        
        # Save to HDF5
        print(f"Saving database to {output_file}...")
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('inputs', data=param_samples)
            f.create_dataset('outputs', data=output_array)
            f.create_dataset('output_names', data=np.array(output_keys, dtype='S'))
            f.create_dataset('valid_mask', data=valid_mask)
            
            # Metadata
            f.attrs['n_samples'] = self.n_samples
            f.attrs['n_valid'] = n_valid
            f.attrs['mean_wind_speed'] = self.wind_dist.mean_speed
            f.attrs['generation_time'] = elapsed
        
        print(f"Database saved successfully!")
        
        return param_samples, output_array
    
    @staticmethod
    def load_database(filepath: str) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Load database from HDF5 file.
        
        Args:
            filepath: Path to HDF5 file
            
        Returns:
            (inputs, outputs, output_names)
        """
        with h5py.File(filepath, 'r') as f:
            inputs = f['inputs'][:]
            outputs = f['outputs'][:]
            output_names = [name.decode() for name in f['output_names'][:]]
            
            print(f"Loaded database:")
            print(f"  Samples: {len(inputs)}")
            print(f"  Input dims: {inputs.shape[1]}")
            print(f"  Output dims: {outputs.shape[1]}")
            print(f"  Valid samples: {f.attrs.get('n_valid', 'N/A')}")
        
        return inputs, outputs, output_names


if __name__ == "__main__":
    import os
    
    print("Testing database generator...")
    
    # Create small test database
    generator = DatabaseGenerator(n_samples=100, n_processes=2)
    
    # Generate database
    output_file = "data/test_database.h5"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    inputs, outputs = generator.generate_database(
        output_file,
        use_parallel=False  # Sequential for testing
    )
    
    # Load and verify
    inputs_loaded, outputs_loaded, names = DatabaseGenerator.load_database(output_file)
    
    print("\nDatabase statistics:")
    print(f"  Mean AEP: {outputs_loaded[:, names.index('aep_mwh')].mean():.1f} MWh")
    print(f"  Mean blade mass: {outputs_loaded[:, names.index('blade_mass_kg')].mean():.0f} kg")
    print(f"  Mean Cp: {outputs_loaded[:, names.index('max_cp')].mean():.3f}")
