"""
Main Pipeline for Wind Turbine Blade Optimization

Integrates all components: parameterization, simulation, ML surrogate, optimization.
"""

import numpy as np
import argparse
from pathlib import Path
import json
import time
from typing import Dict, Optional

from parameterization import BladeGeometry, create_baseline_blade
from simulation import BEMSolver
from structural import StructuralModel
from aep_calculator import AEPCalculator, create_weibull_distribution, compare_designs
from ml_models import SurrogateEnsemble
from optimization import GeneticAlgorithmOptimizer, PSOOptimizer
from visualization import BladeVisualizer
from database_generator import DatabaseGenerator


class WindTurbineOptimizer:
    """
    Complete optimization pipeline for wind turbine blades.
    """
    
    def __init__(
        self,
        wind_distribution=None,
        results_dir: str = "results",
        verbose: bool = True
    ):
        """
        Initialize optimizer.
        
        Args:
            wind_distribution: Wind distribution for AEP
            results_dir: Directory for results
            verbose: Print progress
        """
        # Create wind distribution
        if wind_distribution is None:
            wind_distribution = create_weibull_distribution(mean_speed=8.5, k=2.0)
        self.wind_dist = wind_distribution
        
        # Initialize components
        self.bem_solver = BEMSolver()
        self.struct_model = StructuralModel()
        self.aep_calc = AEPCalculator(wind_distribution)
        self.visualizer = BladeVisualizer(save_dir=f"{results_dir}/figures")
        
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.verbose = verbose
        
        # Surrogate model
        self.surrogate = None
        
        # Results storage
        self.baseline_results = None
        self.optimized_results = None
    
    def evaluate_blade_design(
        self,
        blade_geometry: BladeGeometry,
        return_detailed: bool = False
    ) -> Dict:
        """
        Evaluate a blade design completely.
        
        Args:
            blade_geometry: BladeGeometry to evaluate
            return_detailed: If True, return full BEM results
            
        Returns:
            Dictionary of performance metrics
        """
        # Compute power curve
        wind_speeds = np.linspace(3, 25, 23)
        power_curve = self.bem_solver.compute_power_curve(blade_geometry, wind_speeds)
        
        # AEP
        aep_results = self.aep_calc.compute_aep(power_curve)
        
        # Operating point analysis (8 m/s)
        bem_result = self.bem_solver.solve(blade_geometry, wind_speed=8.0, rotor_speed=12.1)
        
        # Structural analysis
        struct_result = self.struct_model.analyze(
            blade_geometry,
            loads_flap=bem_result.normal_force,
            loads_edge=bem_result.tangent_force
        )
        
        results = {
            'aep_mwh': aep_results['aep_net_mwh'],
            'capacity_factor': aep_results['capacity_factor'],
            'blade_mass_kg': struct_result.blade_mass,
            'max_cp': np.max(power_curve['cp']),
            'power_curve': power_curve,
            'deflection_ratio': struct_result.deflection_ratio,
            'stress_ratio': struct_result.stress_ratio,
            'is_feasible': struct_result.is_feasible()
        }
        
        if return_detailed:
            results.update({
                'bem_result': bem_result,
                'struct_result': struct_result,
                'aep_full': aep_results
            })
        
        return results
    
    def evaluate_baseline(self) -> Dict:
        """Evaluate NREL 5MW baseline design."""
        if self.verbose:
            print("\n" + "="*60)
            print("Evaluating Baseline Design (NREL 5MW)")
            print("="*60)
        
        baseline_blade = create_baseline_blade()
        self.baseline_results = self.evaluate_blade_design(baseline_blade, return_detailed=True)
        
        if self.verbose:
            print(f"\nBaseline Results:")
            print(f"  AEP: {self.baseline_results['aep_mwh']:.1f} MWh/year")
            print(f"  Capacity Factor: {self.baseline_results['capacity_factor']:.1%}")
            print(f"  Blade Mass: {self.baseline_results['blade_mass_kg']:.0f} kg")
            print(f"  Max Cp: {self.baseline_results['max_cp']:.3f}")
            print(f"  Feasible: {self.baseline_results['is_feasible']}")
        
        return self.baseline_results
    
    def train_surrogate(
        self,
        database_file: Optional[str] = None,
        n_samples: int = 10000,
        force_regenerate: bool = False
    ):
        """
        Train ML surrogate model.
        
        Args:
            database_file: Path to database file (or generate new)
            n_samples: Number of samples if generating new database
            force_regenerate: Force regeneration even if file exists
        """
        if self.verbose:
            print("\n" + "="*60)
            print("Training ML Surrogate Model")
            print("="*60)
        
        # Database path
        if database_file is None:
            database_file = self.results_dir / "training_database.h5"
        
        # Check if database exists
        if not Path(database_file).exists() or force_regenerate:
            if self.verbose:
                print(f"\nGenerating training database ({n_samples} samples)...")
            
            generator = DatabaseGenerator(
                n_samples=n_samples,
                wind_distribution=self.wind_dist
            )
            generator.generate_database(str(database_file), use_parallel=True)
        
        # Load database
        if self.verbose:
            print(f"\nLoading database from {database_file}...")
        
        inputs, outputs, output_names = DatabaseGenerator.load_database(str(database_file))
        
        # Filter valid samples
        is_feasible_idx = output_names.index('is_feasible')
        valid_mask = outputs[:, is_feasible_idx] > 0.5
        
        inputs_valid = inputs[valid_mask]
        outputs_valid = outputs[valid_mask]
        
        if self.verbose:
            print(f"Using {len(inputs_valid)} valid samples for training")
        
        # Select key outputs for surrogate
        # AEP, Cp, mass, deflection_ratio, stress_ratio
        output_indices = [
            output_names.index('aep_mwh'),
            output_names.index('max_cp'),
            output_names.index('blade_mass_kg'),
            output_names.index('deflection_ratio'),
            output_names.index('stress_ratio')
        ]
        
        outputs_selected = outputs_valid[:, output_indices]
        
        # Train/val split
        n_train = int(0.8 * len(inputs_valid))
        X_train = inputs_valid[:n_train]
        y_train = outputs_selected[:n_train]
        X_val = inputs_valid[n_train:]
        y_val = outputs_selected[n_train:]
        
        # Create and train surrogate
        if self.verbose:
            print("\nTraining ensemble surrogate...")
        
        self.surrogate = SurrogateEnsemble(
            n_features=inputs.shape[1],
            n_outputs=len(output_indices)
        )
        
        history = self.surrogate.train(
            X_train, y_train,
            X_val, y_val,
            epochs=200,
            batch_size=256,
            verbose=self.verbose
        )
        
        # Evaluate
        metrics = self.surrogate.evaluate(X_val, y_val)
        
        if self.verbose:
            print(f"\nSurrogate Performance:")
            print(f"  R² Score: {metrics['r2_mean']:.4f}")
            print(f"  RMSE: {metrics['rmse_mean']:.2f}")
            print(f"  MAPE: {metrics['mape_mean']:.2f}%")
        
        # Save model
        model_path = self.results_dir / "surrogate_model.pkl"
        self.surrogate.save(str(model_path))
        
        if self.verbose:
            print(f"\nModel saved to {model_path}")
    
    def optimize(
        self,
        method: str = 'GA',
        use_surrogate: bool = True,
        population_size: int = 100,
        n_generations: int = 500
    ) -> Dict:
        """
        Optimize blade design.
        
        Args:
            method: Optimization method ('GA', 'PSO')
            use_surrogate: Use ML surrogate (much faster)
            population_size: Population size for GA/PSO
            n_generations: Number of generations/iterations
            
        Returns:
            Optimized design results
        """
        if self.verbose:
            print("\n" + "="*60)
            print(f"Running Optimization ({method})")
            print("="*60)
            print(f"  Method: {method}")
            print(f"  Use surrogate: {use_surrogate}")
            print(f"  Population: {population_size}")
            print(f"  Generations: {n_generations}")
        
        # Define parameter bounds
        n_params = 15  # 6 chord + 6 twist + 3 airfoil
        param_bounds = np.array([
            # Chord (6 points, decreasing)
            [2.0, 5.0],
            [1.8, 4.5],
            [1.5, 4.0],
            [1.2, 3.5],
            [0.8, 3.0],
            [0.5, 2.0],
            # Twist (6 points, decreasing)
            [12.0, 20.0],
            [10.0, 16.0],
            [8.0, 14.0],
            [6.0, 12.0],
            [3.0, 8.0],
            [0.0, 5.0],
            # Airfoil (3 points, discrete)
            [0, 2],
            [0, 2],
            [0, 2]
        ])
        
        # Define objective function
        if use_surrogate:
            if self.surrogate is None:
                raise ValueError("Surrogate not trained! Call train_surrogate() first.")
            
            def objective(params):
                """Objective: maximize AEP (minimize negative AEP)."""
                # Predict [aep, cp, mass, deflection_ratio, stress_ratio]
                predictions, _ = self.surrogate.predict(params.reshape(1, -1))
                aep = predictions[0, 0]
                return -aep  # Minimize negative AEP
            
            def constraints(params):
                """Constraints: deflection and stress ratios."""
                predictions, _ = self.surrogate.predict(params.reshape(1, -1))
                deflection_ratio = predictions[0, 3]
                stress_ratio = predictions[0, 4]
                
                # Return violations (should be <= 0)
                return np.array([
                    deflection_ratio - 0.05,  # Max 5% deflection
                    stress_ratio - 1.0        # Stress below allowable
                ])
        
        else:
            def objective(params):
                """Direct evaluation objective."""
                blade = BladeGeometry.from_array(params)
                blade = blade.repair()  # Ensure valid
                results = self.evaluate_blade_design(blade)
                if not results['is_feasible']:
                    return 1e6  # Penalty
                return -results['aep_mwh']
            
            def constraints(params):
                blade = BladeGeometry.from_array(params)
                blade = blade.repair()
                results = self.evaluate_blade_design(blade)
                return np.array([
                    results['deflection_ratio'] - 0.05,
                    results['stress_ratio'] - 1.0
                ])
        
        # Run optimization
        if method.upper() == 'GA':
            optimizer = GeneticAlgorithmOptimizer(
                population_size=population_size,
                n_generations=n_generations,
                verbose=self.verbose
            )
        elif method.upper() == 'PSO':
            optimizer = PSOOptimizer(
                swarm_size=population_size,
                max_iterations=n_generations,
                verbose=self.verbose
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        opt_result = optimizer.optimize(
            objective,
            param_bounds,
            constraints
        )
        
        # Evaluate optimized design with full simulation
        if self.verbose:
            print("\n" + "="*60)
            print("Evaluating Optimized Design")
            print("="*60)
        
        optimized_blade = BladeGeometry.from_array(opt_result.best_params)
        self.optimized_results = self.evaluate_blade_design(optimized_blade, return_detailed=True)
        self.optimized_results['optimization'] = opt_result
        self.optimized_results['blade_geometry'] = optimized_blade
        
        if self.verbose:
            print(f"\nOptimized Results:")
            print(f"  AEP: {self.optimized_results['aep_mwh']:.1f} MWh/year")
            print(f"  Capacity Factor: {self.optimized_results['capacity_factor']:.1%}")
            print(f"  Blade Mass: {self.optimized_results['blade_mass_kg']:.0f} kg")
            print(f"  Max Cp: {self.optimized_results['max_cp']:.3f}")
            print(f"  Feasible: {self.optimized_results['is_feasible']}")
        
        return self.optimized_results
    
    def generate_report(self):
        """Generate comprehensive comparison report."""
        if self.baseline_results is None or self.optimized_results is None:
            raise ValueError("Must evaluate baseline and run optimization first!")
        
        if self.verbose:
            print("\n" + "="*60)
            print("Generating Report")
            print("="*60)
        
        # Compare designs
        comparison = compare_designs(
            self.baseline_results['power_curve'],
            self.optimized_results['power_curve'],
            self.wind_dist
        )
        
        # Create visualizations
        baseline_blade = create_baseline_blade()
        optimized_blade = self.optimized_results['blade_geometry']
        
        # Geometry comparison
        self.visualizer.plot_blade_geometry(baseline_blade, "Baseline Geometry")
        self.visualizer.plot_blade_geometry(optimized_blade, "Optimized Geometry")
        
        # Power curves
        self.visualizer.plot_power_curve(
            self.optimized_results['power_curve'],
            self.baseline_results['power_curve'],
            "Power Curve Comparison"
        )
        
        # Cp-TSR
        self.visualizer.plot_cp_tsr(self.optimized_results['power_curve'])
        
        # Save comparison metrics
        report = {
            'baseline': {
                'aep_mwh': self.baseline_results['aep_mwh'],
                'capacity_factor': self.baseline_results['capacity_factor'],
                'blade_mass_kg': self.baseline_results['blade_mass_kg'],
                'max_cp': self.baseline_results['max_cp']
            },
            'optimized': {
                'aep_mwh': self.optimized_results['aep_mwh'],
                'capacity_factor': self.optimized_results['capacity_factor'],
                'blade_mass_kg': self.optimized_results['blade_mass_kg'],
                'max_cp': self.optimized_results['max_cp']
            },
            'comparison': comparison,
            'optimization_time': self.optimized_results['optimization'].total_time,
            'n_evaluations': self.optimized_results['optimization'].n_evaluations
        }
        
        # Save report
        report_path = self.results_dir / "optimization_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("OPTIMIZATION SUMMARY")
            print(f"{'='*60}")
            print(f"\nAEP Improvement: {comparison['aep_improvement_percent']:+.2f}%")
            print(f"CF Improvement: {comparison['cf_improvement_percent']:+.2f}%")
            print(f"Mass Change: {(self.optimized_results['blade_mass_kg'] - self.baseline_results['blade_mass_kg']) / self.baseline_results['blade_mass_kg'] * 100:+.2f}%")
            print(f"\nOptimization Time: {report['optimization_time']:.1f} seconds")
            print(f"Function Evaluations: {report['n_evaluations']}")
            print(f"\nReport saved to: {report_path}")
            print(f"Figures saved to: {self.visualizer.save_dir}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Wind Turbine Blade Optimizer")
    parser.add_argument('--wind-speed', type=float, default=8.5,
                        help='Mean wind speed [m/s]')
    parser.add_argument('--n-samples', type=int, default=5000,
                        help='Database samples for surrogate training')
    parser.add_argument('--method', type=str, default='GA',
                        choices=['GA', 'PSO'],
                        help='Optimization method')
    parser.add_argument('--population', type=int, default=100,
                        help='Population size')
    parser.add_argument('--generations', type=int, default=300,
                        help='Number of generations')
    parser.add_argument('--skip-surrogate', action='store_true',
                        help='Skip surrogate training (use direct evaluation)')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Results directory')
    
    args = parser.parse_args()
    
    # Create wind distribution
    wind_dist = create_weibull_distribution(mean_speed=args.wind_speed, k=2.0)
    
    # Initialize optimizer
    optimizer = WindTurbineOptimizer(
        wind_distribution=wind_dist,
        results_dir=args.results_dir,
        verbose=True
    )
    
    # Run pipeline
    optimizer.evaluate_baseline()
    
    if not args.skip_surrogate:
        optimizer.train_surrogate(n_samples=args.n_samples)
    
    optimizer.optimize(
        method=args.method,
        use_surrogate=not args.skip_surrogate,
        population_size=args.population,
        n_generations=args.generations
    )
    
    optimizer.generate_report()
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
