"""
Wind Turbine Blade Optimization System

A comprehensive ML-accelerated design optimization framework for wind turbine blades.
Combines physics-based simulation (BEM theory, structural analysis, fatigue) with
machine learning surrogate models and multi-algorithm optimization.
"""

__version__ = "1.0.0"
__author__ = "Wind Energy Optimization Team"
__description__ = "ML-accelerated wind turbine blade design optimization"

# Core parameterization
from .parameterization import (
    BladeGeometry,
    BladeParameters,
    create_baseline_blade,
    create_random_blade
)

# Simulation components
from .simulation import (
    BEMSolver,
    BEMResult,
    AirfoilDatabase,
    AirfoilPolar
)

# Structural analysis
from .structural import (
    StructuralModel,
    StructuralResult,
    MaterialProperties,
    FatigueAnalyzer,
    SNCurve
)

# AEP calculation
from .aep_calculator import (
    AEPCalculator,
    WindDistribution,
    create_weibull_distribution,
    load_wind_rose_from_csv,
    compare_designs
)

# Machine learning
from .ml_models import (
    SurrogateEnsemble,
    NeuralNetSurrogate
)

# Optimization
from .optimization import (
    GeneticAlgorithmOptimizer,
    PSOOptimizer,
    GradientOptimizer,
    OptimizationResult
)

# Visualization
from .visualization import (
    BladeVisualizer
)

# Database generation
from .database_generator import (
    DatabaseGenerator
)

__all__ = [
    # Parameterization
    'BladeGeometry',
    'BladeParameters',
    'create_baseline_blade',
    'create_random_blade',
    
    # Simulation
    'BEMSolver',
    'BEMResult',
    'AirfoilDatabase',
    'AirfoilPolar',
    
    # Structural
    'StructuralModel',
    'StructuralResult',
    'MaterialProperties',
    'FatigueAnalyzer',
    'SNCurve',
    
    # AEP
    'AEPCalculator',
    'WindDistribution',
    'create_weibull_distribution',
    'load_wind_rose_from_csv',
    'compare_designs',
    
    # ML
    'SurrogateEnsemble',
    'NeuralNetSurrogate',
    
    # Optimization
    'GeneticAlgorithmOptimizer',
    'PSOOptimizer',
    'GradientOptimizer',
    'OptimizationResult',
    
    # Visualization
    'BladeVisualizer',
    
    # Database
    'DatabaseGenerator',
]

# Package-level info
def get_version():
    """Return the version string."""
    return __version__

def get_info():
    """Return package information."""
    return {
        'name': 'wind-turbine-optimizer',
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'modules': len(__all__),
        'features': [
            'BEM aerodynamic simulation',
            'Structural analysis with fatigue',
            'ML surrogate models (500× speedup)',
            'Multi-algorithm optimization (GA, PSO, gradient)',
            'Interactive Streamlit dashboard',
            'Comprehensive visualization',
            'Real-world wind data integration'
        ]
    }

# Print info when imported with verbose flag
if __name__ != "__main__":
    pass  # Silent import
else:
    # If run as script, print package info
    info = get_info()
    print(f"\n{info['name']} v{info['version']}")
    print(f"{info['description']}\n")
    print("Available modules:")
    for item in __all__:
        print(f"  - {item}")
    print(f"\nTotal: {info['modules']} public components")
    print("\nFeatures:")
    for feature in info['features']:
        print(f"  ✓ {feature}")
