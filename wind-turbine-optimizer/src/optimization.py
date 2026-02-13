"""
Optimization Module

Implements Genetic Algorithm (NSGA-II), Particle Swarm Optimization,
and gradient-based methods for blade design optimization.
"""

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from scipy.optimize import differential_evolution, minimize as scipy_minimize
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import time


@dataclass
class OptimizationResult:
    """Results from optimization."""
    
    best_params: np.ndarray      # Best parameter vector
    best_fitness: float          # Best fitness value
    best_objectives: np.ndarray  # Multi-objective values
    history: List[Dict]          # Optimization history
    convergence_iter: int        # Iteration of convergence
    total_time: float           # Total optimization time
    n_evaluations: int          # Number of function evaluations
    
    # For multi-objective
    pareto_front: Optional[np.ndarray] = None
    pareto_params: Optional[np.ndarray] = None


class BladeOptimizationProblem(Problem):
    """
    Optimization problem wrapper for PyMOO.
    
    Handles objective evaluation, constraint checking, and parameter bounds.
    """
    
    def __init__(
        self,
        objective_func: Callable,
        n_params: int,
        param_bounds: np.ndarray,
        n_objectives: int = 1,
        constraint_func: Optional[Callable] = None
    ):
        """
        Initialize optimization problem.
        
        Args:
            objective_func: Function(params) -> objectives
            n_params: Number of design parameters
            param_bounds: (n_params, 2) array of [lower, upper] bounds
            n_objectives: Number of objectives (1 for single, >1 for multi)
            constraint_func: Optional function(params) -> constraints
        """
        self.objective_func = objective_func
        self.constraint_func = constraint_func
        
        n_constraints = 0
        if constraint_func is not None:
            # Dummy evaluation to get number of constraints
            dummy_params = (param_bounds[:, 0] + param_bounds[:, 1]) / 2
            dummy_constraints = constraint_func(dummy_params)
            n_constraints = len(dummy_constraints) if isinstance(dummy_constraints, (list, np.ndarray)) else 1
        
        super().__init__(
            n_var=n_params,
            n_obj=n_objectives,
            n_constr=n_constraints,
            xl=param_bounds[:, 0],
            xu=param_bounds[:, 1]
        )
        
        self.n_evaluations = 0
    
    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate objectives and constraints for population X."""
        
        n_pop = X.shape[0]
        objectives = np.zeros((n_pop, self.n_obj))
        
        if self.n_constr > 0:
            constraints = np.zeros((n_pop, self.n_constr))
        
        for i in range(n_pop):
            # Evaluate objectives
            obj_vals = self.objective_func(X[i])
            if isinstance(obj_vals, (int, float)):
                objectives[i, 0] = obj_vals
            else:
                objectives[i] = obj_vals
            
            # Evaluate constraints
            if self.n_constr > 0:
                constr_vals = self.constraint_func(X[i])
                if isinstance(constr_vals, (int, float)):
                    constraints[i, 0] = constr_vals
                else:
                    constraints[i] = constr_vals
            
            self.n_evaluations += 1
        
        out["F"] = objectives
        if self.n_constr > 0:
            out["G"] = constraints


class GeneticAlgorithmOptimizer:
    """Genetic Algorithm optimizer using NSGA-II."""
    
    def __init__(
        self,
        population_size: int = 100,
        n_generations: int = 500,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        verbose: bool = True
    ):
        """
        Initialize GA optimizer.
        
        Args:
            population_size: Size of population
            n_generations: Number of generations
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability
            verbose: Print progress
        """
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.verbose = verbose
    
    def optimize(
        self,
        objective_func: Callable,
        param_bounds: np.ndarray,
        constraint_func: Optional[Callable] = None,
        n_objectives: int = 1
    ) -> OptimizationResult:
        """
        Run genetic algorithm optimization.
        
        Args:
            objective_func: Objective function
            param_bounds: Parameter bounds (n_params, 2)
            constraint_func: Constraint function
            n_objectives: Number of objectives
            
        Returns:
            OptimizationResult object
        """
        n_params = param_bounds.shape[0]
        
        # Create problem
        problem = BladeOptimizationProblem(
            objective_func,
            n_params,
            param_bounds,
            n_objectives,
            constraint_func
        )
        
        # Create algorithm
        algorithm = NSGA2(
            pop_size=self.population_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=self.crossover_prob, eta=20),
            mutation=PM(prob=self.mutation_prob, eta=20),
            eliminate_duplicates=True
        )
        
        # Run optimization
        start_time = time.time()
        
        if self.verbose:
            print(f"Starting GA optimization...")
            print(f"  Population: {self.population_size}")
            print(f"  Generations: {self.n_generations}")
        
        res = minimize(
            problem,
            algorithm,
            ('n_gen', self.n_generations),
            verbose=self.verbose,
            seed=42
        )
        
        total_time = time.time() - start_time
        
        # Extract results
        if n_objectives == 1:
            best_params = res.X
            best_fitness = res.F[0] if isinstance(res.F, np.ndarray) else res.F
            best_objectives = np.array([best_fitness])
            pareto_front = None
            pareto_params = None
        else:
            # Multi-objective: use first solution from Pareto front
            best_params = res.X[0] if len(res.X.shape) > 1 else res.X
            best_objectives = res.F[0] if len(res.F.shape) > 1 else res.F
            best_fitness = np.sum(best_objectives)  # Combined fitness
            pareto_front = res.F
            pareto_params = res.X
        
        if self.verbose:
            print(f"\nOptimization complete!")
            print(f"  Time: {total_time:.1f}s")
            print(f"  Evaluations: {problem.n_evaluations}")
            print(f"  Best fitness: {best_fitness:.4f}")
        
        return OptimizationResult(
            best_params=best_params,
            best_fitness=best_fitness,
            best_objectives=best_objectives,
            history=[],
            convergence_iter=self.n_generations,
            total_time=total_time,
            n_evaluations=problem.n_evaluations,
            pareto_front=pareto_front,
            pareto_params=pareto_params
        )


class PSOOptimizer:
    """Particle Swarm Optimization."""
    
    def __init__(
        self,
        swarm_size: int = 50,
        max_iterations: int = 300,
        w: float = 0.7,
        c1: float = 1.4,
        c2: float = 1.4,
        verbose: bool = True
    ):
        """
        Initialize PSO optimizer.
        
        Args:
            swarm_size: Number of particles
            max_iterations: Maximum iterations
            w: Inertia weight
            c1: Cognitive coefficient
            c2: Social coefficient
            verbose: Print progress
        """
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.verbose = verbose
    
    def optimize(
        self,
        objective_func: Callable,
        param_bounds: np.ndarray,
        constraint_func: Optional[Callable] = None
    ) -> OptimizationResult:
        """
        Run PSO optimization.
        
        Args:
            objective_func: Objective function (minimize)
            param_bounds: Parameter bounds (n_params, 2)
            constraint_func: Constraint function
            
        Returns:
            OptimizationResult object
        """
        n_params = param_bounds.shape[0]
        lower_bounds = param_bounds[:, 0]
        upper_bounds = param_bounds[:, 1]
        
        # Initialize swarm
        particles = np.random.uniform(
            lower_bounds,
            upper_bounds,
            (self.swarm_size, n_params)
        )
        
        velocities = np.random.uniform(
            -1, 1,
            (self.swarm_size, n_params)
        )
        
        # Evaluate initial positions
        fitness = np.array([objective_func(p) for p in particles])
        
        # Apply constraints
        if constraint_func is not None:
            for i in range(self.swarm_size):
                constraints = constraint_func(particles[i])
                if isinstance(constraints, (list, np.ndarray)):
                    violation = np.sum(np.maximum(0, constraints))
                else:
                    violation = max(0, constraints)
                fitness[i] += 1e6 * violation  # Penalty
        
        # Personal best
        pbest_positions = particles.copy()
        pbest_fitness = fitness.copy()
        
        # Global best
        gbest_idx = np.argmin(fitness)
        gbest_position = particles[gbest_idx].copy()
        gbest_fitness = fitness[gbest_idx]
        
        history = []
        start_time = time.time()
        n_evaluations = self.swarm_size
        
        if self.verbose:
            print(f"Starting PSO optimization...")
            print(f"  Swarm size: {self.swarm_size}")
            print(f"  Max iterations: {self.max_iterations}")
        
        for iteration in range(self.max_iterations):
            for i in range(self.swarm_size):
                # Update velocity
                r1, r2 = np.random.rand(2)
                
                velocities[i] = (
                    self.w * velocities[i] +
                    self.c1 * r1 * (pbest_positions[i] - particles[i]) +
                    self.c2 * r2 * (gbest_position - particles[i])
                )
                
                # Update position
                particles[i] = particles[i] + velocities[i]
                
                # Enforce bounds
                particles[i] = np.clip(particles[i], lower_bounds, upper_bounds)
                
                # Evaluate
                fitness[i] = objective_func(particles[i])
                n_evaluations += 1
                
                # Apply constraints
                if constraint_func is not None:
                    constraints = constraint_func(particles[i])
                    if isinstance(constraints, (list, np.ndarray)):
                        violation = np.sum(np.maximum(0, constraints))
                    else:
                        violation = max(0, constraints)
                    fitness[i] += 1e6 * violation
                
                # Update personal best
                if fitness[i] < pbest_fitness[i]:
                    pbest_fitness[i] = fitness[i]
                    pbest_positions[i] = particles[i].copy()
                
                # Update global best
                if fitness[i] < gbest_fitness:
                    gbest_fitness = fitness[i]
                    gbest_position = particles[i].copy()
            
            # Record history
            history.append({
                'iteration': iteration,
                'best_fitness': gbest_fitness,
                'mean_fitness': np.mean(fitness)
            })
            
            if self.verbose and (iteration + 1) % 50 == 0:
                print(f"  Iteration {iteration+1}/{self.max_iterations} - "
                      f"Best: {gbest_fitness:.4f}")
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print(f"\nOptimization complete!")
            print(f"  Time: {total_time:.1f}s")
            print(f"  Evaluations: {n_evaluations}")
            print(f"  Best fitness: {gbest_fitness:.4f}")
        
        return OptimizationResult(
            best_params=gbest_position,
            best_fitness=gbest_fitness,
            best_objectives=np.array([gbest_fitness]),
            history=history,
            convergence_iter=self.max_iterations,
            total_time=total_time,
            n_evaluations=n_evaluations
        )


class GradientOptimizer:
    """Gradient-based optimizer using SciPy."""
    
    def __init__(self, method: str = 'SLSQP', verbose: bool = True):
        """
        Initialize gradient optimizer.
        
        Args:
            method: Optimization method ('SLSQP', 'L-BFGS-B', 'trust-constr')
            verbose: Print progress
        """
        self.method = method
        self.verbose = verbose
    
    def optimize(
        self,
        objective_func: Callable,
        param_bounds: np.ndarray,
        initial_params: Optional[np.ndarray] = None,
        constraint_func: Optional[Callable] = None
    ) -> OptimizationResult:
        """
        Run gradient-based optimization.
        
        Args:
            objective_func: Objective function
            param_bounds: Parameter bounds (n_params, 2)
            initial_params: Initial guess (optional)
            constraint_func: Constraint function
            
        Returns:
            OptimizationResult object
        """
        if initial_params is None:
            # Random initialization
            initial_params = np.random.uniform(
                param_bounds[:, 0],
                param_bounds[:, 1]
            )
        
        # Convert bounds to scipy format
        bounds = [(low, high) for low, high in param_bounds]
        
        # Convert constraints to scipy format
        constraints = []
        if constraint_func is not None:
            constraints = {
                'type': 'ineq',
                'fun': lambda x: -constraint_func(x)  # g(x) <= 0 -> -g(x) >= 0
            }
        
        start_time = time.time()
        n_evaluations = [0]
        
        def wrapped_objective(x):
            n_evaluations[0] += 1
            return objective_func(x)
        
        if self.verbose:
            print(f"Starting gradient optimization ({self.method})...")
        
        result = scipy_minimize(
            wrapped_objective,
            initial_params,
            method=self.method,
            bounds=bounds,
            constraints=constraints if constraints else (),
            options={'disp': self.verbose}
        )
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print(f"\nOptimization complete!")
            print(f"  Time: {total_time:.1f}s")
            print(f"  Evaluations: {n_evaluations[0]}")
            print(f"  Success: {result.success}")
            print(f"  Best fitness: {result.fun:.4f}")
        
        return OptimizationResult(
            best_params=result.x,
            best_fitness=result.fun,
            best_objectives=np.array([result.fun]),
            history=[],
            convergence_iter=result.nit if hasattr(result, 'nit') else 0,
            total_time=total_time,
            n_evaluations=n_evaluations[0]
        )


if __name__ == "__main__":
    # Test optimizers with Rosenbrock function
    print("Testing optimizers...")
    
    def rosenbrock(x):
        """Rosenbrock function (minimum at (1,1,...,1))."""
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    n_params = 10
    bounds = np.array([[-5, 5]] * n_params)
    
    # Test GA
    print("\n" + "="*50)
    print("Testing Genetic Algorithm")
    print("="*50)
    ga_opt = GeneticAlgorithmOptimizer(
        population_size=50,
        n_generations=100,
        verbose=True
    )
    ga_result = ga_opt.optimize(rosenbrock, bounds)
    print(f"GA Best: {ga_result.best_fitness:.6f}")
    print(f"GA Best params: {ga_result.best_params[:3]}...")
    
    # Test PSO
    print("\n" + "="*50)
    print("Testing Particle Swarm Optimization")
    print("="*50)
    pso_opt = PSOOptimizer(
        swarm_size=30,
        max_iterations=100,
        verbose=True
    )
    pso_result = pso_opt.optimize(rosenbrock, bounds)
    print(f"PSO Best: {pso_result.best_fitness:.6f}")
    print(f"PSO Best params: {pso_result.best_params[:3]}...")
    
    # Test Gradient
    print("\n" + "="*50)
    print("Testing Gradient Optimizer")
    print("="*50)
    grad_opt = GradientOptimizer(method='L-BFGS-B', verbose=True)
    grad_result = grad_opt.optimize(rosenbrock, bounds)
    print(f"Gradient Best: {grad_result.best_fitness:.6f}")
    print(f"Gradient Best params: {grad_result.best_params[:3]}...")
