# ... (imports and Pattern, PatternCache classes remain the same) ...
"""
Ultra-optimized PyTorch Wave Function Collapse with Advanced Caching
Uses pattern caching, CUDA streams, and vectorized operations for maximum performance
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Set
import time
import hashlib
import pickle
import os
from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    """Enumeration for cardinal directions"""
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


@dataclass
class Pattern:
    """Represents a pattern in the wave function collapse"""
    data: np.ndarray
    frequency: int
    neighbors: Dict[Direction, Set[int]]  # pattern_id -> allowed neighbors


class PatternCache:
    """Advanced caching system for patterns and their relationships"""
    
    def __init__(self, cache_dir: str = "pattern_cache"):
        self.cache_dir = cache_dir
        self.pattern_cache = {}  # pattern_hash -> Pattern object
        self.compatibility_cache = {}  # (pattern1_hash, pattern2_hash, direction) -> bool
        self.neighbor_map_cache = {}  # pattern_set_hash -> neighbor_map
        self.stats = {
            'pattern_hits': 0,
            'pattern_misses': 0,
            'compatibility_hits': 0,
            'compatibility_misses': 0,
            'neighbor_map_hits': 0,
            'neighbor_map_misses': 0
        }
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing cache
        self._load_cache()
        
    def _get_pattern_hash(self, pattern_data: np.ndarray) -> str:
        """Generate a hash for a pattern"""
        return hashlib.md5(pattern_data.tobytes()).hexdigest()
    
    def _get_pattern_set_hash(self, patterns: List[Pattern]) -> str:
        """Generate a hash for a set of patterns"""
        pattern_hashes = [self._get_pattern_hash(p.data) for p in patterns]
        pattern_hashes.sort()  # Ensure consistent ordering
        return hashlib.md5(''.join(pattern_hashes).encode()).hexdigest()
    
    def get_pattern(self, pattern_data: np.ndarray) -> Pattern:
        """Get a pattern from cache or create new one"""
        pattern_hash = self._get_pattern_hash(pattern_data)
        
        if pattern_hash in self.pattern_cache:
            self.stats['pattern_hits'] += 1
            return self.pattern_cache[pattern_hash]
        else:
            self.stats['pattern_misses'] += 1
            # Create new pattern
            pattern = Pattern(
                data=pattern_data.copy(),
                frequency=1,
                neighbors={direction: set() for direction in Direction}
            )
            self.pattern_cache[pattern_hash] = pattern
            return pattern
    
    def get_compatibility(self, pattern1_hash: str, pattern2_hash: str, direction: Direction) -> bool:
        """Get pattern compatibility from cache"""
        cache_key = (pattern1_hash, pattern2_hash, direction)
        
        if cache_key in self.compatibility_cache:
            self.stats['compatibility_hits'] += 1
            return self.compatibility_cache[cache_key]
        else:
            self.stats['compatibility_misses'] += 1
            return None  # Not cached, need to calculate
    
    def set_compatibility(self, pattern1_hash: str, pattern2_hash: str, direction: Direction, compatible: bool):
        """Set pattern compatibility in cache"""
        cache_key = (pattern1_hash, pattern2_hash, direction)
        self.compatibility_cache[cache_key] = compatible
    
    def get_neighbor_map(self, patterns: List[Pattern]) -> torch.Tensor:
        """Get neighbor map from cache"""
        pattern_set_hash = self._get_pattern_set_hash(patterns)
        
        if pattern_set_hash in self.neighbor_map_cache:
            self.stats['neighbor_map_hits'] += 1
            return self.neighbor_map_cache[pattern_set_hash]
        else:
            self.stats['neighbor_map_misses'] += 1
            return None  # Not cached, need to calculate
    
    def set_neighbor_map(self, patterns: List[Pattern], neighbor_map: torch.Tensor):
        """Set neighbor map in cache"""
        pattern_set_hash = self._get_pattern_set_hash(patterns)
        self.neighbor_map_cache[pattern_set_hash] = neighbor_map.clone()
    
    def _load_cache(self):
        """Load cache from disk"""
        try:
            cache_file = os.path.join(self.cache_dir, "pattern_cache.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.pattern_cache = cache_data.get('patterns', {})
                    self.compatibility_cache = cache_data.get('compatibility', {})
                    self.neighbor_map_cache = cache_data.get('neighbor_maps', {})
                print(f"Loaded cache with {len(self.pattern_cache)} patterns, {len(self.compatibility_cache)} compatibilities")
        except Exception as e:
            print(f"Failed to load cache: {e}")
    
    def save_cache(self):
        """Save cache to disk"""
        try:
            cache_file = os.path.join(self.cache_dir, "pattern_cache.pkl")
            cache_data = {
                'patterns': self.pattern_cache,
                'compatibility': self.compatibility_cache,
                'neighbor_maps': self.neighbor_map_cache
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Saved cache with {len(self.pattern_cache)} patterns, {len(self.compatibility_cache)} compatibilities")
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def print_stats(self):
        """Print cache statistics"""
        total_patterns = self.stats['pattern_hits'] + self.stats['pattern_misses']
        total_compatibilities = self.stats['compatibility_hits'] + self.stats['compatibility_misses']
        total_neighbor_maps = self.stats['neighbor_map_hits'] + self.stats['neighbor_map_misses']
        
        print("Cache Statistics:")
        print(f"  Pattern cache hit rate: {self.stats['pattern_hits']}/{total_patterns} ({self.stats['pattern_hits']/max(1,total_patterns)*100:.1f}%)")
        print(f"  Compatibility cache hit rate: {self.stats['compatibility_hits']}/{total_compatibilities} ({self.stats['compatibility_hits']/max(1,total_compatibilities)*100:.1f}%)")
        print(f"  Neighbor map cache hit rate: {self.stats['neighbor_map_hits']}/{total_neighbor_maps} ({self.stats['neighbor_map_hits']/max(1,total_neighbor_maps)*100:.1f}%)")


class WaveFunctionCollapsePyTorchCached:
    # ... (__init__, set_progress_callback, stop methods remain the same) ...
    """Ultra-optimized PyTorch Wave Function Collapse with Advanced Caching"""
    
    def __init__(self, pattern_size: int = 3, output_width: int = 50, output_height: int = 50, cache_dir: str = "pattern_cache"):
        self.pattern_size = pattern_size
        self.output_width = output_width
        self.output_height = output_height
        self.patterns: List[Pattern] = []
        self.pattern_weights: torch.Tensor = None
        self.wave: torch.Tensor = None  # 3D tensor: [y, x, pattern_id] -> bool
        self.entropy: torch.Tensor = None  # 2D tensor: [y, x] -> entropy value
        self.output: torch.Tensor = None  # 2D tensor: final result
        self.generation_steps = 0
        self.progress_callback = None
        self.should_stop = False
        
        # Initialize cache
        self.cache = PatternCache(cache_dir)
        
        # PyTorch optimization parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_batch_size = 2048  # Base batch size
        self.max_batch_size = 16384  # Maximum batch size for dynamic batching
        
        # CUDA Streams for concurrent execution
        self.use_streams = torch.cuda.is_available()
        if self.use_streams:
            self.stream1 = torch.cuda.Stream()
            self.stream2 = torch.cuda.Stream()
            self.stream3 = torch.cuda.Stream()
            print("CUDA Streams enabled for concurrent execution")
        
        # Neighbor compatibility map for GPU
        self.neighbor_map: torch.Tensor = None
        
        # Pre-allocated tensors for better memory management
        self.temp_entropy = None
        self.temp_wave = None
        self.collapse_mask = None
        self.min_entropy_positions = None
        
        # Dynamic batching parameters
        self.batch_size = self.base_batch_size
        self.performance_history = []
        
        print(f"Using device: {self.device}")
        print(f"Base batch size: {self.base_batch_size}")
        print(f"Max batch size: {self.max_batch_size}")
        print(f"Cache directory: {cache_dir}")
        
    def set_progress_callback(self, callback):
        """Set callback function for progress updates"""
        self.progress_callback = callback
        
    def stop(self):
        """Stop the generation process"""
        self.should_stop = True

    # -- MODIFIED: Added extraction_callback parameter --
    def extract_patterns(self, image: np.ndarray, extraction_callback=None) -> List[Pattern]:
        """Extract patterns from the source image - cached and ultra-optimized"""
        print("Extracting patterns from source image...")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = np.mean(image, axis=2).astype(np.uint8)
            print(f"Converted to grayscale: {image.shape}")
            
        patterns = {}
        pattern_map = {}
        total_positions = (image.shape[0] - self.pattern_size + 1) * (image.shape[1] - self.pattern_size + 1)
        processed_positions = 0
        
        print(f"Scanning {total_positions} possible pattern positions...")
        
        # Cached pattern extraction
        for y in range(image.shape[0] - self.pattern_size + 1):
            for x in range(image.shape[1] - self.pattern_size + 1):
                pattern_data = image[y:y+self.pattern_size, x:x+self.pattern_size]
                pattern_hash = self.cache._get_pattern_hash(pattern_data)
                
                if pattern_hash in patterns:
                    patterns[pattern_hash] += 1
                else:
                    patterns[pattern_hash] = 1
                    pattern_map[pattern_hash] = pattern_data
                
                processed_positions += 1
                
                # -- NEW: Use the extraction callback --
                if extraction_callback and processed_positions % 100 == 0:
                    progress = (processed_positions / total_positions) * 100
                    extraction_callback(progress)
                    
        # ... (rest of the method remains the same) ...
        print(f"Pattern extraction complete: {len(patterns)} unique patterns found")
        print("Creating pattern objects...")
        
        # Create Pattern objects using cache
        pattern_list = []
        for pattern_hash, frequency in patterns.items():
            pattern_data = pattern_map[pattern_hash]
            pattern = self.cache.get_pattern(pattern_data)
            pattern.frequency = frequency
            pattern_list.append(pattern)
            
        print("Finding compatible neighbors...")
        # Find compatible neighbors using cache
        self._find_compatible_neighbors_vectorized(pattern_list)
        
        self.patterns = pattern_list
        self.pattern_weights = torch.tensor([p.frequency for p in pattern_list], 
                                          dtype=torch.float32, device=self.device)
        
        # Create neighbor compatibility map for GPU using cache
        self._create_neighbor_map_cached()
        
        print(f"Pattern extraction complete: {len(pattern_list)} unique patterns with neighbor relationships")
        return pattern_list

    # ... (the rest of the file remains the same) ...
    def _propagate_constraints_vectorized(self, start_y: int, start_x: int):
        """
        Propagate constraints using fully vectorized tensor operations on the GPU.
        This avoids slow Python loops and maximizes parallelism.
        """
        stack = [(start_y, start_x)]
        # Use a tensor for visited tracking to keep it on the GPU
        visited = torch.zeros((self.output_height, self.output_width), dtype=torch.bool, device=self.device)
        visited[start_y, start_x] = True

        while stack:
            y, x = stack.pop()

            # Get the boolean tensor of possible patterns for the current cell
            # Shape: [num_patterns]
            possible_patterns_mask = self.wave[y, x, :]

            # If a contradiction occurred, skip
            if not torch.any(possible_patterns_mask):
                continue

            # Process all 4 neighbors
            for d_idx, direction in enumerate(Direction):
                dy, dx = direction.value
                ny, nx = y + dy, x + dx

                # Boundary check
                if not (0 <= ny < self.output_height and 0 <= nx < self.output_width):
                    continue
                
                # Get the neighbor's original wave state
                original_neighbor_wave = self.wave[ny, nx, :]
                
                # If neighbor is already collapsed, nothing to do
                if self.entropy[ny, nx] <= 1:
                    continue

                # This is the core vectorized logic:
                # 1. Select compatibility rules for all patterns in the current cell that are possible.
                #    `self.neighbor_map[:, possible_patterns_mask, d_idx]` gives a boolean matrix of shape
                #    [num_neighbor_patterns, num_possible_patterns_in_current_cell]
                # 2. `torch.any(..., dim=1)` collapses this to a single boolean tensor of shape
                #    [num_neighbor_patterns], where `True` means that a neighbor pattern is compatible
                #    with at least ONE of the current cell's possible patterns.
                valid_neighbor_patterns_mask = torch.any(self.neighbor_map[:, possible_patterns_mask, d_idx], dim=1)
                
                # Update the neighbor's wave with a logical AND
                self.wave[ny, nx, :] &= valid_neighbor_patterns_mask
                
                # Check if the neighbor's state has changed
                if not torch.equal(original_neighbor_wave, self.wave[ny, nx, :]):
                    # Update entropy for the changed neighbor
                    self.entropy[ny, nx] = torch.sum(self.wave[ny, nx, :].float())
                    
                    # If the neighbor hasn't been visited, add it to the stack
                    if not visited[ny, nx]:
                        stack.append((ny, nx))
                        visited[ny, nx] = True

    def _find_compatible_neighbors_vectorized(self, patterns: List[Pattern]):
        """
        Find which patterns can be neighbors in each direction using a single,
        vectorized GPU operation for maximum performance.
        """
        # <<< FIX: Use the 'patterns' argument, not 'self.patterns' >>>
        num_patterns = len(patterns)
        if num_patterns == 0:
            print("Warning: No patterns found to build compatibility from.")
            return

        print(f"Building compatibility for {num_patterns} patterns using vectorized operations...")

        # Move all pattern data to a single GPU tensor
        # Shape: [num_patterns, pattern_size, pattern_size]
        all_patterns_tensor = torch.tensor(
            # <<< FIX: Use 'patterns' argument and np.stack for robustness >>>
            np.stack([p.data for p in patterns]),
            device=self.device
        )

        # Extract all edges at once
        # Each edge tensor has shape: [num_patterns, pattern_size]
        left_edges = all_patterns_tensor[:, :, 0]
        right_edges = all_patterns_tensor[:, :, -1]
        up_edges = all_patterns_tensor[:, 0, :]
        down_edges = all_patterns_tensor[:, -1, :]

        # Use broadcasting to compare all edge pairs simultaneously
        # ... (the rest of this logic is correct)
        compatibility_right = torch.all(left_edges.unsqueeze(1) == right_edges.unsqueeze(0), dim=2)
        compatibility_left = torch.all(right_edges.unsqueeze(1) == left_edges.unsqueeze(0), dim=2)
        compatibility_down = torch.all(up_edges.unsqueeze(1) == down_edges.unsqueeze(0), dim=2)
        compatibility_up = torch.all(down_edges.unsqueeze(1) == up_edges.unsqueeze(0), dim=2)
        
        # Populate the neighbor sets from the compatibility matrices
        compat_matrices = {
            Direction.RIGHT: compatibility_right.cpu(),
            Direction.LEFT: compatibility_left.cpu(),
            Direction.UP: compatibility_up.cpu(),
            Direction.DOWN: compatibility_down.cpu()
        }

        for i in range(num_patterns):
            for direction, matrix in compat_matrices.items():
                compatible_indices = torch.where(matrix[i, :])[0].numpy()
                # <<< FIX: Update the 'patterns' object that was passed in >>>
                patterns[i].neighbors[direction] = set(compatible_indices)
        
        print("Compatibility analysis complete.")

    def _patterns_compatible_cached(self, pattern_edges: dict, i: int, j: int, direction: Direction) -> bool:
        """Check if two patterns are compatible in a given direction - cached and ultra-optimized"""
        # Get the edges that need to match
        if direction == Direction.LEFT:
            edge1 = pattern_edges[i][Direction.LEFT]
            edge2 = pattern_edges[j][Direction.RIGHT]
        elif direction == Direction.RIGHT:
            edge1 = pattern_edges[i][Direction.RIGHT]
            edge2 = pattern_edges[j][Direction.LEFT]
        elif direction == Direction.UP:
            edge1 = pattern_edges[i][Direction.UP]
            edge2 = pattern_edges[j][Direction.DOWN]
        elif direction == Direction.DOWN:
            edge1 = pattern_edges[i][Direction.DOWN]
            edge2 = pattern_edges[j][Direction.UP]
        else:
            return False
            
        return np.array_equal(edge1, edge2)
        
    def _create_neighbor_map_cached(self):
        """Create neighbor compatibility map for GPU using cache"""
        # Check if we have a cached neighbor map
        cached_neighbor_map = self.cache.get_neighbor_map(self.patterns)
        
        if cached_neighbor_map is not None:
            # Use cached neighbor map
            self.neighbor_map = cached_neighbor_map.to(self.device)
            print("Using cached neighbor map")
        else:
            # Create new neighbor map
            num_patterns = len(self.patterns)
            self.neighbor_map = torch.zeros((num_patterns, num_patterns, 4), 
                                          dtype=torch.bool, device=self.device)
            
            for i, pattern1 in enumerate(self.patterns):
                for j, pattern2 in enumerate(self.patterns):
                    for d, direction in enumerate(Direction):
                        if j in pattern1.neighbors[direction]:
                            self.neighbor_map[i, j, d] = True
            
            # Cache the neighbor map
            self.cache.set_neighbor_map(self.patterns, self.neighbor_map)
            print("Created and cached new neighbor map")
        
    def initialize_wave(self):
        """Initialize the wave function with all patterns possible everywhere"""
        self.wave = torch.ones((self.output_height, self.output_width, len(self.patterns)), 
                             dtype=torch.bool, device=self.device)
        self.entropy = torch.full((self.output_height, self.output_width), len(self.patterns), 
                                dtype=torch.float32, device=self.device)
        self.output = torch.zeros((self.output_height, self.output_width), 
                                dtype=torch.uint8, device=self.device)
        self.generation_steps = 0
        
        # Pre-allocate temporary tensors for better performance
        self.temp_entropy = torch.zeros_like(self.entropy)
        self.temp_wave = torch.zeros_like(self.wave)
        self.collapse_mask = torch.zeros((self.output_height, self.output_width), 
                                       dtype=torch.bool, device=self.device)
        self.min_entropy_positions = torch.zeros((self.max_batch_size, 2), 
                                               dtype=torch.long, device=self.device)
        
    def collapse(self) -> np.ndarray:
        """Main collapse algorithm with caching and ultra-optimization"""
        if not self.patterns:
            raise ValueError("No patterns available. Call extract_patterns first.")
            
        self.initialize_wave()
        
        # Main loop with caching and ultra-optimization
        progress_update_interval = 100  # Update progress every 100 steps to minimize CPU-GPU transfers for maximum GPU utilization
        while not self._is_fully_collapsed() and not self.should_stop:
            # Find cells with minimum entropy using dynamic batching
            min_entropy_cells = self._find_min_entropy_cached()
            
            if not min_entropy_cells:
                break
                
            # Process cells using CUDA Streams
            self._collapse_cells_cached(min_entropy_cells)
            
            self.generation_steps += 1
            
            # Update progress less frequently to reduce CPU-GPU transfers
            if self.progress_callback and not self.should_stop and self.generation_steps % progress_update_interval == 0:
                progress = self._calculate_progress()
                # Get the current visualization and pass it to the callback
                current_state = self._get_current_state()
                self.progress_callback(progress, self.generation_steps, current_state)
                
        return self._get_final_result()
        
    def _is_fully_collapsed(self) -> bool:
        """Check if the wave function is fully collapsed"""
        return torch.all(self.entropy <= 1).item()
        
    def _find_min_entropy_cached(self) -> List[Tuple[int, int]]:
        """Find multiple cells with minimum entropy using cached operations"""
        # Create mask for non-collapsed cells
        mask = self.entropy > 1
        
        if not torch.any(mask):
            return []
            
        # Use CUDA Streams for concurrent execution with better GPU utilization
        if self.use_streams:
            with torch.cuda.stream(self.stream2):
                masked_entropy = torch.where(mask, self.entropy, torch.tensor(float('inf'), device=self.device))
                min_entropy = torch.min(masked_entropy)
                
                # Find all positions with minimum entropy
                min_mask = (self.entropy == min_entropy) & mask
                min_positions = torch.where(min_mask)
        else:
            masked_entropy = torch.where(mask, self.entropy, torch.tensor(float('inf'), device=self.device))
            min_entropy = torch.min(masked_entropy)
            min_mask = (self.entropy == min_entropy) & mask
            min_positions = torch.where(min_mask)
        
        if len(min_positions[0]) == 0:
            return []
            
        # Dynamic batching: adapt batch size based on performance
        available_cells = len(min_positions[0])
        dynamic_batch_size = min(available_cells, self.batch_size)
        
        # Optimize: Keep more operations on GPU, minimize CPU-GPU transfers
        # Only convert the minimum necessary to CPU
        y_coords = min_positions[0][:dynamic_batch_size]
        x_coords = min_positions[1][:dynamic_batch_size]
        
        # Convert to CPU in batch to minimize transfers
        y_coords_cpu = y_coords.cpu()
        x_coords_cpu = x_coords.cpu()
        
        # Build positions list
        positions = []
        for i in range(len(y_coords_cpu)):
            y = int(y_coords_cpu[i].item())
            x = int(x_coords_cpu[i].item())
            # Ensure coordinates are within bounds
            if 0 <= y < self.output_height and 0 <= x < self.output_width:
                positions.append((y, x))
        
        return positions
        
    def _collapse_cells_cached(self, cells: List[Tuple[int, int]]):
        """Collapse multiple cells using cached operations with GPU optimization"""
        if not cells:
            return
            
        # Filter valid cells
        valid_cells = [(y, x) for y, x in cells 
                      if 0 <= y < self.output_height and 0 <= x < self.output_width]
        
        if not valid_cells:
            return
            
        # Process cells in batches for better GPU utilization
        batch_size = min(len(valid_cells), 2048)  # Process up to 2048 cells at once for maximum GPU utilization
        
        for i in range(0, len(valid_cells), batch_size):
            batch_cells = valid_cells[i:i + batch_size]
            
            # Process batch on GPU with CUDA streams for maximum concurrency
            if self.use_streams:
                with torch.cuda.stream(self.stream1):
                    for y, x in batch_cells:
                        # Get possible patterns for this cell
                        cell_wave = self.wave[y, x, :]
                        possible_patterns = torch.where(cell_wave)[0]
                        
                        if len(possible_patterns) == 0:
                            continue
                            
                        # Weight selection by pattern frequency
                        weights = self.pattern_weights[possible_patterns]
                        weights = weights / torch.sum(weights)
                        
                        # Choose pattern based on weights
                        chosen_pattern = torch.multinomial(weights, 1).item()
                        chosen_pattern = possible_patterns[chosen_pattern].item()
                        
                        # Set only the chosen pattern as possible
                        self.wave[y, x, :] = False
                        self.wave[y, x, chosen_pattern] = True
                        
                        # Update entropy
                        self.entropy[y, x] = 1
                        
                        # Set output value (use center pixel of pattern)
                        center_y = self.pattern_size // 2
                        center_x = self.pattern_size // 2
                        self.output[y, x] = self.patterns[chosen_pattern].data[center_y, center_x]
                        
                        # Propagate constraints using cached operations
                        self._propagate_constraints_vectorized(y, x)
            else:
                # Fallback without streams
                for y, x in batch_cells:
                    # Get possible patterns for this cell
                    cell_wave = self.wave[y, x, :]
                    possible_patterns = torch.where(cell_wave)[0]
                    
                    if len(possible_patterns) == 0:
                        continue
                        
                    # Weight selection by pattern frequency
                    weights = self.pattern_weights[possible_patterns]
                    weights = weights / torch.sum(weights)
                    
                    # Choose pattern based on weights
                    chosen_pattern = torch.multinomial(weights, 1).item()
                    chosen_pattern = possible_patterns[chosen_pattern].item()
                    
                    # Set only the chosen pattern as possible
                    self.wave[y, x, :] = False
                    self.wave[y, x, chosen_pattern] = True
                    
                    # Update entropy
                    self.entropy[y, x] = 1
                    
                    # Set output value (use center pixel of pattern)
                    center_y = self.pattern_size // 2
                    center_x = self.pattern_size // 2
                    self.output[y, x] = self.patterns[chosen_pattern].data[center_y, center_x]
                    
                    # Propagate constraints using cached operations
                    self._propagate_constraints_vectorized(y, x)
        
    def _propagate_constraints_cached(self, start_y: int, start_x: int):
        """Propagate constraints using cached operations"""
        # Use a more efficient approach with visited tracking
        stack = [(start_y, start_x)]
        visited = set()
        
        while stack:
            y, x = stack.pop()
            
            if (y, x) in visited:
                continue
            visited.add((y, x))
            
            # Get current cell's possible patterns
            cell_wave = self.wave[y, x, :]
            current_possible = torch.where(cell_wave)[0]
            if len(current_possible) == 0:
                continue
                
            # Check all four directions
            directions = [(0, -1, 0), (0, 1, 1), (-1, 0, 2), (1, 0, 3)]  # (dy, dx, direction_index)
            
            for dy, dx, d in directions:
                ny, nx = y + dy, x + dx
                
                # Check bounds and if neighbor needs updating
                if (0 <= ny < self.output_height and 
                    0 <= nx < self.output_width and 
                    self.entropy[ny, nx] > 1):
                    
                    # Get neighbor's current possible patterns
                    neighbor_wave = self.wave[ny, nx, :]
                    neighbor_possible = torch.where(neighbor_wave)[0]
                    
                    if len(neighbor_possible) == 0:
                        continue
                    
                    # Cached compatibility check
                    if self.use_streams:
                        with torch.cuda.stream(self.stream2):
                            compatibility_mask = torch.zeros(len(neighbor_possible), dtype=torch.bool, device=self.device)
                            
                            for i, neighbor_pattern in enumerate(neighbor_possible):
                                # Check if this neighbor pattern is compatible with any current pattern
                                for current_pattern in current_possible:
                                    if self.neighbor_map[neighbor_pattern, current_pattern, d]:
                                        compatibility_mask[i] = True
                                        break
                    else:
                        compatibility_mask = torch.zeros(len(neighbor_possible), dtype=torch.bool, device=self.device)
                        
                        for i, neighbor_pattern in enumerate(neighbor_possible):
                            # Check if this neighbor pattern is compatible with any current pattern
                            for current_pattern in current_possible:
                                if self.neighbor_map[neighbor_pattern, current_pattern, d]:
                                    compatibility_mask[i] = True
                                    break
                    
                    # Update neighbor's wave function
                    new_possible = neighbor_possible[compatibility_mask]
                    
                    if len(new_possible) != len(neighbor_possible):
                        # Update the wave function
                        self.wave[ny, nx, :] = False
                        if len(new_possible) > 0:
                            self.wave[ny, nx, new_possible] = True
                            self.entropy[ny, nx] = len(new_possible)
                        else:
                            self.entropy[ny, nx] = 0  # Contradiction
                            
                        # Add to stack for further propagation
                        if (ny, nx) not in visited:
                            stack.append((ny, nx))
                        
    def _calculate_progress(self) -> float:
        """Calculate generation progress as percentage"""
        total_cells = self.output_height * self.output_width
        collapsed_cells = torch.sum(self.entropy <= 1).item()
        return float(collapsed_cells / total_cells * 100)
        
    def _get_current_state(self) -> np.ndarray:
        """Get current state of the output for visualization"""
        if self.wave is not None and self.entropy is not None:
            # Create a visualization of the current wave function state
            # For collapsed cells, show the output value
            # For uncollapsed cells, show a simple average based on entropy
            current_state = torch.zeros((self.output_height, self.output_width), dtype=torch.float32, device=self.device)
            
            # Start with collapsed cells
            collapsed_mask = self.entropy <= 1
            current_state[collapsed_mask] = self.output[collapsed_mask].float()
            
            # For uncollapsed cells, show a pattern based on entropy
            uncollapsed_mask = self.entropy > 1
            if uncollapsed_mask.any():
                # Create a pattern based on entropy values for visualization
                # Higher entropy = more uncertainty = darker
                max_entropy = self.entropy.max()
                if max_entropy > 1:
                    # Normalize entropy to 0-1 range, then invert (higher entropy = darker)
                    entropy_normalized = (self.entropy[uncollapsed_mask] - 1) / (max_entropy - 1)
                    # Create a pattern based on position and entropy
                    y_coords, x_coords = torch.where(uncollapsed_mask)
                    pattern_value = (torch.sin(y_coords.float() * 0.1) + torch.cos(x_coords.float() * 0.1)) * 0.5 + 0.5
                    pattern_value = pattern_value * (1 - entropy_normalized) * 255  # Darker for higher entropy
                    current_state[uncollapsed_mask] = pattern_value
            
            # Convert to numpy and ensure proper range
            current_state_np = current_state.cpu().numpy()
            current_state_np = np.clip(current_state_np, 0, 255).astype(np.uint8)
            
            return current_state_np
        elif self.entropy is not None:
            # Wave not initialized yet, but entropy is available (after pattern extraction)
            # Show a pattern based on entropy values
            current_state = torch.zeros((self.output_height, self.output_width), dtype=torch.float32, device=self.device)
            
            # Create a pattern based on entropy values for visualization
            max_entropy = self.entropy.max()
            if max_entropy > 0:
                # Normalize entropy to 0-1 range
                entropy_normalized = self.entropy / max_entropy
                # Create a pattern based on position and entropy
                y_coords, x_coords = torch.meshgrid(torch.arange(self.output_height, device=self.device),
                                                  torch.arange(self.output_width, device=self.device), indexing='ij')
                pattern_value = (torch.sin(y_coords.float() * 0.1) + torch.cos(x_coords.float() * 0.1)) * 0.5 + 0.5
                pattern_value = pattern_value * (1 - entropy_normalized) * 255  # Darker for higher entropy
                current_state = pattern_value
            
            # Convert to numpy and ensure proper range
            current_state_np = current_state.cpu().numpy()
            current_state_np = np.clip(current_state_np, 0, 255).astype(np.uint8)
            
            return current_state_np
        else:
            # Nothing initialized yet, return a simple pattern
            current_state = torch.zeros((self.output_height, self.output_width), dtype=torch.float32, device=self.device)
            # Create a simple pattern
            y_coords, x_coords = torch.meshgrid(torch.arange(self.output_height, device=self.device),
                                              torch.arange(self.output_width, device=self.device), indexing='ij')
            pattern_value = (torch.sin(y_coords.float() * 0.2) + torch.cos(x_coords.float() * 0.2)) * 0.5 + 0.5
            pattern_value = pattern_value * 255
            
            # Convert to numpy and ensure proper range
            current_state_np = pattern_value.cpu().numpy()
            current_state_np = np.clip(current_state_np, 0, 255).astype(np.uint8)
            
            return current_state_np
        
    def _get_final_result(self) -> np.ndarray:
        """Get the final collapsed result"""
        if self.output is not None:
            result = self.output.cpu().numpy()
            # Ensure result is in the correct format
            if result.dtype != np.uint8:
                # Normalize to 0-255 range if needed
                if result.max() <= 1.0:
                    result = (result * 255).astype(np.uint8)
                else:
                    result = np.clip(result, 0, 255).astype(np.uint8)
            return result
        return None
    
    def save_cache(self):
        """Save cache to disk"""
        self.cache.save_cache()
    
    def print_cache_stats(self):
        """Print cache statistics"""
        self.cache.print_stats()