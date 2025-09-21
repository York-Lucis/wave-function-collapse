"""
GUI Interface for Wave Function Collapse
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import cv2
import sys
import io

# ... (rest of the import section remains the same) ...
# Try to import PyTorch CUDA version, fallback to optimized CPU if not available
try:
    import torch
    # Test if CUDA actually works by trying a simple operation
    if torch.cuda.is_available():
        test_tensor = torch.tensor([1, 2, 3], device='cuda')
        result = torch.sum(test_tensor)
        from wave_function_collapse_pytorch_cached import WaveFunctionCollapsePyTorchCached as WaveFunctionCollapse
        CUDA_AVAILABLE = True
        print("PyTorch CUDA acceleration available (GPU optimized)")
    else:
        raise RuntimeError("CUDA not available")
except (ImportError, RuntimeError, Exception) as e:
    # Try optimized CPU version with Numba
    try:
        from wave_function_collapse_cpu_optimized import WaveFunctionCollapseCPUOptimized as WaveFunctionCollapse
        CUDA_AVAILABLE = False
        print("Using CPU-optimized mode with Numba acceleration")
    except ImportError:
        # Fallback to basic CPU version
        from wave_function_collapse_cpu import WaveFunctionCollapseCPU as WaveFunctionCollapse
        CUDA_AVAILABLE = False
        print(f"Using basic CPU mode (PyTorch error: {e})")


class LogCapture:
    # ... (LogCapture class remains the same) ...
    """Capture stdout and redirect to GUI log display"""
    
    def __init__(self, gui_instance):
        self.gui = gui_instance
        self.original_stdout = sys.stdout
        self.buffer = io.StringIO()
        
    def write(self, message):
        """Write message to buffer and GUI"""
        self.buffer.write(message)
        if message.strip():  # Only append non-empty messages
            try:
                # Use a lambda with the message captured to avoid closure issues
                msg = message.strip()
                self.gui.root.after(0, lambda m=msg: self.gui.append_log(m))
            except Exception as e:
                # If GUI is destroyed, just ignore the error
                pass
            
    def flush(self):
        """Flush the buffer"""
        self.buffer.flush()
        
    def start_capture(self):
        """Start capturing stdout"""
        sys.stdout = self
        
    def stop_capture(self):
        """Stop capturing and restore stdout"""
        sys.stdout = self.original_stdout


class WaveFunctionCollapseGUI:
    # ... (__init__, setup_gui, setup_control_panel, setup_image_display remain the same) ...
    """Main GUI class for the Wave Function Collapse application"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Wave Function Collapse with CUDA")
        self.root.geometry("1200x800")
        
        # Application state
        self.source_image = None
        self.wfc = None
        self.generation_thread = None
        self.is_generating = False
        self.current_output = None
        self.cuda_available = CUDA_AVAILABLE
        self.log_capture = LogCapture(self)
        
        # GUI components
        self.setup_gui()
        
    def setup_gui(self):
        """Set up the GUI layout"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)  # Image display area
        main_frame.rowconfigure(3, weight=1)  # Log display area
        
        # Control panel
        self.setup_control_panel(main_frame)
        
        # Image display area
        self.setup_image_display(main_frame)
        
        # Progress and status
        self.setup_progress_display(main_frame)
        
        # Log display
        self.setup_log_display(main_frame)
        
    def setup_control_panel(self, parent):
        """Set up the control panel with buttons and parameters"""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Image loading
        ttk.Button(control_frame, text="Load Source Image", 
                  command=self.load_image).grid(row=0, column=0, padx=(0, 10))
        
        self.image_label = ttk.Label(control_frame, text="No image loaded")
        self.image_label.grid(row=0, column=1, padx=(0, 20))
        
        # Parameters
        params_frame = ttk.Frame(control_frame)
        params_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Pattern size
        ttk.Label(params_frame, text="Pattern Size:").grid(row=0, column=0, sticky=tk.W)
        self.pattern_size_var = tk.IntVar(value=3)
        pattern_spinbox = ttk.Spinbox(params_frame, from_=2, to=5, width=5,
                                     textvariable=self.pattern_size_var)
        pattern_spinbox.grid(row=0, column=1, padx=(5, 20))
        
        # Output dimensions
        ttk.Label(params_frame, text="Output Width:").grid(row=0, column=2, sticky=tk.W)
        self.output_width_var = tk.IntVar(value=50)
        width_spinbox = ttk.Spinbox(params_frame, from_=20, to=200, width=5,
                                   textvariable=self.output_width_var)
        width_spinbox.grid(row=0, column=3, padx=(5, 20))
        
        ttk.Label(params_frame, text="Output Height:").grid(row=0, column=4, sticky=tk.W)
        self.output_height_var = tk.IntVar(value=50)
        height_spinbox = ttk.Spinbox(params_frame, from_=20, to=200, width=5,
                                    textvariable=self.output_height_var)
        height_spinbox.grid(row=0, column=5, padx=(5, 0))
        
        # CUDA status
        cuda_status = "CUDA Available" if self.cuda_available else "CPU Mode"
        ttk.Label(control_frame, text=f"Mode: {cuda_status}").grid(row=2, column=0, pady=(10, 0))
        
        # Generate button
        self.generate_button = ttk.Button(control_frame, text="Generate", 
                                         command=self.start_generation)
        self.generate_button.grid(row=2, column=1, pady=(10, 0))
        
        # Stop button
        self.stop_button = ttk.Button(control_frame, text="Stop", 
                                     command=self.stop_generation, state=tk.DISABLED)
        self.stop_button.grid(row=2, column=2, pady=(10, 0), padx=(10, 0))
        
    def setup_image_display(self, parent):
        """Set up the image display area with matplotlib"""
        display_frame = ttk.LabelFrame(parent, text="Visualization", padding="10")
        display_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.fig, (self.ax_source, self.ax_output) = plt.subplots(1, 2, figsize=(12, 6))
        self.ax_source.set_title("Source Image")
        self.ax_output.set_title("Generated Output")
        
        # Initialize with placeholder images
        self.ax_source.imshow(np.zeros((100, 100), dtype=np.uint8), cmap='gray')
        self.ax_output.imshow(np.zeros((100, 100), dtype=np.uint8), cmap='gray')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, display_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def setup_progress_display(self, parent):
        """Set up progress and status display"""
        self.progress_frame = ttk.LabelFrame(parent, text="Progress", padding="10")
        self.progress_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
        self.progress_frame.columnconfigure(0, weight=1)

        # -- NEW: Pattern Extraction Progress Bar --
        self.pattern_progress_var = tk.DoubleVar()
        self.pattern_progress_bar = ttk.Progressbar(self.progress_frame, variable=self.pattern_progress_var,
                                                   maximum=100, length=400)
        self.pattern_progress_label = ttk.Label(self.progress_frame, text="0%")

        # -- Main Generation Progress Bar --
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var,
                                           maximum=100, length=400)
        self.progress_label = ttk.Label(self.progress_frame, text="0%")

        # Status label
        self.status_label = ttk.Label(self.progress_frame, text="Ready")
        self.status_label.grid(row=2, column=0, columnspan=2, pady=(5, 0))

    # ... (setup_log_display, clear_log, append_log, load_image remain the same) ...
    def setup_log_display(self, parent):
        """Set up the log display area"""
        log_frame = ttk.LabelFrame(parent, text="Log Output", padding="10")
        log_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # Create text widget with scrollbar
        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD, font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Clear log button
        ttk.Button(log_frame, text="Clear Log", command=self.clear_log).grid(row=1, column=0, pady=(5, 0), sticky=tk.W)
        
    def clear_log(self):
        """Clear the log display"""
        self.log_text.delete(1.0, tk.END)
        
    def append_log(self, message):
        """Append a message to the log display"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def load_image(self):
        """Load source image from file"""
        file_path = filedialog.askopenfilename(
            title="Select Source Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                # Load image
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError("Could not load image")
                    
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.source_image = image
                
                # Update GUI
                self.image_label.config(text=f"Loaded: {file_path.split('/')[-1]}")
                self.ax_source.imshow(image)
                self.ax_source.set_title(f"Source Image ({image.shape[1]}x{image.shape[0]})")
                self.canvas.draw()
                
                self.status_label.config(text="Image loaded successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.status_label.config(text="Failed to load image")

    def start_generation(self):
        """Start the wave function collapse generation"""
        if self.source_image is None:
            messagebox.showwarning("Warning", "Please load a source image first")
            return
            
        if self.is_generating:
            return
            
        # Get parameters
        pattern_size = self.pattern_size_var.get()
        output_width = self.output_width_var.get()
        output_height = self.output_height_var.get()
        
        # Validate parameters
        if (self.source_image.shape[0] < pattern_size or 
            self.source_image.shape[1] < pattern_size):
            messagebox.showerror("Error", 
                               f"Source image too small for pattern size {pattern_size}")
            return
            
        # Update UI state
        self.is_generating = True
        self.generate_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # -- NEW: Show pattern progress bar, hide main one --
        self.pattern_progress_var.set(0)
        self.status_label.config(text="Extracting patterns...")
        self.pattern_progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        self.pattern_progress_label.grid(row=0, column=1)
        self.progress_bar.grid_remove()
        self.progress_label.grid_remove()

        # Clear log and start capturing
        self.clear_log()
        self.log_capture.start_capture()
        
        # Start generation in separate thread
        self.generation_thread = threading.Thread(
            target=self.run_generation,
            args=(pattern_size, output_width, output_height)
        )
        self.generation_thread.daemon = True
        self.generation_thread.start()

    def run_generation(self, pattern_size, output_width, output_height):
        """Run the wave function collapse generation"""
        try:
            print(f"Starting generation with pattern_size={pattern_size}, output={output_width}x{output_height}")
            print(f"Source image shape: {self.source_image.shape}")
            
            # Create WFC instance
            print("Creating WFC instance...")
            self.wfc = WaveFunctionCollapse(pattern_size, output_width, output_height)
            self.wfc.set_progress_callback(self.update_progress)
            
            # Extract patterns
            print("Starting pattern extraction...")
            # -- NEW: Pass the new callback to extract_patterns --
            self.wfc.extract_patterns(self.source_image, self.update_extraction_progress)
            print("Pattern extraction completed")
            
            # -- NEW: Switch to main generation progress bar --
            def switch_bars():
                self.pattern_progress_bar.grid_remove()
                self.pattern_progress_label.grid_remove()
                self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
                self.progress_label.grid(row=0, column=1)
                self.status_label.config(text="Generating...")
            self.root.after(0, switch_bars)

            # Start generation
            print("Starting wave function collapse...")
            result = self.wfc.collapse()
            print("Generation completed successfully")
            
            # Generation complete
            self.root.after(0, lambda: self.generation_complete(result))
            
        except Exception as e:
            print(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.generation_error(str(e)))
        finally:
            # Stop log capture
            self.log_capture.stop_capture()

    # -- NEW: Callback function for the pattern extraction progress --
    def update_extraction_progress(self, progress):
        """Update pattern extraction progress display"""
        if not self.is_generating:
            return
            
        def update_ui():
            if not self.is_generating:
                return
            self.pattern_progress_var.set(progress)
            self.pattern_progress_label.config(text=f"{progress:.1f}%")

        self.root.after(0, update_ui)

    def update_progress(self, progress, steps, current_state):
        # ... (This function remains the same as the previously updated version) ...
        """Update progress display (called from generation thread)"""
        if not self.is_generating:
            return
            
        def update_ui():
            if not self.is_generating:
                return
                
            self.progress_var.set(progress)
            self.progress_label.config(text=f"{progress:.1f}%")
            self.status_label.config(text=f"Generating... Step {steps}")
            
            # --- REAL-TIME UPDATE LOGIC ---
            # Check if we have valid image data to display
            if current_state is not None and current_state.size > 0:
                try:
                    # Clear the previous image
                    self.ax_output.clear()
                    
                    # Display the current state of the generation
                    self.ax_output.imshow(current_state, cmap='gray', vmin=0, vmax=255)
                    self.ax_output.set_title(f"Generating... ({progress:.1f}%)")
                    self.ax_output.axis('off')
                    
                    # Redraw the canvas to show the update
                    self.canvas.draw()
                    self.canvas.flush_events()
                except Exception as e:
                    # This might happen if the GUI is closing, so we can ignore it
                    pass
                
        self.root.after(0, update_ui)
        
    # ... (generation_complete, generation_error, stop_generation, and run methods remain the same) ...
    def generation_complete(self, result):
        """Handle generation completion"""
        self.is_generating = False
        self.generate_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text=f"Generation complete! ({self.wfc.generation_steps} steps)")
        
        # Show final result
        if result is not None and result.size > 0:
            try:
                # Clear the output area
                self.ax_output.clear()
                
                # Display the final result
                self.ax_output.imshow(result, cmap='gray', vmin=0, vmax=255)
                self.ax_output.set_title("Generated Output - Complete")
                self.ax_output.axis('off')
                
                # Force canvas update
                self.canvas.draw()
                self.canvas.flush_events()
                
                # Save the result
                self.current_output = result
                cv2.imwrite('generated_output.png', result)
                print("✓ Generation complete! Saved generated_output.png")
                print(f"  Result shape: {result.shape}")
                print(f"  Result range: {result.min()} - {result.max()}")
                
            except Exception as e:
                print(f"Error displaying final result: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Warning: No result generated or result is empty")
        
    def generation_error(self, error_msg):
        """Handle generation error"""
        self.is_generating = False
        self.generate_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Generation failed")
        messagebox.showerror("Error", f"Generation failed: {error_msg}")
        
    def stop_generation(self):
        """Stop the current generation"""
        if self.is_generating:
            print("Stopping generation...")
            self.is_generating = False
            
            # Stop the WFC algorithm if it exists
            if self.wfc and hasattr(self.wfc, 'stop'):
                self.wfc.stop()
            
            self.generate_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="Generation stopped")
            self.log_capture.stop_capture()
            
            # Update the log to show stop message
            self.append_log("Generation stopped by user")
            
    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()


if __name__ == "__main__":
    app = WaveFunctionCollapseGUI()
    app.run()